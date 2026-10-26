"""RAG workflow: retrieve knowledge-base content, then have the local LLM
expand the ML one-line solution into a detailed, customer-friendly reply.

1. Build a search query from the ticket subject, description and the ML solution.
2. Retrieve the top-k most similar knowledge-base chunks (cosine similarity).
3. Put the ticket, the predicted solution and the retrieved chunks in a prompt
   that tells the LLM to use only that content.
4. Ask the local LLM (Ollama) for the reply.
5. Run a simple grounding check that flags sentences (and numbers) in the reply
   that are not supported by the retrieved content, so the agent can review them.

If the LLM is disabled, not running or fails, a deterministic fallback builds
the reply from the retrieved knowledge-base sentences themselves, so the agent
always gets a grounded draft and the ticket flow never breaks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from .knowledge_base import KnowledgeBase, SearchResult
from .llm import OllamaLLM

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a customer support assistant writing a reply that a support agent will review before sending.

Rules:
- Use ONLY the facts in the KNOWLEDGE BASE section and the ticket. Do not invent steps, policies, time frames, prices, links, phone numbers or email addresses that are not written there.
- The SUGGESTED SOLUTION comes from a prediction model. Follow it only where the knowledge base supports it.
- If the knowledge base does not cover the customer's problem, say that a support specialist will review the request and follow up. Do not guess.
- Start with one short, empathetic sentence. Then explain what to do; use a short numbered list when there are several steps.
- Address the customer by first name. Be warm, clear and concise (under 180 words).
- Do not use placeholders such as [Your Name]. Sign off as "The Support Team".
- Output only the reply text."""

STOPWORDS = set("""a an and are as at be been but by can could did do does for from has have if in into is it its
of on or our please should so than that the their them then there these they this to up was we were what when
which will with you your yours we're you're don't can't it's i'm i've also any all more just us how here""".split())


# Polite, non-factual words that any support reply may use without KB backing.
COURTESY_WORDS = set("""sorry apologize apologies inconvenience happy glad help helping assist assistance thank thanks
contacting reaching reach issue issues continues continue persists reply message support team further trouble
understand frustrating hope regards best sincerely feel free experiencing having problem concern questions
need know let following steps resolve resolved quickly soon look forward great""".split())

# Rewrites knowledge-base phrasing (written about "customers") into a reply to the customer.
SECOND_PERSON = [
    (r"\b[Cc]ustomers'", "your"), (r"\bthe customer's\b", "your"), (r"\bcustomer's\b", "your"),
    (r"\bCustomers\b", "You"), (r"\bcustomers\b", "you"), (r"\bThe customer\b", "You"), (r"\bthe customer\b", "you"),
    (r"\btheir\b", "your"), (r"\bTheir\b", "Your"), (r"\bthey\b", "you"), (r"\bThey\b", "You"),
    (r"\bA customer\b", "You"), (r"\ba customer\b", "you"), (r"\bthem\b", "you"),
    (r"\byou (receive|prefer|need|want|see|get|use)s\b", r"you \1"), (r"\byou (no longer )?has\b", r"you \1have"), (r"\byou is\b", "you are"), (r"\bYou is\b", "You are"),
]


def to_second_person(sentence: str) -> str:
    for pattern, repl in SECOND_PERSON:
        sentence = re.sub(pattern, repl, sentence)
    return sentence


@dataclass
class GenerationResult:
    response: str
    source: str                    # "llm" or "fallback"
    model: str | None
    sources: list[dict]
    error: str | None = None
    grounding: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_query(subject: str, description: str, solution: str | None) -> str:
    return " ".join(x for x in [subject, description, solution or ""] if x).strip()


def build_prompt(ticket: dict, solution: str | None, results: list[SearchResult]) -> str:
    if results:
        kb = "\n\n".join(f"[{i}] ({r.chunk.title} > {r.chunk.section})\n{r.chunk.text}" for i, r in enumerate(results, 1))
    else:
        kb = "(No relevant knowledge-base content was found.)"
    return (
        "TICKET\n"
        f"Customer name: {ticket.get('customer_name') or 'Customer'}\n"
        f"Subject: {ticket.get('subject') or '(none)'}\n"
        f"Description: {ticket.get('description') or '(none)'}\n"
        f"Category: {ticket.get('category') or 'Unknown'}\n\n"
        f"SUGGESTED SOLUTION (from ML model)\n{solution or '(none)'}\n\n"
        f"KNOWLEDGE BASE\n{kb}\n\n"
        "Write the reply to the customer now."
    )


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if w not in STOPWORDS and (len(w) > 3 or w.isdigit())}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def grounding_check(response: str, context: str) -> dict:
    """Flags reply sentences whose content words / numbers are mostly absent from the context.

    It is a cheap lexical check, not proof of correctness, so the agent still reviews the draft.
    """
    ctx_words = _words(context) | COURTESY_WORDS
    ctx_numbers = set(re.findall(r"\d+", context))
    checked, unsupported = 0, []
    for sent in _sentences(response):
        words = _words(sent)
        if len(words) < 4:  # greetings, sign-offs
            continue
        checked += 1
        support = len(words & ctx_words) / len(words)
        new_numbers = set(re.findall(r"\d+", sent)) - ctx_numbers
        if support < 0.5 or new_numbers:
            unsupported.append({"sentence": sent, "support": round(support, 2), "unsupported_numbers": sorted(new_numbers)})
    score = 1.0 if checked == 0 else round(1 - len(unsupported) / checked, 2)
    return {"score": score, "sentences_checked": checked, "unsupported": unsupported}


def compose_fallback(ticket: dict, solution: str | None, results: list[SearchResult]) -> str:
    """Grounded reply built only from retrieved KB sentences (used when the LLM is unavailable)."""
    first_name = (ticket.get("customer_name") or "there").split()[0]
    subject = (ticket.get("subject") or "your request").strip()
    lines = [f"Hi {first_name},", "",
             f"Thank you for contacting us about \"{subject}\". We're sorry for the inconvenience, and we're happy to help."]
    if solution:
        lines += ["", f"Recommended solution: {solution}"]
    if results:
        best = results[0]
        # Use the top chunk, plus any other chunk that scores almost as well.
        chosen = [best] + [r for r in results[1:] if r.score >= best.score * 0.85]
        steps = [s for r in chosen for s in _sentences(r.chunk.text)][:7]
        lines += ["", "Here is what you need to know:"] + [f"- {to_second_person(s)}" for s in steps]
        lines += ["", "If the issue continues, please reply to this message and our support team will be glad to help further."]
    else:
        lines += ["", "A support specialist will review your request and follow up with you shortly."]
    lines += ["", "Best regards,", "The Support Team"]
    return "\n".join(lines)


class ResponseGenerator:
    def __init__(self, kb: KnowledgeBase, llm: OllamaLLM | None, top_k: int = 3, min_score: float = 0.1):
        self.kb = kb
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, ticket: dict, solution: str | None) -> list[SearchResult]:
        query = build_query(ticket.get("subject", ""), ticket.get("description", ""), solution)
        return self.kb.search(query, top_k=self.top_k, min_score=self.min_score)

    def generate(self, ticket: dict, solution: str | None) -> GenerationResult:
        results = self.retrieve(ticket, solution)
        sources = [r.to_dict() for r in results]
        context = " ".join([r.chunk.text for r in results] + [solution or "", ticket.get("subject") or "",
                                                             ticket.get("description") or "", ticket.get("customer_name") or ""])
        error = None
        if self.llm is not None:
            try:
                reply = self.llm.generate(SYSTEM_PROMPT, build_prompt(ticket, solution, results))
                return GenerationResult(reply, "llm", self.llm.name, sources, None, grounding_check(reply, context))
            except Exception as exc:
                log.warning("LLM generation failed, using fallback: %s", exc)
                error = f"Local LLM unavailable, used knowledge-base template instead: {exc}"
        else:
            error = "LLM disabled (LLM_ENABLED=false); used knowledge-base template."
        reply = compose_fallback(ticket, solution, results)
        return GenerationResult(reply, "fallback", "template", sources, error, grounding_check(reply, context))
