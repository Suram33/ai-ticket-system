"""RAG retrieval + LLM expansion, and that knowledge-base content drives the reply."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.embeddings import TfidfEmbedder
from app.rag.knowledge_base import KnowledgeBase, chunk_markdown
from app.rag.pipeline import compose_fallback, grounding_check

from .conftest import ROOT, FakeLLM, make_services


def test_markdown_is_chunked_by_section():
    chunks = chunk_markdown(ROOT / "data" / "knowledge_base" / "account_access.md")
    assert [c.section for c in chunks][:2] == ["Resetting a forgotten password", "Temporary account lock"]
    assert all(c.text for c in chunks)


def test_retrieval_finds_relevant_section():
    kb = KnowledgeBase(ROOT / "data" / "knowledge_base", TfidfEmbedder()).build()
    cases = {
        "forgot password reset account recovery": "Resetting a forgotten password",
        "charged twice duplicate payment refund": "Duplicate or double charges",
        "app keeps crashing update the app clear cache": "Mobile app crashing or freezing",
        "package late where is my order tracking": "Tracking an order",
    }
    for query, section in cases.items():
        assert kb.search(query, top_k=1)[0].chunk.section == section, query


def test_llm_receives_ticket_solution_and_retrieved_kb(client, fake_llm, ticket_payload):
    t = client.post("/api/tickets", json=ticket_payload).json()
    prompt = fake_llm.calls[-1]["prompt"]
    system = fake_llm.calls[-1]["system"]
    assert "Unable to log in" in prompt                                         # ticket details
    assert "Reset the password using the account recovery process." in prompt  # ML solution
    assert "Forgot Password option on the login page" in prompt                 # retrieved KB text
    assert "ONLY the facts in the KNOWLEDGE BASE" in system                     # grounding instruction
    assert t["response_source"] == "llm"
    assert t["response_model"] == "fake:test-llm"
    assert t["rag_sources"][0]["section"] == "Resetting a forgotten password"
    assert t["ai_response"].startswith("We're sorry for the trouble.")  # the LLM's text is what gets saved


def test_llm_expands_one_line_solution_into_detailed_reply(tmp_path, trained_model, ticket_payload):
    # A stand-in LLM that writes its reply from the KB text it was given in the prompt.
    def write_from_prompt(prompt: str) -> str:
        kb_text = prompt.split("KNOWLEDGE BASE")[1].split("\n")[2]
        return f"Hi Jane,\n\nWe're sorry you're having trouble accessing your account. {kb_text}\n\nThe Support Team"

    services = make_services(tmp_path, trained_model, FakeLLM(reply=write_from_prompt))
    with TestClient(create_app(Settings(), services=services)) as c:
        t = c.post("/api/tickets", json=ticket_payload).json()
    assert len(t["ai_response"]) > 3 * len(t["predicted_solution"])
    assert "Forgot Password option" in t["ai_response"]


def test_knowledge_base_content_changes_the_response(tmp_path, trained_model, ticket_payload):
    """Same ticket, two different knowledge bases -> the KB fact shows up in the reply."""
    kb_a = tmp_path / "kb_a"
    kb_a.mkdir()
    (kb_a / "pw.md").write_text("# Passwords\n\n## Password reset\nTo reset a password, open the PurpleKey portal and press Recover. The link expires after 15 minutes.\n")
    kb_b = tmp_path / "kb_b"
    kb_b.mkdir()
    (kb_b / "pw.md").write_text("# Passwords\n\n## Password reset\nTo reset a password, call the GreenDesk hotline and confirm your date of birth.\n")

    replies = {}
    for name, kb_dir in (("a", kb_a), ("b", kb_b)):
        services = make_services(tmp_path / name, trained_model, llm=None, kb_dir=kb_dir)  # template generator
        with TestClient(create_app(Settings(), services=services)) as c:
            replies[name] = c.post("/api/tickets", json=ticket_payload).json()["ai_response"]
    assert "PurpleKey portal" in replies["a"] and "GreenDesk" not in replies["a"]
    assert "GreenDesk hotline" in replies["b"] and "PurpleKey" not in replies["b"]


def test_grounding_check_flags_invented_content():
    context = "You can reset your password using the Forgot Password option. The link is valid for 60 minutes."
    good = grounding_check("Please reset your password using the Forgot Password option. The link is valid for 60 minutes.", context)
    bad = grounding_check("The link is valid for 24 hours. We will also send you a free gift card voucher tomorrow.", context)
    assert good["unsupported"] == []
    assert len(bad["unsupported"]) == 2
    assert bad["unsupported"][0]["unsupported_numbers"] == ["24"]


def test_fallback_reply_without_kb_match_escalates():
    reply = compose_fallback({"customer_name": "Sam Lee", "subject": "Odd question"}, None, [])
    assert reply.startswith("Hi Sam,")
    assert "support specialist will review your request" in reply


def test_kb_search_endpoint(client):
    r = client.get("/api/kb/search", params={"q": "cancel subscription"})
    assert r.status_code == 200
    assert r.json()[0]["section"] == "Cancelling a subscription"


def test_response_can_be_reviewed_edited_and_approved(client, ticket_payload):
    t = client.post("/api/tickets", json=ticket_payload).json()
    edited = t["ai_response"] + "\n\nP.S. Edited by agent."
    r = client.put(f"/api/tickets/{t['id']}/response", json={"final_response": edited, "approve": False})
    assert r.json()["final_response"] == edited and r.json()["response_status"] == "draft"

    r = client.put(f"/api/tickets/{t['id']}/response", json={"final_response": edited, "approve": True})
    assert r.json()["response_status"] == "approved"
    stored = client.get(f"/api/tickets/{t['id']}").json()
    assert stored["final_response"] == edited
    assert stored["ai_response"] == t["ai_response"]  # original AI draft kept for comparison

    # Regenerating does not overwrite an approved response.
    regen = client.post(f"/api/tickets/{t['id']}/generate").json()
    assert regen["final_response"] == edited

    assert client.put(f"/api/tickets/{t['id']}/response", json={"final_response": "   "}).status_code == 422


def test_ollama_client_request_and_response(monkeypatch):
    """The real Ollama client sends a chat request and parses the reply (HTTP layer mocked)."""
    import app.rag.llm as llm_mod

    sent = {}

    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"role": "assistant", "content": "  Hello from the local model.  "}}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json, timeout=timeout)
        return Resp()

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    client = llm_mod.OllamaLLM("http://localhost:11434/", "llama3.2:3b", timeout=30, temperature=0.1)
    assert client.generate("system rules", "user prompt") == "Hello from the local model."
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["body"]["model"] == "llama3.2:3b" and sent["body"]["stream"] is False
    assert sent["body"]["options"]["temperature"] == 0.1
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]


def test_ollama_client_missing_model_error(monkeypatch):
    import pytest

    import app.rag.llm as llm_mod

    class Resp:
        status_code = 404
        text = "model not found"

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: Resp())
    with pytest.raises(llm_mod.LLMError, match="ollama pull"):
        llm_mod.OllamaLLM("http://localhost:11434", "llama3.2:3b").generate("s", "p")
