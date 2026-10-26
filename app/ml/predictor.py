"""ML model that predicts a one-line solution (and a category) for a ticket.

Model choice
------------
The training data is a small labelled set of tickets (subject + description)
where every ticket is mapped to one of a fixed set of one-line solutions.
That makes this a short-text *classification* problem: each distinct solution
is a class.

* Features: TF-IDF over word 1-2 grams (captures phrases like "charged twice")
  combined with TF-IDF over character 2-5 grams (robust to typos and word
  forms like "login" / "log in" / "logging").
* Classifier: multinomial Logistic Regression. It trains in milliseconds,
  works well on small text datasets, is easy to explain, and gives calibrated-
  enough probabilities that we use as a confidence score.

If the best class has a probability below ML_CONFIDENCE_THRESHOLD, we do not
trust the guess and return a safe "escalate" solution instead.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

FALLBACK_SOLUTION = "Escalate the ticket to a support specialist for manual review."
REQUIRED_COLUMNS = {"subject", "description", "category", "solution"}


@dataclass
class Prediction:
    solution: str
    confidence: float
    category: str | None
    category_confidence: float | None
    low_confidence: bool = False
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def ticket_text(subject: str | None, description: str | None) -> str:
    """Text the model sees. The subject is repeated so it weighs a bit more."""
    subject = (subject or "").strip()
    description = (description or "").strip()
    return f"{subject}. {subject}. {description}".strip(" .")


def load_training_data(path: Path) -> tuple[list[str], list[str], list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Training file {path} is missing columns: {sorted(missing)}")
        rows = [r for r in reader if (r["subject"] or r["description"]) and r["solution"]]
    if len(rows) < 10:
        raise ValueError(f"Training file {path} needs at least 10 labelled rows, found {len(rows)}")
    texts = [ticket_text(r["subject"], r["description"]) for r in rows]
    return texts, [r["solution"].strip() for r in rows], [r["category"].strip() for r in rows]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=10.0, max_iter=5000, class_weight="balanced")),
    ])


def train(training_path: Path, model_path: Path | None = None, evaluate: bool = True) -> dict:
    """Train the solution + category models and optionally save them. Returns the bundle."""
    texts, solutions, categories = load_training_data(training_path)

    metrics: dict = {}
    if evaluate:
        # Cross-validation gives an honest accuracy estimate on unseen tickets.
        min_class = min(np.unique(solutions, return_counts=True)[1])
        folds = max(2, min(5, int(min_class)))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(build_pipeline(), texts, solutions, cv=cv)
        metrics = {"cv_folds": folds, "cv_accuracy_mean": round(float(scores.mean()), 3),
                   "cv_accuracy_std": round(float(scores.std()), 3)}

    solution_model = build_pipeline().fit(texts, solutions)
    category_model = build_pipeline().fit(texts, categories)
    bundle = {
        "solution_model": solution_model,
        "category_model": category_model,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_file": str(training_path),
            "n_samples": len(texts),
            "n_solutions": len(set(solutions)),
            "categories": sorted(set(categories)),
            **metrics,
        },
    }
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
    return bundle


class SolutionPredictor:
    """Loads (or trains on first run) the model and serves predictions."""

    def __init__(self, model_path: Path, training_path: Path, confidence_threshold: float = 0.2):
        self.model_path = Path(model_path)
        self.training_path = Path(training_path)
        self.confidence_threshold = confidence_threshold
        self.bundle: dict | None = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self.bundle is not None

    @property
    def metadata(self) -> dict:
        return self.bundle["metadata"] if self.bundle else {}

    def load(self) -> "SolutionPredictor":
        try:
            if self.model_path.exists():
                self.bundle = joblib.load(self.model_path)
                log.info("Loaded ML model from %s", self.model_path)
            else:
                log.info("No saved model at %s, training from %s", self.model_path, self.training_path)
                self.bundle = train(self.training_path, self.model_path, evaluate=False)
            self.error = None
        except Exception as exc:  # model must never take the app down
            log.exception("ML model could not be loaded or trained")
            self.bundle = None
            self.error = f"ML model unavailable: {exc}"
        return self

    def retrain(self) -> dict:
        self.bundle = train(self.training_path, self.model_path, evaluate=True)
        self.error = None
        return self.metadata

    def predict(self, subject: str | None, description: str | None) -> Prediction:
        if not self.available:
            raise RuntimeError(self.error or "ML model is not loaded")
        text = ticket_text(subject, description)
        if len(text.replace(".", "").strip()) < 3:
            raise ValueError("Ticket has no subject or description to analyse")

        sol_model = self.bundle["solution_model"]
        probs = sol_model.predict_proba([text])[0]
        order = np.argsort(probs)[::-1]
        classes = sol_model.classes_
        best, confidence = str(classes[order[0]]), float(probs[order[0]])
        alternatives = [{"solution": str(classes[i]), "confidence": round(float(probs[i]), 3)} for i in order[:3]]

        cat_model = self.bundle["category_model"]
        cat_probs = cat_model.predict_proba([text])[0]
        cat_idx = int(np.argmax(cat_probs))

        low = confidence < self.confidence_threshold
        return Prediction(
            solution=FALLBACK_SOLUTION if low else best,
            confidence=round(confidence, 3),
            category=str(cat_model.classes_[cat_idx]),
            category_confidence=round(float(cat_probs[cat_idx]), 3),
            low_confidence=low,
            alternatives=alternatives,
        )
