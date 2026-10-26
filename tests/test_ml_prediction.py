"""The ML model predicts a one-line solution for different ticket types."""
import pytest

from app.ml.predictor import FALLBACK_SOLUTION, SolutionPredictor, train

from .conftest import ROOT

CASES = [
    ("Unable to log in", "I forgot my password and cannot access my account.",
     "Reset the password using the account recovery process.", "Account"),
    ("Charged two times", "My credit card shows the same payment twice for one order.",
     "Verify the duplicate charge and refund the extra payment to the original payment method.", "Billing"),
    ("Where is my parcel", "My package has not arrived yet and it is late.",
     "Track the order using the tracking link and allow the standard delivery window.", "Shipping"),
    ("App crash", "The mobile app keeps crashing when I open it.",
     "Update the app to the latest version and clear the app cache.", "Technical"),
    ("Item broken", "The product arrived damaged with a cracked screen.",
     "Send photos of the damage so a free replacement or full refund can be arranged.", "Returns"),
    ("Stop my plan", "I want to cancel my subscription so I am not charged again.",
     "Cancel the subscription from the Billing section of Account Settings.", "Billing"),
]


@pytest.fixture(scope="module")
def predictor(trained_model):
    return SolutionPredictor(trained_model, ROOT / "data" / "sample_tickets.csv").load()


@pytest.mark.parametrize("subject,description,expected,category", CASES)
def test_predicts_expected_solution(predictor, subject, description, expected, category):
    p = predictor.predict(subject, description)
    assert p.solution == expected
    assert p.category == category
    assert 0 < p.confidence <= 1
    assert "\n" not in p.solution and len(p.solution) < 160  # one line


def test_different_tickets_get_different_solutions(predictor):
    solutions = {predictor.predict(s, d).solution for s, d, _, _ in CASES}
    assert len(solutions) == len(CASES)


def test_low_confidence_falls_back_to_escalation(trained_model):
    strict = SolutionPredictor(trained_model, ROOT / "data" / "sample_tickets.csv", confidence_threshold=0.99).load()
    p = strict.predict("Question", "Something about my thing")
    assert p.low_confidence is True
    assert p.solution == FALLBACK_SOLUTION
    assert p.alternatives  # the model's guesses are still available to the agent


def test_empty_text_is_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict("", "  ")


def test_training_reports_cross_validation_accuracy():
    bundle = train(ROOT / "data" / "sample_tickets.csv", model_path=None, evaluate=True)
    meta = bundle["metadata"]
    assert meta["n_samples"] >= 90
    assert meta["cv_accuracy_mean"] > 0.6


def test_predict_endpoint(client):
    r = client.post("/api/predict", json={"subject": "Forgot password", "description": "cannot sign in"})
    assert r.status_code == 200
    assert r.json()["solution"] == "Reset the password using the account recovery process."
    assert client.post("/api/predict", json={"subject": "", "description": ""}).status_code == 422
