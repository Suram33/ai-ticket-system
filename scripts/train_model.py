"""Train the ticket solution model and print cross-validation accuracy.

Usage:  python -m scripts.train_model [path/to/tickets.csv]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.ml.predictor import SolutionPredictor, train  # noqa: E402


def main() -> None:
    settings = Settings()
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.training_data_path
    bundle = train(data, settings.model_path, evaluate=True)
    print(f"Model saved to {settings.model_path}")
    print(json.dumps(bundle["metadata"], indent=2))

    predictor = SolutionPredictor(settings.model_path, data, settings.ml_confidence_threshold).load()
    examples = [
        ("Unable to log in", "I forgot my password and cannot access my account."),
        ("Charged two times", "My credit card shows the same payment twice."),
        ("Package not here", "My parcel is late, where is it?"),
        ("App crash", "The app closes as soon as I open it on my phone."),
    ]
    print("\nSample predictions:")
    for subject, description in examples:
        p = predictor.predict(subject, description)
        print(f"- {subject!r} -> {p.solution} (confidence {p.confidence:.2f}, category {p.category})")


if __name__ == "__main__":
    main()
