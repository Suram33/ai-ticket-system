"""Create a few demo tickets through the full workflow and print the results.

Usage:  python -m scripts.seed_demo
Uses the same settings as the app (DATABASE_PATH, OLLAMA_MODEL, ...).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.main import build_services  # noqa: E402

DEMO = [
    {"customer_name": "Jane Doe", "subject": "Unable to log in", "description": "I forgot my password and cannot access my account.",
     "priority": "High", "custom_fields": {"contact_channel": "Email"}},
    {"customer_name": "Ravi Kumar", "subject": "Charged twice", "description": "My card shows two identical payments for order ORD-1042.",
     "priority": "High", "custom_fields": {"contact_channel": "Phone", "order_number": "ORD-1042", "order_amount": 59.99}},
    {"customer_name": "Maria Garcia", "subject": "Package is late", "description": "My order was supposed to arrive on Monday and it is still not here.",
     "custom_fields": {"contact_channel": "Chat", "order_number": "ORD-2210", "purchase_date": "2026-10-15"}},
    {"customer_name": "Tom Becker", "subject": "App crashes on startup", "description": "The Android app closes immediately after I open it.",
     "custom_fields": {"contact_channel": "Web form", "vip_customer": True}},
    {"customer_name": "Aiko Tanaka", "subject": "Item arrived broken", "description": "The ceramic lamp I ordered came with a big crack.",
     "custom_fields": {"contact_channel": "Email", "order_number": "ORD-3307"}},
]


def main() -> None:
    services = build_services(Settings())
    for data in DEMO:
        t = services["tickets"].create(dict(data))
        print("=" * 80)
        print(f"#{t['id']} {t['subject']}  [{t['category']}]")
        print(f"ML solution ({t['prediction_confidence']:.0%}): {t['predicted_solution']}")
        print("KB sources:", ", ".join(f"{s['section']} ({s['score']:.2f})" for s in t["rag_sources"] or []))
        print(f"Response by: {t['response_source']} {t['response_model'] or ''}")
        if t["generation_error"]:
            print("Note:", t["generation_error"])
        print("-" * 80)
        print(t["ai_response"])


if __name__ == "__main__":
    main()
