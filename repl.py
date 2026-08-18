"""
repl.py — Interactive command-line interface for the Scheme Navigator.

A quick way to test the RAG pipeline without starting the Streamlit UI
or the FastAPI server. Great for development and debugging.

Usage:
    python repl.py

Type a question and press Enter to get an answer with source citations.
Type 'quit' or press Ctrl-C to exit.
"""
import sys
from pathlib import Path

# Ensure src/ is importable when run from the project root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.generator import answer


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🏛️  Sarkari Scheme Navigator — REPL CLI             ║
║  Ask questions about Indian Government schemes in plain text. ║
║  Type  'quit'  or press  Ctrl-C  to exit.                    ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_repl():
    """Start the interactive question-answer loop."""
    print(BANNER)

    while True:
        try:
            query = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not query:
            print("  ⚠  Please enter a question.\n")
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye! 👋")
            break

        print("\n⏳ Searching official scheme archives...\n")
        result = answer(query)

        print("─" * 64)
        print(f"Answer:\n{result['answer']}")
        print()

        if result["abstained"]:
            print(f"⚠  System abstained (confidence {result['confidence']:.2f} below threshold).")
        elif result["sources"]:
            print(f"📚 Sources (confidence: {result['confidence']:.2%}):")
            for i, src in enumerate(result["sources"], start=1):
                print(f"  {i}. {src['source']}  |  Page {src['page']}  |  Score {src['score']:.4f}")
                if src.get("excerpt"):
                    # Show a short preview of the matched clause
                    excerpt = src["excerpt"][:120] + ("..." if len(src["excerpt"]) > 120 else "")
                    print(f"     \"{excerpt}\"")

        print("─" * 64)
        print()


if __name__ == "__main__":
    run_repl()