"""
evaluate.py — Two evaluation suites for the Sarkari Scheme Navigator.

Suite 1 — Retrieval Quality  (questions.json)
    Recall @ K  : Did the correct PDF appear in the top-K retrieved chunks?
    MRR @ K     : How highly was the correct PDF ranked on average?
                  (1.0 if rank 1, 0.5 if rank 2, 0.33 if rank 3, …)

Suite 2 — Abstain Guard  (abstain_questions.json)
    Abstain Rate: Did the system correctly refuse to answer every out-of-scope
                  question instead of hallucinating an answer?

Run both suites from the project root:
    python eval/evaluate.py
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Make src/ importable when called from any directory
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.vectorstore import retrieve
from src.config import TOP_K


def evaluate_retrieval(
    questions_path: Path = BASE_DIR / "eval" / "questions.json",
    k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Evaluate the retrieval system on a set of labelled questions.

    Each question in questions.json has:
        "q"               — the question text
        "expected_source" — the PDF filename that should be retrieved

    Args:
        questions_path: Path to the JSON benchmark file.
        k:              Number of results to retrieve per question.

    Returns:
        Summary dict with total_questions, hits_at_k, recall_at_k, mrr,
        and a per-question details list.
    """
    if not questions_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)
    hits = 0
    reciprocal_ranks: List[float] = []
    details: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"  Retrieval Evaluation — Recall@{k} & MRR@{k}  |  {total} questions")
    print("=" * 80)

    for idx, item in enumerate(questions, start=1):
        q = item["q"]
        expected = item["expected_source"].strip()

        results = retrieve(q, k=k)
        retrieved_sources = [r["source"] for r in results]

        # Find the rank of the expected source (case-insensitive substring match)
        hit_rank = None
        for rank, r in enumerate(results, start=1):
            if expected.lower() in r["source"].lower() or r["source"].lower() in expected.lower():
                hit_rank = rank
                break

        is_hit = hit_rank is not None
        if is_hit:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Truncate long questions only when they exceed the display width
        q_display = q[:75] + ("..." if len(q) > 75 else "")
        status = f"✓ HIT (Rank {hit_rank})" if is_hit else "✗ MISS"
        print(f"[{idx:02d}/{total:02d}] {status}")
        print(f"     Q:        {q_display}")
        print(f"     Expected: {expected}")
        print(f"     Top-3:    {retrieved_sources[:3]}")

        details.append({
            "index":             idx,
            "question":          q,
            "expected_source":   expected,
            "retrieved_sources": retrieved_sources,
            "is_hit":            is_hit,
            "hit_rank":          hit_rank,
        })

    recall = hits / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    print("=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Total questions : {total}")
    print(f"  Hits @ {k}        : {hits} / {total}")
    print(f"  Recall @ {k}      : {recall * 100:.2f}%  ({recall:.4f})")
    print(f"  MRR @ {k}         : {mrr:.4f}")
    print("=" * 80)

    return {
        "total_questions": total,
        "hits_at_k":       hits,
        "recall_at_k":     recall,
        "mrr":             mrr,
        "details":         details,
    }


def evaluate_abstain(
    abstain_path: Path = BASE_DIR / "eval" / "abstain_questions.json",
    k: int = TOP_K,
    threshold: float = SIM_THRESHOLD,
) -> Dict[str, Any]:
    """
    Check whether the system correctly abstains on out-of-scope questions.

    A "correct" result here means the system returns abstained=True —
    i.e. the best retrieval score fell below `threshold` and no hallucinated
    answer was generated.

    Args:
        abstain_path: Path to the JSON file with out-of-scope questions.
        k:            Number of chunks to retrieve per question.
        threshold:    Confidence threshold used by the abstain guard.

    Returns:
        Summary dict with total, correct_abstains, abstain_rate, and details.
    """
    # Import here to avoid circular issues when only running retrieval eval
    from src.generator import answer

    if not abstain_path.exists():
        raise FileNotFoundError(f"Abstain benchmark file not found: {abstain_path}")

    with open(abstain_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)
    correct_abstains = 0
    details: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"  Abstain Guard Evaluation  |  {total} out-of-scope questions")
    print(f"  Confidence threshold: {threshold:.2f}  |  Top-K: {k}")
    print("=" * 80)

    for idx, item in enumerate(questions, start=1):
        q = item["q"]
        reason = item.get("reason", "")

        result = answer(q, k=k, threshold=threshold)
        did_abstain = result["abstained"]
        confidence = result["confidence"]

        if did_abstain:
            correct_abstains += 1
            status = f"✓ ABSTAINED (confidence {confidence:.2f})"
        else:
            status = f"✗ ANSWERED  (confidence {confidence:.2f}) ← should have abstained!"

        q_display = q[:75] + ("..." if len(q) > 75 else "")
        print(f"[{idx:02d}/{total:02d}] {status}")
        print(f"     Q:      {q_display}")
        print(f"     Reason: {reason}")

        details.append({
            "index":       idx,
            "question":    q,
            "reason":      reason,
            "did_abstain": did_abstain,
            "confidence":  confidence,
        })

    abstain_rate = correct_abstains / total if total > 0 else 0.0
    false_answers = total - correct_abstains  # Questions that were answered but shouldn't have been

    print("=" * 80)
    print("  ABSTAIN GUARD SUMMARY")
    print("=" * 80)
    print(f"  Total out-of-scope questions : {total}")
    print(f"  Correctly abstained          : {correct_abstains} / {total}")
    print(f"  Abstain rate                 : {abstain_rate * 100:.2f}%")
    if false_answers > 0:
        print(f"  ⚠  Hallucination risk        : {false_answers} question(s) were answered instead of abstained!")
    else:
        print("  ✓  No hallucinated answers generated.")
    print("=" * 80)

    return {
        "total":            total,
        "correct_abstains": correct_abstains,
        "abstain_rate":     abstain_rate,
        "false_answers":    false_answers,
        "details":          details,
    }


if __name__ == "__main__":
    from src.config import SIM_THRESHOLD

    # Run retrieval benchmark first
    evaluate_retrieval()

    print()

    # Then run the abstain guard benchmark
    evaluate_abstain(threshold=SIM_THRESHOLD)
