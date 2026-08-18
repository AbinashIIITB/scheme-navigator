import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from typing import List, Dict, Any

from src.vectorstore import retrieve
from src.config import TOP_K


def evaluate_retrieval(
    questions_path: Path = BASE_DIR / "eval" / "questions.json",
    k: int = TOP_K
) -> Dict[str, Any]:
    """
    Evaluate retrieval Recall@K on the hand-written evaluation dataset.
    """
    if not questions_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_questions = len(questions)
    hits_at_k = 0
    reciprocal_ranks = []
    detailed_results = []

    print("=" * 80)
    print(f"Evaluating Retrieval System (Recall@{k}) on {total_questions} Questions")
    print("=" * 80)

    for idx, item in enumerate(questions, start=1):
        q = item["q"]
        expected_source = item["expected_source"].strip()

        # Retrieve top k documents
        results = retrieve(q, k=k)
        retrieved_sources = [r["source"] for r in results]

        # Check if expected source is in retrieved sources
        hit_rank = None
        for rank, r in enumerate(results, start=1):
            src = r["source"].lower()
            exp = expected_source.lower()
            if exp == src or exp in src or src in exp:
                hit_rank = rank
                break

        is_hit = hit_rank is not None
        if is_hit:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        status = f"✓ HIT (Rank {hit_rank})" if is_hit else "✗ MISS"
        print(f"[{idx:02d}/{total_questions:02d}] {status}")
        print(f"     Q: {q[:75]}...")
        print(f"     Expected : {expected_source}")
        print(f"     Retrieved: {retrieved_sources[:3]}")

        detailed_results.append({
            "index": idx,
            "question": q,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "is_hit": is_hit,
            "hit_rank": hit_rank
        })

    recall_at_k = hits_at_k / total_questions if total_questions > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total_questions if total_questions > 0 else 0.0

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated : {total_questions}")
    print(f"Hits @ {k}                   : {hits_at_k}")
    print(f"Recall @ {k}                 : {recall_at_k * 100:.2f}% ({recall_at_k:.4f})")
    print(f"MRR @ {k}                    : {mrr:.4f}")
    print("=" * 80)

    return {
        "total_questions": total_questions,
        "hits_at_k": hits_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "details": detailed_results
    }


if __name__ == "__main__":
    evaluate_retrieval()
