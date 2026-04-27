"""
Eval runner for VibeFinder Agent.

Runs all eval cases through the agent and scores results.
Logs scores to Langfuse if configured, prints summary to stdout.

Run with:
    python -m backend.evals.run_evals
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.graph import run_agent
from backend.session import create_session
from backend.evals.eval_datasets import V1_PROFILES, VIBE_CASES, EDGE_CASES
from backend.langfuse_callback import log_score


def score_recommendations(result: dict, case: dict) -> dict:
    """Score a single eval case result. Returns dict of metric -> score (0-1)."""
    scores = {}
    recs = result.get("final_recommendations") or result.get("candidate_songs") or []
    profile = result.get("user_profile", {})

    # 1. Recommendation relevance: did we get any recommendations?
    scores["has_recommendations"] = 1.0 if len(recs) > 0 else 0.0

    # 2. Genre relevance: are recommended genres in the expected set?
    expected_genres = case.get("expected_top_genres", [])
    if expected_genres and recs:
        matching = sum(1 for r in recs if r.get("genre") in expected_genres)
        scores["genre_relevance"] = round(matching / len(recs), 2)
    else:
        scores["genre_relevance"] = None

    # 3. Genre avoidance: did we avoid genres we shouldn't recommend?
    avoid_genres = case.get("should_not_include_genres", [])
    if avoid_genres and recs:
        violations = sum(1 for r in recs if r.get("genre") in avoid_genres)
        scores["genre_avoidance"] = round(1 - violations / len(recs), 2)
    else:
        scores["genre_avoidance"] = None

    # 4. Average confidence score
    if recs:
        avg_conf = sum(r.get("confidence", 0.5) for r in recs) / len(recs)
        scores["avg_confidence"] = round(avg_conf, 2)
    else:
        scores["avg_confidence"] = 0.0

    # 5. Conflict detection (for conflicted vibe cases)
    if case.get("expect_conflict_detection"):
        scores["conflict_detected"] = 1.0 if result.get("conflict_detected") else 0.0

    # 6. Intent routing (for edge cases)
    expected_intent = case.get("expected_intent")
    if expected_intent:
        scores["correct_intent"] = 1.0 if result.get("intent") == expected_intent else 0.0

    # 7. Bias audit passed
    audit = result.get("bias_audit") or {}
    if isinstance(audit, dict):
        scores["bias_audit_passed"] = 1.0 if audit.get("passed") else 0.0

    # 8. No error
    scores["no_error"] = 0.0 if result.get("error") else 1.0

    return scores


def run_case(case: dict, delay: float = 1.0) -> dict:
    """Run a single eval case through the agent. Returns result + scores."""
    session_id = create_session()
    messages = case.get("messages") or [case.get("message", "")]

    result = None
    for msg in messages:
        result = run_agent(session_id, msg, result)
        time.sleep(delay)

    scores = score_recommendations(result or {}, case)
    return {"case_id": case["id"], "label": case["label"], "result": result, "scores": scores}


def run_all_evals(delay: float = 2.0) -> list[dict]:
    """Run all eval cases and return results with scores."""
    all_cases = V1_PROFILES + VIBE_CASES + EDGE_CASES
    results = []
    total = len(all_cases)

    print(f"\n{'='*60}")
    print(f"VibeFinder Agent Eval Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Running {total} cases...")
    print(f"{'='*60}\n")

    for i, case in enumerate(all_cases, 1):
        print(f"[{i}/{total}] {case['label']}...", end=" ", flush=True)
        try:
            result = run_case(case, delay=delay)
            scores = result["scores"]
            print(f"✓  recs={len(result['result'].get('final_recommendations') or [])}, "
                  f"conf={scores.get('avg_confidence', 'N/A')}, "
                  f"error={'yes' if result['result'].get('error') else 'no'}")
            results.append(result)
        except Exception as e:
            print(f"✗  FAILED: {e}")
            results.append({"case_id": case["id"], "label": case["label"], "error": str(e), "scores": {}})

    return results


def print_summary(results: list[dict]) -> None:
    """Print a human-readable summary of eval results."""
    print(f"\n{'='*60}")
    print("EVAL SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    errored = sum(1 for r in results if r.get("error") or (r.get("scores", {}).get("no_error") == 0.0))
    has_recs = sum(1 for r in results if r.get("scores", {}).get("has_recommendations") == 1.0)

    all_confidences = [r["scores"].get("avg_confidence") for r in results
                       if r.get("scores", {}).get("avg_confidence") is not None]
    avg_conf = round(sum(all_confidences) / len(all_confidences), 2) if all_confidences else 0

    genre_relevances = [r["scores"].get("genre_relevance") for r in results
                        if r.get("scores", {}).get("genre_relevance") is not None]
    avg_genre = round(sum(genre_relevances) / len(genre_relevances), 2) if genre_relevances else 0

    audit_passes = [r["scores"].get("bias_audit_passed") for r in results
                    if r.get("scores", {}).get("bias_audit_passed") is not None]
    audit_rate = round(sum(audit_passes) / len(audit_passes), 2) if audit_passes else 0

    print(f"Total cases:           {total}")
    print(f"Got recommendations:   {has_recs}/{total} ({has_recs/total*100:.0f}%)")
    print(f"Errors:                {errored}/{total}")
    print(f"Avg confidence score:  {avg_conf}")
    print(f"Genre relevance rate:  {avg_genre}")
    print(f"Bias audit pass rate:  {audit_rate}")

    print(f"\n{'─'*40}")
    print("Per-case results:")
    for r in results:
        s = r.get("scores", {})
        status = "✓" if not r.get("error") else "✗"
        recs = len((r.get("result") or {}).get("final_recommendations") or [])
        print(f"  {status} [{r['case_id']}] {r['label'][:35]:<35} "
              f"recs={recs} conf={s.get('avg_confidence', '-')} "
              f"genre_rel={s.get('genre_relevance', '-')}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    results = run_all_evals(delay=3.0)
    print_summary(results)

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # Strip non-serializable result dicts
        clean = [
            {"case_id": r["case_id"], "label": r["label"], "scores": r.get("scores", {})}
            for r in results
        ]
        json.dump(clean, f, indent=2)
    print(f"Results saved to {output_path}")
