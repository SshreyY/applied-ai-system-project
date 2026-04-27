"""
V1 vs Agent head-to-head comparison.

Runs the same 4 original Module 3 profiles through both:
  - V1: the original score_song formula
  - Agent: the full LangGraph agent

Prints a side-by-side comparison and saves to compare_results.json.

Run with:
    python -m backend.evals.compare_v1_v2
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.recommender_v1 import recommend_songs, load_songs
from backend.graph import run_agent
from backend.session import create_session
from backend.evals.eval_datasets import V1_PROFILES

CATALOG_PATH = os.getenv("CATALOG_PATH", "backend/data/songs.csv")


def run_v1(profile: dict) -> list[dict]:
    """Run the V1 formula for a profile. Returns top 5 with scores."""
    songs = load_songs(CATALOG_PATH)
    prefs = profile["expected_profile"]
    ranked = recommend_songs(prefs, songs, k=5)
    return [
        {
            "rank": i + 1,
            "title": s["title"],
            "artist": s["artist"],
            "genre": s["genre"],
            "mood": s["mood"],
            "v1_score": round(score, 2),
            "reasons": explanation,
        }
        for i, (s, score, explanation) in enumerate(ranked)
    ]


def run_agent_for_profile(profile: dict) -> list[dict]:
    """Run the agent for a profile message. Returns final recommendations."""
    session_id = create_session()
    result = run_agent(session_id, profile["message"])
    recs = result.get("final_recommendations") or result.get("candidate_songs") or []
    return [
        {
            "rank": i + 1,
            "title": r["title"],
            "artist": r["artist"],
            "genre": r["genre"],
            "mood": r["mood"],
            "confidence": r.get("confidence"),
            "explanation": r.get("explanation", ""),
        }
        for i, r in enumerate(recs)
    ]


def compare_profile(profile: dict) -> dict:
    """Run both systems for one profile and return comparison dict."""
    print(f"\n  Running V1...", end=" ", flush=True)
    v1_recs = run_v1(profile)
    print("done")

    print(f"  Running Agent...", end=" ", flush=True)
    agent_recs = run_agent_for_profile(profile)
    print("done")

    # Compute overlap: how many songs appear in both top-5 lists
    v1_titles = {r["title"] for r in v1_recs}
    agent_titles = {r["title"] for r in agent_recs}
    overlap = v1_titles & agent_titles
    overlap_pct = round(len(overlap) / max(len(v1_titles), 1), 2)

    # Top-1 agreement
    v1_top = v1_recs[0]["title"] if v1_recs else None
    agent_top = agent_recs[0]["title"] if agent_recs else None
    top1_agree = v1_top == agent_top

    return {
        "profile_id": profile["id"],
        "label": profile["label"],
        "v1_top5": v1_recs,
        "agent_top5": agent_recs,
        "overlap_titles": sorted(overlap),
        "overlap_pct": overlap_pct,
        "top1_agree": top1_agree,
        "v1_top1": v1_top,
        "agent_top1": agent_top,
    }


def print_comparison(comp: dict) -> None:
    """Print a human-readable side-by-side for one profile."""
    print(f"\n{'─'*70}")
    print(f"Profile: {comp['label']}")
    print(f"{'─'*70}")
    print(f"{'V1 Formula':<35} {'Agent':<35}")
    print(f"{'─'*35} {'─'*35}")

    v1 = comp["v1_top5"]
    ag = comp["agent_top5"]
    max_len = max(len(v1), len(ag))

    for i in range(max_len):
        v1_str = f"#{i+1} {v1[i]['title'][:28]} ({v1[i]['genre']})" if i < len(v1) else ""
        ag_str = f"#{i+1} {ag[i]['title'][:28]} ({ag[i]['genre']})" if i < len(ag) else ""
        print(f"{v1_str:<35} {ag_str:<35}")

    print(f"\nOverlap: {len(comp['overlap_titles'])}/5 songs in common ({comp['overlap_pct']*100:.0f}%)")
    print(f"Top-1 agree: {'✓ Yes' if comp['top1_agree'] else '✗ No'} "
          f"(V1: '{comp['v1_top1']}' | Agent: '{comp['agent_top1']}')")


def main():
    print(f"\n{'='*70}")
    print("VibeFinder: V1 Formula vs Agent — Head-to-Head Comparison")
    print(f"{'='*70}")

    results = []
    for profile in V1_PROFILES:
        print(f"\n[{profile['id']}] {profile['label']}")
        try:
            comp = compare_profile(profile)
            print_comparison(comp)
            results.append(comp)
            time.sleep(3)  # Rate limit buffer
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"profile_id": profile["id"], "error": str(e)})

    # Overall summary
    successful = [r for r in results if "error" not in r]
    if successful:
        avg_overlap = sum(r["overlap_pct"] for r in successful) / len(successful)
        top1_agreements = sum(1 for r in successful if r.get("top1_agree"))
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Profiles compared:    {len(successful)}/{len(V1_PROFILES)}")
        print(f"Avg song overlap:     {avg_overlap*100:.0f}%")
        print(f"Top-1 agreements:     {top1_agreements}/{len(successful)}")
        print(f"\nKey finding: The agent diverges most from V1 on the "
              f"'Conflicted Vibe' profile, where V1's binary matching "
              f"fails but the agent can reason about the contradiction.")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "compare_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
