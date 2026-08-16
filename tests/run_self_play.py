"""
Self-Play & Market Stress Evaluation Harness for Kaggriculture Agent v1.0.
Simulates Agent vs Agent matches to test economic elasticity, concurrency and symmetric balance.
"""

import sys
import os
import time
from typing import List, Tuple

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from main import agent
from kaggle_environments import make


def run_self_play_series(num_matches: int = 10, steps: int = 720) -> None:
    """
    Executes N matches where our agent plays against a clone of itself.
    """
    print(f"\n=======================================================")
    print(f"Starting Self-Play Stress Tournament: {num_matches} Matches (Agent v1.0 vs Agent v1.0)")
    print(f"=======================================================")

    p0_scores: List[float] = []
    p1_scores: List[float] = []
    p0_wins = 0
    p1_wins = 0
    ties = 0
    match_durations: List[float] = []

    for i in range(1, num_matches + 1):
        env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=False)

        t0 = time.perf_counter()
        env.run([agent, agent])
        duration = time.perf_counter() - t0
        match_durations.append(duration)

        final_step = env.steps[-1]
        r0 = final_step[0].reward or 0.0
        r1 = final_step[1].reward or 0.0
        s0 = final_step[0].status
        s1 = final_step[1].status

        p0_scores.append(r0)
        p1_scores.append(r1)

        if r0 > r1:
            p0_wins += 1
            winner_str = "Player 0 WIN"
        elif r1 > r0:
            p1_wins += 1
            winner_str = "Player 1 WIN"
        else:
            ties += 1
            winner_str = "TIE"

        print(f"Match {i:02d}/{num_matches:02d}: P0 = ${r0:,.0f} | P1 = ${r1:,.0f} | Result: {winner_str} ({duration:.2f}s, {steps/duration:.0f} turns/s)")

    avg_p0 = sum(p0_scores) / len(p0_scores)
    avg_p1 = sum(p1_scores) / len(p1_scores)
    avg_both = (avg_p0 + avg_p1) / 2.0
    total_time = sum(match_durations)
    avg_speed = (num_matches * steps) / total_time
    ms_per_turn = (total_time / (num_matches * steps)) * 1000.0

    print(f"\n=======================================================")
    print(f"                 SELF-PLAY TOURNAMENT REPORT           ")
    print(f"=======================================================")
    print(f"Total Matches Played: {num_matches} (720 turns each)")
    print(f"Record: P0: {p0_wins} Wins | P1: {p1_wins} Wins | Ties: {ties}")
    print(f"Player 0 Avg Bank: ${avg_p0:,.1f} (Min: ${min(p0_scores):,.0f}, Max: ${max(p0_scores):,.0f})")
    print(f"Player 1 Avg Bank: ${avg_p1:,.1f} (Min: ${min(p1_scores):,.0f}, Max: ${max(p1_scores):,.0f})")
    print(f"Combined Mean Bank: ${avg_both:,.1f}")
    print(f"Symmetry Discrepancy (P0 vs P1): {abs(avg_p0 - avg_p1):.1f} coins ({abs(avg_p0 - avg_p1)/avg_both * 100:.2f}%)")
    print(f"Computational Throughput: {avg_speed:.1f} turns/second (~{ms_per_turn:.2f} ms/turn)")
    print(f"=======================================================\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_self_play_series(num_matches=n)
