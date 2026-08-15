"""
Offline Local Evaluation Harness for Kaggriculture.
Executes matches between agents, benchmark performance and logs economics.
"""

import sys
import os
import time
from typing import List, Tuple

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from main import agent
from kaggle_environments import make


def run_single_match(
    opponent_name: str = "starter",
    steps: int = 720,
    debug: bool = False,
) -> Tuple[float, float, str]:
    """
    Runs a single simulation match between our agent and an opponent.
    Returns (my_score, opponent_score, outcome).
    """
    print(f"\n=======================================================")
    print(f"Starting Kaggriculture Match: [Our Agent] vs [{opponent_name}] ({steps} turns)")
    print(f"=======================================================")

    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=debug)

    t0 = time.perf_counter()
    env.run([agent, opponent_name])
    duration = time.perf_counter() - t0

    final_step = env.steps[-1]
    my_reward = final_step[0].reward or 0.0
    opp_reward = final_step[1].reward or 0.0
    status_0 = final_step[0].status
    status_1 = final_step[1].status

    outcome = "WIN" if my_reward > opp_reward else ("TIE" if my_reward == opp_reward else "LOSS")

    print(f"Match Finished in {duration:.2f} seconds ({steps / max(0.001, duration):.1f} turns/sec)")
    print(f"Player 0 (Our Agent): Bank = ${my_reward:,.0f} | Status = {status_0}")
    print(f"Player 1 ({opponent_name}):  Bank = ${opp_reward:,.0f} | Status = {status_1}")
    print(f"Result: {outcome}")

    return my_reward, opp_reward, outcome


def run_benchmark_series(
    opponent_name: str = "starter",
    num_matches: int = 5,
    steps: int = 720,
) -> None:
    """
    Runs a tournament series of N matches and prints summary statistics.
    """
    print(f"\n--- Running Benchmark Series ({num_matches} games vs {opponent_name}) ---")
    my_scores: List[float] = []
    opp_scores: List[float] = []
    wins = 0
    ties = 0
    losses = 0

    for i in range(1, num_matches + 1):
        print(f"\nGame {i}/{num_matches}:")
        s0, s1, outcome = run_single_match(opponent_name=opponent_name, steps=steps, debug=False)
        my_scores.append(s0)
        opp_scores.append(s1)
        if outcome == "WIN":
            wins += 1
        elif outcome == "TIE":
            ties += 1
        else:
            losses += 1

    avg_my = sum(my_scores) / len(my_scores)
    avg_opp = sum(opp_scores) / len(opp_scores)
    win_pct = (wins / num_matches) * 100.0

    print(f"\n================ BENCHMARK SUMMARY ================")
    print(f"Opponent: {opponent_name} | Total Games: {num_matches}")
    print(f"Record: {wins} W - {losses} L - {ties} T (Winrate: {win_pct:.1f}%)")
    print(f"Our Avg Bank: ${avg_my:,.1f} (Min: ${min(my_scores):,.0f}, Max: ${max(my_scores):,.0f})")
    print(f"Opp Avg Bank: ${avg_opp:,.1f} (Min: ${min(opp_scores):,.0f}, Max: ${max(opp_scores):,.0f})")
    print(f"Net Advantage: ${avg_my - avg_opp:+,.1f}")
    print(f"====================================================")


if __name__ == "__main__":
    opp = sys.argv[1] if len(sys.argv) > 1 else "starter"
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 720

    if n_games == 1:
        run_single_match(opponent_name=opp, steps=steps, debug=True)
    else:
        run_benchmark_series(opponent_name=opp, num_matches=n_games, steps=steps)
