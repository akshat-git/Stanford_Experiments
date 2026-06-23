r"""
This is the follow-along reference. ../real_model/grpo_test.py mirrors it exactly,
swapping the simulated policy for a real torch+transformers model and adding a
gradient update.

Files (the four concept modules live loose in grpo/ and are shared by both folders):

    ../parse_tags.py          pull <think>/<answer> text out of an output
    ../define_tasks.py        Task interface + arithmetic example ("specific tasks")
    ../score_rewards.py       score one output (format + accuracy + thought)  [imports parse_tags]
    ../compute_advantages.py  the GRPO step: reward -> group-relative advantage
    ./generate_policy.py      the SIMULATED model that generates M outputs
    ./grpo_test.py            this file -- orchestration and demo

Dependency graph:
    ../parse_tags  ../define_tasks  ../compute_advantages   ./generate_policy   (leaves)
            \        /
        ../score_rewards
              \
            grpo_test  (this entry point, depends on grpo/ modules + local generate_policy)

STEP-BY-STEP, what happens when you run `python grpo_test.py`:

    __main__ calls demo(); inside demo(), for each task:
      1. MockPolicy(seed=42)              [generate_policy]      create the simulated model
      2. arithmetic_task(7, 5, "+")       [../define_tasks]      build a specific task (+checker)
      3. run_grpo_on_task(task, policy):
           a. policy.generate(...)        [generate_policy]      sample a GROUP of M outputs
           b. score_output(...) per item  [../score_rewards]     -> parse_output() [../parse_tags]
                                                                    + task.check()  [../define_tasks]
           c. compute_group_advantages()  [../compute_advantages] reward -> relative advantage
           d. _print_group(...)                                   show reinforce/suppress per output

Libraries: standard library only (re, random, statistics, dataclasses, typing).
"""

import os
import sys

# Dependency: make the shared concept modules (one level up, in grpo/) importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from define_tasks import arithmetic_task               # grpo/ (shared)
from score_rewards import RewardWeights, score_output  # grpo/ (shared)
from compute_advantages import compute_group_advantages  # grpo/ (shared)
from generate_policy import MockPolicy                 # local (this folder)


def run_grpo_on_task(task, policy, group_size=6, weights=None, correct_answer=0, verbose=True):
    """One full GRPO pass over a single task. Returns the scored group.

    Step by step:
      1. policy.generate(...)        -> sample a GROUP of `group_size` completions.
      2. score_output(...) per item  -> reward each (format + accuracy + thought).
      3. compute_group_advantages()  -> rewards become advantages RELATIVE to the
                                         group mean (the actual GRPO move).
      4. _print_group(...)           -> display who gets reinforced vs suppressed.
    """
    weights = weights or RewardWeights()

    outputs = policy.generate(task.prompt, group_size, correct_answer=correct_answer)   # 1
    scored = [score_output(text, task, weights) for text in outputs]                    # 2
    mean, std = compute_group_advantages(scored)                                        # 3

    if verbose:
        _print_group(task, scored, mean, std)                                           # 4
    return scored


def _print_group(task, scored, mean, std):
    """Pretty-print one group's outputs, rewards, and advantages."""
    print(f"\nTASK: {task.prompt}   (correct: {task.label})")
    print(f"  group reward mean={mean:.2f}  std={std:.2f}")
    print(f"  {'#':>2} {'fmt':>4} {'acc':>4} {'thk':>4} {'reward':>7} {'adv':>6}  direction")
    for i, s in enumerate(scored):
        direction = "reinforce ^" if s.advantage > 0 else "suppress  v" if s.advantage < 0 else "neutral"
        print(f"  {i:>2} {s.format_score:>4.1f} {s.accuracy_score:>4.1f} "
              f"{s.thought_score:>4.1f} {s.reward:>7.2f} {s.advantage:>6.2f}  {direction}")


def demo():
    """Run GRPO scoring over a few specific arithmetic tasks."""
    policy = MockPolicy(seed=42)          # the simulated model [generate_policy]
    weights = RewardWeights()             # reward component weights [../score_rewards]

    # "Inputted tasks of a very specific nature": swap these for your own Tasks.
    tasks = [
        (arithmetic_task(7, 5, "+"), 12),
        (arithmetic_task(9, 4, "-"), 5),
        (arithmetic_task(6, 3, "*"), 18),
    ]

    print("=" * 72)
    print("GRPO concept test (SIMULATED policy) -- each task samples a group, scores")
    print("it, and ranks outputs by their advantage relative to the group average.")
    print("=" * 72)
    for task, answer in tasks:
        run_grpo_on_task(task, policy, group_size=6, weights=weights, correct_answer=answer)


if __name__ == "__main__":
    demo()
