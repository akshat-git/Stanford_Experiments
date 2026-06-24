r"""Entry point (run this file): the REAL-MODEL GRPO loop.

This mirrors ../test/grpo_test.py one-for-one -- same files, same function names,
same flow -- but swaps the simulated policy for a real torch+transformers model
and adds the gradient update GRPO actually performs. Read the test/ version first,
then this, to see exactly what changes when the model becomes real.

Files (the four concept modules live loose in grpo/ and are shared with the test/ folder):

    ../parse_tags.py          pull <think>/<answer> text out of an output
    ../define_tasks.py        Task interface + arithmetic example ("specific tasks")
    ../score_rewards.py       score one output (format + accuracy + thought)  [imports parse_tags]
    ../compute_advantages.py  the GRPO step: reward -> group-relative advantage
    ./generate_policy.py      the REAL model (torch+transformers) that generates + trains
    ./grpo_test.py            this file -- orchestration and demo

What differs from ../test/grpo_test.py:
    * demo() builds RealPolicy() (loads a model) instead of MockPolicy().
    * run_grpo_on_task() has one extra step (5): policy.train_step(advantages),
      a real policy-gradient update. Everything else is identical.

STEP-BY-STEP, what happens when you run `python grpo_test.py`:
    1. RealPolicy(...)                  [generate_policy]       load model + tokenizer + optimizer
    2. for each GRPO step, per task -> run_grpo_on_task(task, policy):
         a. policy.generate(...)        [generate_policy]       sample a GROUP of M completions
         b. score_output(...) per item  [../score_rewards]      parse tags + check accuracy
         c. compute_group_advantages()  [../compute_advantages] reward -> relative advantage
         d. _print_group(...)                                   show reinforce/suppress
         e. policy.train_step(adv)      [generate_policy]       ONE real gradient update

Libraries: torch + transformers (in generate_policy) on top of the shared stdlib modules.
"""

import os
import sys

# Dependency: make the shared concept modules (one level up, in grpo/) importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from define_tasks import arithmetic_task               # grpo/ (shared)
from score_rewards import RewardWeights, score_output  # grpo/ (shared)
from compute_advantages import compute_group_advantages  # grpo/ (shared)
from generate_policy import RealPolicy                 # local (this folder)


def run_grpo_on_task(task, policy, group_size=6, weights=None, correct_answer=0, train=True, verbose=True):
    """One full GRPO pass over a single task. Returns (scored_group, loss).

    Identical to the test/ version through step 4; step 5 is the real update:
      1. policy.generate(...)        -> sample a GROUP of completions.
      2. score_output(...) per item  -> reward each (format + accuracy + thought).
      3. compute_group_advantages()  -> rewards become group-relative advantages.
      4. _print_group(...)           -> show who is reinforced vs suppressed.
      5. policy.train_step(adv)      -> ONE policy-gradient update (real model only).
    """
    weights = weights or RewardWeights()

    outputs = policy.generate(task.prompt, group_size, correct_answer=correct_answer)   # 1
    scored = [score_output(text, task, weights) for text in outputs]                    # 2
    mean, std = compute_group_advantages(scored)                                        # 3

    if verbose:
        _print_group(task, scored, mean, std)                                           # 4

    loss = None
    if train:
        loss = policy.train_step([s.advantage for s in scored])                         # 5
        if verbose:
            print(f"  train_step loss: {loss:.4f}")
    return scored, loss


def _print_group(task, scored, mean, std):
    """Pretty-print one group's outputs, rewards, and advantages."""
    print(f"\nTASK: {task.prompt}   (correct: {task.label})")
    print(f"  group reward mean={mean:.2f}  std={std:.2f}")
    print(f"  {'#':>2} {'fmt':>4} {'acc':>4} {'thk':>4} {'reward':>7} {'adv':>6}  direction")
    for i, s in enumerate(scored):
        direction = "reinforce ^" if s.advantage > 0 else "suppress  v" if s.advantage < 0 else "neutral"
        print(f"  {i:>2} {s.format_score:>4.1f} {s.accuracy_score:>4.1f} "
              f"{s.thought_score:>4.1f} {s.reward:>7.2f} {s.advantage:>6.2f}  {direction}")


def demo(steps=3, group_size=4):
    """Run a few real GRPO update steps over specific arithmetic tasks."""
    policy = RealPolicy()  # the real model [generate_policy]
    weights = RewardWeights()                     # reward component weights [../score_rewards]

    # "Inputted tasks of a very specific nature": swap these for your own Tasks.
    tasks = [
        (arithmetic_task(7, 5, "+"), 12),
        (arithmetic_task(9, 4, "-"), 5),
    ]

    print("=" * 72)
    print("GRPO real-model loop -- sample a group, score it, compute relative")
    print("advantages, then apply one policy-gradient update per task.")
    print("=" * 72)
    for step in range(steps):
        print(f"\n########## GRPO step {step + 1}/{steps} ##########")
        for task, answer in tasks:
            run_grpo_on_task(task, policy, group_size=group_size, weights=weights,
                             correct_answer=answer, train=True)


if __name__ == "__main__":
    demo()
