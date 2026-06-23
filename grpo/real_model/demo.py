"""Clean demo (REAL model): run many GRPO passes, plot loss + accuracy.

Runs the full GRPO loop over several passes so the model can improve, records the
training loss and group accuracy per pass, and saves the curves to
performance_graphs.png in this folder.

Usage:
    python demo.py                      # default equations
    python demo.py "7 + 5" "12 - 4"     # your own numerical equations
    python demo.py --passes 30 "6 * 7"  # more passes

Per pass, for each task: generate a group -> score each output -> compute
group-relative advantages -> train_step (gradient update) -> record loss/accuracy.
"""

import os
import sys

# Dependency: make the shared concept modules (one level up, in grpo/) importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from define_tasks import equation_task                  # grpo/ (shared)
from score_rewards import RewardWeights, score_output   # grpo/ (shared)
from compute_advantages import compute_group_advantages  # grpo/ (shared)
from plot_performance import plot_performance           # grpo/ (shared)
from show_thoughts import show_group                    # grpo/ (shared)
from generate_policy import RealPolicy                  # local (this folder)

DEFAULT_EQUATIONS = ["7 + 5", "9 - 4", "6 * 3"]
DEFAULT_PASSES = 15
GROUP_SIZE = 4


def parse_cli(argv):
    """Pull `--passes N` and positional equations out of argv."""
    passes, equations, i = DEFAULT_PASSES, [], 0
    while i < len(argv):
        if argv[i] == "--passes":
            passes, i = int(argv[i + 1]), i + 2
        else:
            equations.append(argv[i]); i += 1
    return (equations or DEFAULT_EQUATIONS), passes


def run(equations, passes, group_size=GROUP_SIZE):
    tasks = [equation_task(e) for e in equations]
    policy = RealPolicy(model_name="distilgpt2")
    weights = RewardWeights()
    print(f"Tasks: {[t.label for t in tasks]}")
    print(f"Running {passes} GRPO passes (real model)...")

    losses, accuracies = [], []
    for p in range(passes):
        print(f"\n===== pass {p + 1}/{passes} =====")
        pass_losses, pass_accs = [], []
        for task in tasks:
            # generate -> score -> advantages -> update, recording loss + accuracy.
            outputs = policy.generate(task.prompt, group_size, correct_answer=task.answer)
            scored = [score_output(text, task, weights) for text in outputs]
            compute_group_advantages(scored)
            show_group(task, scored)  # show each output's thinking + prediction
            pass_losses.append(policy.train_step([s.advantage for s in scored]))
            pass_accs.append(sum(s.correct for s in scored) / len(scored))

        losses.append(sum(pass_losses) / len(pass_losses))
        accuracies.append(sum(pass_accs) / len(pass_accs))
        print(f"  pass loss: {losses[-1]:.4f}   pass accuracy: {accuracies[-1]:.3f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_graphs.png")
    plot_performance(losses, accuracies, out_path, title="GRPO (real model) performance")
    return losses, accuracies


if __name__ == "__main__":
    equations, passes = parse_cli(sys.argv[1:])
    run(equations, passes)
