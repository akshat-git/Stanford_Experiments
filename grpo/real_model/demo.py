"""Clean demo (REAL model): run many GRPO passes, plot loss + accuracy.

Runs the full GRPO loop over several passes so the model can improve, records the
training loss and group accuracy per pass, and saves the curves to
performance_graphs.png in this folder.

Edit tasks and hyperparameters in config.py; this file just runs the loop.

Usage:
    python demo.py                          # tasks/tuning from config.py
    python demo.py "7 + 5 - 3" "2 + 3 * 4"  # override the equations
    python demo.py --passes 60 "6 * 7 - 1"  # override passes
    python demo.py --group 24 "6 * 7 - 1"   # override outputs sampled per task

Per pass, for each task: generate a group -> score each output -> compute
group-relative advantages -> train_step (gradient update) -> record loss/accuracy.
"""

import os
import sys

# Dependency: make the shared concept modules (one level up, in grpo/) importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                           # local user-editable settings
from define_tasks import equation_task                  # grpo/ (shared)
from score_rewards import RewardWeights, score_output, max_reward  # grpo/ (shared)
from compute_advantages import compute_group_advantages  # grpo/ (shared)
from plot_performance import plot_performance           # grpo/ (shared)
from show_thoughts import show_group                    # grpo/ (shared)
from generate_policy import RealPolicy                  # local (this folder)


def parse_cli(argv):
    """Defaults from config.py; `--passes N`, `--group N`, positional equations override."""
    passes, group, equations, i = config.PASSES, config.GROUP_SIZE, [], 0
    while i < len(argv):
        if argv[i] == "--passes":
            passes, i = int(argv[i + 1]), i + 2
        elif argv[i] == "--group":
            group, i = int(argv[i + 1]), i + 2
        else:
            equations.append(argv[i]); i += 1
    return (equations or config.EQUATIONS), passes, group


def run(equations, passes, group_size=None):
    group_size = group_size or config.GROUP_SIZE
    tasks = [equation_task(e) for e in equations]
    policy = RealPolicy(
        model_name=config.MODEL_NAME, learning_rate=config.LEARNING_RATE,
        kl_coef=config.KL_COEF, max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE, top_k=config.TOP_K, micro_batch=config.MICRO_BATCH,
    )
    weights = RewardWeights()
    folder = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(folder, "run_log.txt")
    print(f"Tasks: {[t.label for t in tasks]}")
    print(f"Running {passes} GRPO passes (real model); per-output thoughts -> {log_path}")

    losses, rewards, accuracies = [], [], []
    with open(log_path, "w", encoding="utf-8") as log:
        for p in range(passes):
            print(f"\n===== pass {p + 1}/{passes} =====", file=log)
            pass_losses, pass_rewards, pass_accs = [], [], []
            for task in tasks:
                # generate -> score -> advantages -> update, recording loss/reward/accuracy.
                outputs = policy.generate(task.prompt, group_size, correct_answer=task.answer)
                scored = [score_output(text, task, weights) for text in outputs]
                compute_group_advantages(scored)
                show_group(task, scored, stream=log)  # detailed thoughts -> log file
                pass_losses.append(policy.train_step([s.advantage for s in scored]))
                pass_rewards.append(sum(s.reward for s in scored) / len(scored))
                pass_accs.append(sum(s.correct for s in scored) / len(scored))

            losses.append(sum(pass_losses) / len(pass_losses))
            rewards.append(sum(pass_rewards) / len(pass_rewards))
            accuracies.append(sum(pass_accs) / len(pass_accs))
            log.flush()
            # terminal: just the metrics.
            print(f"pass {p + 1:>3}/{passes}   loss {losses[-1]:.4f}   reward {rewards[-1]:.3f}   accuracy {accuracies[-1]:.3f}")

    out_path = os.path.join(folder, "performance_graphs.png")
    plot_performance(losses, rewards, accuracies, out_path,
                     title="GRPO (real model) performance", max_reward=max_reward(weights))
    return losses, rewards, accuracies


if __name__ == "__main__":
    equations, passes, group = parse_cli(sys.argv[1:])
    run(equations, passes, group)
