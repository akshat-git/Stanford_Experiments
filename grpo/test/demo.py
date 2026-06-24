"""Clean demo (SIMULATED policy): run many GRPO passes, plot loss + accuracy.

Runs the full GRPO loop over several passes so the policy can improve, records the
training loss and group accuracy per pass, and saves the curves to
performance_graphs.png in this folder.

Edit tasks and hyperparameters in config.py; this file just runs the loop.

Usage:
    python demo.py                          # tasks/tuning from config.py
    python demo.py "7 + 5 - 3" "2 + 3 * 4"  # ADD these equations to config's set
    python demo.py --passes 100 "6 * 7 - 1" # override passes
    python demo.py --group 24 "6 * 7 - 1"   # override outputs sampled per task

Per pass, for each task: generate a group -> score each output -> compute
group-relative advantages -> train_step (update the policy) -> record loss/accuracy.
"""

import os
import sys

# Dependency: make the shared concept modules (one level up, in grpo/) importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                           # local user-editable settings
from define_tasks import equation_task                  # grpo/ (shared)
from parse_tags import parse_output                      # grpo/ (shared)
from score_rewards import RewardWeights, score_output, max_reward  # grpo/ (shared)
from compute_advantages import compute_group_advantages  # grpo/ (shared)
from plot_performance import plot_performance           # grpo/ (shared)
from show_thoughts import show_group, report_final_modes  # grpo/ (shared)
from generate_policy import MockPolicy                  # local (this folder)


def parse_cli(argv):
    """Defaults from config.py; `--passes N`, `--group N`. Positional equations are
    ADDED to config.EQUATIONS (not replaced)."""
    passes, group, extra, i = config.PASSES, config.GROUP_SIZE, [], 0
    while i < len(argv):
        if argv[i] == "--passes":
            passes, i = int(argv[i + 1]), i + 2
        elif argv[i] == "--group":
            group, i = int(argv[i + 1]), i + 2
        else:
            extra.append(argv[i]); i += 1
    return config.EQUATIONS + extra, passes, group


def run(equations, passes, group_size=None):
    group_size = group_size or config.GROUP_SIZE
    tasks = [equation_task(e) for e in equations]
    policy = MockPolicy(
        seed=config.SEED, init_weights=config.INIT_WEIGHTS,
        learning_rate=config.LEARNING_RATE, kl_coef=config.KL_COEF,
    )
    weights = RewardWeights()
    folder = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(folder, "run_log.txt")
    print(f"Tasks: {[t.label for t in tasks]}")
    print(f"Running {passes} GRPO passes (simulated policy); per-output thoughts -> {log_path}")

    losses, rewards, accuracies = [], [], []
    final_answers = [[] for _ in tasks]  # last pass's parsed answers, per task
    with open(log_path, "w", encoding="utf-8") as log:
        for p in range(passes):
            print(f"\n===== pass {p + 1}/{passes} =====", file=log)
            pass_losses, pass_rewards, pass_accs = [], [], []
            for ti, task in enumerate(tasks):
                # generate -> score -> advantages -> update, recording loss/reward/accuracy.
                outputs = policy.generate(task.prompt, group_size, correct_answer=task.answer, steps=task.steps)
                scored = [score_output(text, task, weights) for text in outputs]
                compute_group_advantages(scored)
                show_group(task, scored, stream=log)  # detailed thoughts -> log file
                final_answers[ti] = [parse_output(t)[1] for t in outputs]
                pass_losses.append(policy.train_step([s.advantage for s in scored]))
                pass_rewards.append(sum(s.reward for s in scored) / len(scored))
                pass_accs.append(sum(s.correct for s in scored) / len(scored))

            losses.append(sum(pass_losses) / len(pass_losses))
            rewards.append(sum(pass_rewards) / len(pass_rewards))
            accuracies.append(sum(pass_accs) / len(pass_accs))
            log.flush()
            # terminal: just the metrics.
            print(f"pass {p + 1:>3}/{passes}   loss {losses[-1]:.4f}   reward {rewards[-1]:.3f}   accuracy {accuracies[-1]:.3f}")

    report_final_modes(tasks, final_answers)

    out_path = os.path.join(folder, "performance_graphs.png")
    plot_performance(losses, rewards, accuracies, out_path,
                     title="GRPO (simulated policy) performance", max_reward=max_reward(weights))
    return losses, rewards, accuracies


if __name__ == "__main__":
    equations, passes, group = parse_cli(sys.argv[1:])
    run(equations, passes, group)
