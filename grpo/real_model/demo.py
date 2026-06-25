"""Clean demo (REAL model): run many GRPO passes, plot loss + accuracy.

Runs the full GRPO loop over several passes so the model can improve, records the
training loss and group accuracy per pass, and saves the curves to
performance_graphs.png in this folder.

Edit tasks and hyperparameters in config.py; this file just runs the loop.

Usage:
    python demo.py                          # tasks/tuning from config.py
    python demo.py "7 + 5 - 3" "2 + 3 * 4"  # ADD these equations to config's set
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
from parse_tags import parse_output                      # grpo/ (shared)
from score_rewards import RewardWeights, score_output, max_reward  # grpo/ (shared)
from compute_advantages import compute_group_advantages  # grpo/ (shared)
from plot_performance import plot_performance           # grpo/ (shared)
from show_thoughts import show_group, report_final_modes  # grpo/ (shared)
from generate_policy import RealPolicy                  # local (this folder)


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


def evaluate_generalization(policy, test_tasks, group_size, weights):
    """Read-only probe: accuracy of the CURRENT policy on each held-out task.

    Uses `policy.eval_generate` (no `train_step`, training cache restored), so the
    policy is never trained on these tasks and the training trajectory is
    unperturbed. Returns one accuracy per held-out task, aligned to `test_tasks`.
    """
    accs = []
    for task in test_tasks:
        outputs = policy.eval_generate(task.prompt, group_size, correct_answer=task.answer, steps=task.steps)
        scored = [score_output(text, task, weights) for text in outputs]
        accs.append(sum(s.correct for s in scored) / len(scored))
    return accs


def run(equations, passes, group_size=None):
    group_size = group_size or config.GROUP_SIZE
    tasks = [equation_task(e) for e in equations]
    test_tasks = [equation_task(e) for e in config.TEST_EQUATIONS]
    eval_every = config.EVAL_EVERY
    policy = RealPolicy(
        model_name=config.MODEL_NAME, learning_rate=config.LEARNING_RATE,
        kl_coef=config.KL_COEF, max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE, top_k=config.TOP_K, micro_batch=config.MICRO_BATCH,
    )
    weights = RewardWeights()
    folder = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(folder, "run_log.txt")
    print(f"Tasks: {[t.label for t in tasks]}")
    print(f"Held-out (generalization) tasks: {[t.label for t in test_tasks]}")
    print(f"Running {passes} GRPO passes (real model); per-output thoughts -> {log_path}")

    losses, rewards, accuracies = [], [], []
    # Generalization probe: pass numbers sampled, and one accuracy curve per held-out task.
    gen_passes, gen_curves = [], [[] for _ in test_tasks]
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

            # Every `eval_every` passes, probe generalization on the held-out tasks.
            if test_tasks and (p + 1) % eval_every == 0:
                gen_accs = evaluate_generalization(policy, test_tasks, group_size, weights)
                gen_passes.append(p + 1)
                for ti, a in enumerate(gen_accs):
                    gen_curves[ti].append(a)
                mean_gen = sum(gen_accs) / len(gen_accs)
                print(f"  [generalization @ pass {p + 1}] held-out accuracy {mean_gen:.3f}  ({len(test_tasks)} expressions)")
                print(f"  [generalization @ pass {p + 1}] mean held-out accuracy {mean_gen:.3f}", file=log)

    report_final_modes(tasks, final_answers)

    out_path = os.path.join(folder, "performance_graphs.png")
    plot_performance(losses, rewards, accuracies, out_path,
                     title="GRPO (real model) performance", max_reward=max_reward(weights),
                     gen_passes=gen_passes, gen_curves=gen_curves, gen_labels=config.TEST_EQUATIONS)
    return losses, rewards, accuracies


if __name__ == "__main__":
    equations, passes, group = parse_cli(sys.argv[1:])
    run(equations, passes, group)
