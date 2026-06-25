"""Shared plotting helper for the GRPO demos: loss, reward, and accuracy per pass.

Used by both folders' demo.py. matplotlib is imported lazily so the core concept
modules stay dependency-free.

Three panels, because the policy-gradient "loss" alone is a poor progress signal:
    * loss     -- the regularized GRPO surrogate -(advantage * logp) + beta*KL.
                  Advantages are z-scored per group (mean 0), so the PG part is
                  essentially the negative covariance between advantage and
                  log-prob; it shrinks toward 0 as the policy converges, leaving
                  the small KL floor. Its sign/magnitude are NOT progress signals.
    * reward   -- mean group reward. THIS is the progress signal; it rises as the
                  policy learns, and it does not collapse at convergence.
    * accuracy -- percentage of the group that is correct.
"""


def plot_performance(losses, rewards, accuracies, out_path, title="GRPO performance",
                     max_reward=None, gen_passes=None, gen_curves=None, gen_labels=None):
    """Save a three-panel figure (loss, mean reward, accuracy) vs pass.

    `max_reward` (if given) is drawn as a ceiling line on the reward panel so the
    actual reward points can be read against the best the reward system allows.

    Generalization probe (optional): `gen_passes` are the pass numbers at which
    held-out expressions were evaluated, `gen_curves` is one accuracy list per
    held-out expression (each aligned to `gen_passes`), and `gen_labels` names
    them. When given, each held-out curve is overlaid on the accuracy panel
    (dashed) so generalization can be read against the training accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passes = range(1, len(losses) + 1)
    fig, (ax_loss, ax_reward, ax_acc) = plt.subplots(1, 3, figsize=(16, 5))

    ax_loss.plot(passes, losses, marker="o", color="tab:red")
    ax_loss.set_title("GRPO loss = PG + KL\n(settles near the KL floor; not a progress signal)")
    ax_loss.set_xlabel("pass"); ax_loss.set_ylabel("loss")
    ax_loss.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_loss.grid(True, alpha=0.3)

    # Actual mean reward per pass (line + points), against the max the reward allows.
    ax_reward.plot(passes, rewards, marker="o", color="tab:green", label="mean reward")
    if max_reward is not None:
        ax_reward.axhline(max_reward, color="tab:gray", linestyle="--",
                          label=f"max possible = {max_reward:g}")
        ax_reward.set_ylim(0, max_reward * 1.12)
    ax_reward.set_title("Mean group reward\n(the progress signal -- should rise toward the max)")
    ax_reward.set_xlabel("pass"); ax_reward.set_ylabel("reward")
    ax_reward.legend(loc="lower right")
    ax_reward.grid(True, alpha=0.3)

    # Accuracy as a percentage: training (solid) + held-out generalization (dashed).
    acc_pct = [a * 100 for a in accuracies]
    ax_acc.plot(passes, acc_pct, marker="o", color="tab:blue", label="training (mean)")
    if gen_passes and gen_curves:
        cmap = plt.get_cmap("autumn")
        labels = gen_labels or [f"held-out {i + 1}" for i in range(len(gen_curves))]
        for i, curve in enumerate(gen_curves):
            color = cmap(i / max(len(gen_curves) - 1, 1))
            ax_acc.plot(gen_passes, [a * 100 for a in curve], marker="s", linestyle="--",
                        color=color, alpha=0.9, label=labels[i])
        ax_acc.set_title(f"Accuracy per pass\n(training vs {len(gen_curves)} held-out expressions)")
        ax_acc.legend(loc="lower right", fontsize=7)
    else:
        ax_acc.set_title("Group accuracy per pass")
    ax_acc.set_xlabel("pass"); ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_ylim(-2, 105)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved performance graphs to {out_path}")
