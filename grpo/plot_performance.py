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
    * accuracy -- fraction of the group that is correct.
"""


def plot_performance(losses, rewards, accuracies, out_path, title="GRPO performance"):
    """Save a three-panel figure (loss, mean reward, accuracy) vs pass."""
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

    ax_reward.plot(passes, rewards, marker="o", color="tab:green")
    ax_reward.set_title("Mean group reward\n(the progress signal -- should rise)")
    ax_reward.set_xlabel("pass"); ax_reward.set_ylabel("reward")
    ax_reward.grid(True, alpha=0.3)

    ax_acc.plot(passes, accuracies, marker="o", color="tab:blue")
    ax_acc.set_title("Group accuracy")
    ax_acc.set_xlabel("pass"); ax_acc.set_ylabel("fraction correct")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved performance graphs to {out_path}")
