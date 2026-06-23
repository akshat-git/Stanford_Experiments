"""Shared plotting helper for the GRPO demos: loss + accuracy across passes.

Used by both folders' demo.py. matplotlib is imported lazily so the core concept
modules stay dependency-free.
"""


def plot_performance(losses, accuracies, out_path, title="GRPO performance"):
    """Save a two-panel figure (training loss and group accuracy vs pass)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passes = range(1, len(losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(passes, losses, marker="o", color="tab:red")
    ax_loss.set_title("Training loss per pass")
    ax_loss.set_xlabel("pass"); ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(passes, accuracies, marker="o", color="tab:blue")
    ax_acc.set_title("Group accuracy per pass")
    ax_acc.set_xlabel("pass"); ax_acc.set_ylabel("fraction correct")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved performance graphs to {out_path}")
