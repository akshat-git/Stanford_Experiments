"""Shared library: tokenizer, data, training, evaluation, and plotting helpers.

This module is a pure dependency: it imports no other local module (except the
leaf data modules, lazily), so the model modules and compare.py can all depend
on it without creating an import cycle.
"""

import os
from collections import namedtuple

import torch
from torch.utils.data import TensorDataset, DataLoader

# All generated files (plots, .pt checkpoints) are written next to this module.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def out_path(filename):
    """Absolute path for an output file inside this folder."""
    return os.path.join(OUTPUT_DIR, filename)


# step_fn(model, batch, device) -> (loss_tensor, num_correct, num_tokens)
ModelSpec = namedtuple("ModelSpec", ["name", "model", "optimizer", "step_fn"])


def get_device():
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_tokenizer(model_name="distilgpt2"):
    """Load a HF tokenizer, ensuring a pad token exists (GPT-2 lacks one)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloader(texts, tokenizer, batch_size=8, max_length=64, shuffle=True):
    """Tokenize texts into a DataLoader of (input_ids, attention_mask)."""
    encodings = tokenizer(
        texts, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_data(tokenizer, batch_size=8, max_length=64, num_train=300, num_eval=60,
              use_real_data=False):
    """Load texts (fake or real) and build train/eval dataloaders.

    Returns (train_loader, eval_loader, source_name).
    """
    # Both data modules share a load_texts(num_train, num_eval) interface.
    data_module = __import__("real_data" if use_real_data else "fake_data")
    train_texts, eval_texts, source = data_module.load_texts(num_train, num_eval)
    print(f"Data source: {source} ({len(train_texts)} train / {len(eval_texts)} eval)")

    train_loader = build_dataloader(train_texts, tokenizer, batch_size, max_length, shuffle=True)
    eval_loader = build_dataloader(eval_texts, tokenizer, batch_size, max_length, shuffle=False)
    return train_loader, eval_loader, source


def save_model(model, filename):
    """Save a model's weights inside this folder; return the path."""
    path = out_path(filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path


def train_model(spec, train_loader, device, epochs):
    """Train for `epochs`; return {"loss": [...], "accuracy": [...]} per epoch."""
    history = {"loss": [], "accuracy": []}
    for epoch in range(epochs):
        spec.model.train()
        total_loss, num_batches, total_correct, total_tokens = 0.0, 0, 0, 0

        for batch in train_loader:
            loss, correct, tokens = spec.step_fn(spec.model, batch, device)

            # Backprop step.
            spec.optimizer.zero_grad()
            loss.backward()
            spec.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            total_correct += correct
            total_tokens += tokens

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        print(f"  Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f}")
    return history


def evaluate_model(spec, eval_loader, device):
    """Evaluate without gradients; return {"loss": ..., "accuracy": ...}."""
    spec.model.eval()
    total_loss, num_batches, total_correct, total_tokens = 0.0, 0, 0, 0

    with torch.no_grad():
        for batch in eval_loader:
            loss, correct, tokens = spec.step_fn(spec.model, batch, device)
            total_loss += loss.item()
            num_batches += 1
            total_correct += correct
            total_tokens += tokens

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }


def run_single(spec, train_loader, eval_loader, device, epochs):
    """Train then evaluate one model; return (history, eval_metrics)."""
    n_params = sum(p.numel() for p in spec.model.parameters())
    print(f"Model '{spec.name}' has {n_params:,} parameters")
    history = train_model(spec, train_loader, device, epochs)
    metrics = evaluate_model(spec, eval_loader, device)
    print(f"  Eval - loss: {metrics['loss']:.4f} - acc: {metrics['accuracy']:.4f}")
    return history, metrics


def plot_history(name, history, filename):
    """Save loss and accuracy curves for a single model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["loss"], marker="o", color="tab:red")
    ax_loss.set_title(f"{name} - training loss")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["accuracy"], marker="o", color="tab:blue")
    ax_acc.set_title(f"{name} - training accuracy")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("next-token accuracy")
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_path(filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def plot_comparison(histories, eval_metrics, filename="model_comparison.png"):
    """Save training curves and final-eval bar charts for both models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (ax_loss, ax_acc), (ax_eval_loss, ax_eval_acc) = axes

    # Per-epoch training curves, one line per model.
    for name, hist in histories.items():
        epochs = range(1, len(hist["loss"]) + 1)
        ax_loss.plot(epochs, hist["loss"], marker="o", label=name)
        ax_acc.plot(epochs, hist["accuracy"], marker="o", label=name)

    ax_loss.set_title("Training loss per epoch")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

    ax_acc.set_title("Training next-token accuracy per epoch")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.legend(); ax_acc.grid(True, alpha=0.3)

    # Final evaluation metrics as bars.
    names = list(eval_metrics.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    ax_eval_loss.bar(names, [eval_metrics[n]["loss"] for n in names], color=bar_colors)
    ax_eval_loss.set_title("Final evaluation loss")
    ax_eval_loss.set_ylabel("loss"); ax_eval_loss.grid(True, axis="y", alpha=0.3)

    ax_eval_acc.bar(names, [eval_metrics[n]["accuracy"] for n in names], color=bar_colors)
    ax_eval_acc.set_title("Final evaluation accuracy")
    ax_eval_acc.set_ylabel("accuracy"); ax_eval_acc.grid(True, axis="y", alpha=0.3)

    fig.suptitle("External LLM vs. hand-built transformer", fontsize=14)
    fig.tight_layout()
    path = out_path(filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


def print_summary(eval_metrics):
    """Print a comparison table of final evaluation metrics."""
    print("\n" + "=" * 52)
    print(f"{'Model':<28}{'Eval loss':>12}{'Eval acc':>12}")
    print("-" * 52)
    for name, m in eval_metrics.items():
        print(f"{name:<28}{m['loss']:>12.4f}{m['accuracy']:>12.4f}")
    print("=" * 52)
