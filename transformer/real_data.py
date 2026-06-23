"""Imported text dataset: loads a real corpus via Hugging Face ``datasets``.

Parallel to ``fake_data.py`` (same ``load_texts`` interface). Requires the
``datasets`` package (``pip install datasets``).
"""


def load_texts(num_train, num_eval):
    """Return (train_texts, eval_texts, source_name) from WikiText-2."""
    from datasets import load_dataset

    # Load and keep only non-trivial lines.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 30]

    if len(texts) < num_train + num_eval:
        raise ValueError("Not enough usable lines in the dataset.")

    train_texts = texts[:num_train]
    eval_texts = texts[num_train:num_train + num_eval]
    return train_texts, eval_texts, "wikitext-2-raw-v1"
