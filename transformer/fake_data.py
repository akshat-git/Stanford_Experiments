"""Synthetic text dataset: generates fake sentences with learnable grammar.

Parallel to ``real_data.py`` (same ``load_texts`` interface), used when no real
corpus is available.
"""

import random

_SUBJECTS = [
    "the transformer", "a language model", "the neural network", "our agent",
    "the researcher", "this experiment", "the optimizer", "a tiny model",
    "the attention head", "the embedding layer",
]
_VERBS = [
    "learns", "predicts", "generates", "encodes", "attends to", "transforms",
    "summarizes", "compresses", "remembers", "explores",
]
_OBJECTS = [
    "the next token", "long sequences", "hidden patterns", "the training data",
    "rare words", "positional information", "useful features", "context windows",
    "gradient updates", "the vocabulary",
]
_TAILS = [
    "during training.", "with surprising accuracy.", "after a few epochs.",
    "on the validation set.", "without overfitting.", "given enough data.",
    "to minimize the loss.", "in a single forward pass.", "across many batches.",
    "while ignoring the padding.",
]


def _generate(num_samples, seed):
    """Build num_samples sentences from the word banks."""
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        clauses = []
        for _ in range(rng.randint(1, 3)):
            clause = " ".join([
                rng.choice(_SUBJECTS), rng.choice(_VERBS),
                rng.choice(_OBJECTS), rng.choice(_TAILS),
            ])
            clauses.append(clause[0].upper() + clause[1:])
        samples.append(" ".join(clauses))
    return samples


def load_texts(num_train, num_eval):
    """Return (train_texts, eval_texts, source_name)."""
    train_texts = _generate(num_train, seed=0)
    eval_texts = _generate(num_eval, seed=1)
    return train_texts, eval_texts, "fake-generated"
