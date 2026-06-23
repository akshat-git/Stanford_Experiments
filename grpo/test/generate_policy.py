"""The policy: what generates the M outputs per task (simulated).

A real GRPO setup samples M decodings from a language model. Here we use a
SIMULATED policy so the concept runs instantly with no GPU or downloads.

The policy itself is a categorical distribution over four behavioural archetypes
(good / lazy / wrong / malformed). `generate()` samples from it; `train_step()`
applies a genuine policy-gradient update to the archetype probabilities from the
group's advantages -- so across passes the policy learns to favour the archetypes
that score well, exactly mirroring what the real model does at the token level.
"""

import math
import random
from typing import List, Protocol

ARCHETYPES = ["good", "lazy", "wrong", "malformed"]


class Policy(Protocol):
    """Interface a real model implements: sample completions, then learn from advantages."""

    def generate(self, prompt: str, num_samples: int, correct_answer: int) -> List[str]: ...
    def train_step(self, advantages: List[float]) -> float: ...


class MockPolicy:
    """A learnable stand-in 'model' whose policy is a distribution over archetypes.

    Archetypes span a quality range so each group has a spread for the advantages
    to discriminate: good = rich reasoning + correct + tags; lazy = correct but
    shallow; wrong = well-formed but incorrect; malformed = no tags. `correct_answer`
    is injected only so the simulation can synthesise a plausibly-correct output.
    """

    def __init__(self, seed=0, init_weights=(0.3, 0.25, 0.25, 0.2), learning_rate=0.5):
        self._rng = random.Random(seed)
        self.logits = [math.log(w) for w in init_weights]  # learnable policy parameters
        self.lr = learning_rate
        self._last = None  # (archetype_indices, probs) from the most recent generate()

    def _softmax(self):
        m = max(self.logits)
        exps = [math.exp(v - m) for v in self.logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _render(self, kind, correct_answer):
        wrong_answer = correct_answer + self._rng.choice([-2, -1, 1, 2])
        if kind == "good":   # rich reasoning + correct answer + proper tags
            return (
                "<think>Break the problem into parts, compute the operation step "
                "by step, then double-check the arithmetic before answering."
                f"</think><answer>{correct_answer}</answer>"
            )
        if kind == "lazy":   # correct, but almost no reasoning
            return f"<think>obviously.</think><answer>{correct_answer}</answer>"
        if kind == "wrong":  # reasoning present but the answer is incorrect
            return (
                "<think>I'll guess based on a rough estimate without checking."
                f"</think><answer>{wrong_answer}</answer>"
            )
        return f"the answer is probably {wrong_answer}"  # malformed: no tags

    def generate(self, prompt, num_samples, correct_answer=0):
        """Sample `num_samples` outputs from the current archetype distribution."""
        probs = self._softmax()
        idxs = [self._rng.choices(range(len(ARCHETYPES)), weights=probs)[0]
                for _ in range(num_samples)]
        self._last = (idxs, probs)
        return [self._render(ARCHETYPES[i], correct_answer) for i in idxs]

    def train_step(self, advantages):
        """One policy-gradient update on the archetype logits; returns the loss.

        loss = -mean(advantage * log p(chosen archetype)); the gradient of log p
        for a categorical policy is (one_hot(chosen) - p), so above-average
        archetypes gain probability and below-average ones lose it.
        """
        idxs, probs = self._last
        n = len(advantages)

        loss = -sum(a * math.log(probs[i]) for a, i in zip(advantages, idxs)) / n

        grad = [0.0] * len(ARCHETYPES)
        for a, i in zip(advantages, idxs):
            for j in range(len(ARCHETYPES)):
                grad[j] += a * ((1.0 if j == i else 0.0) - probs[j])
        self.logits = [self.logits[j] + self.lr * grad[j] / n for j in range(len(ARCHETYPES))]
        return loss
