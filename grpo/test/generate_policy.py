"""The policy: what generates the M outputs per task (simulated).

A real GRPO setup samples M decodings from a language model. Here we use a
SIMULATED policy so the concept runs instantly with no GPU or downloads.

The policy itself is a categorical distribution over four behavioural archetypes
(good / lazy / wrong / malformed). `generate()` samples from it; `train_step()`
applies a genuine policy-gradient update to the archetype probabilities from the
group's advantages -- so across passes the policy learns to favour the archetypes
that score well, exactly mirroring what the real model does at the token level.

Like the real model, the update is regularized with a KL penalty to a FROZEN
reference (the initial distribution), so the policy stays close to its starting
point and cannot collapse -- the categorical analogue of the real GRPO leash.
"""

import math
import random
from typing import List, Protocol

ARCHETYPES = ["good", "lazy", "wrong", "malformed"]


class Policy(Protocol):
    """Interface a real model implements: sample completions, then learn from advantages."""

    def generate(self, prompt: str, num_samples: int, correct_answer: int, steps: int) -> List[str]: ...
    def train_step(self, advantages: List[float]) -> float: ...


class MockPolicy:
    """A learnable stand-in 'model' whose policy is a distribution over archetypes.

    Archetypes span a quality range so each group has a spread for the advantages
    to discriminate: good = rich reasoning + correct + tags; lazy = correct but
    shallow; wrong = well-formed but incorrect; malformed = no tags. `correct_answer`
    is injected only so the simulation can synthesise a plausibly-correct output.
    """

    def __init__(self, seed=0, init_weights=(0.3, 0.25, 0.25, 0.2),
                 learning_rate=0.2, kl_coef=0.1):
        self._rng = random.Random(seed)
        self.logits = [math.log(w) for w in init_weights]  # learnable policy parameters
        self.lr = learning_rate
        self.kl_coef = kl_coef
        # Frozen reference = the initial distribution; the KL leash pulls back to it.
        self.ref_logp = [math.log(p) for p in self._softmax()]
        self._last = None  # (archetype_indices, probs) from the most recent generate()

    def _softmax(self):
        m = max(self.logits)
        exps = [math.exp(v - m) for v in self.logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _render(self, kind, correct_answer, steps):
        wrong_answer = correct_answer + self._rng.choice([-2, -1, 1, 2])
        if kind == "good":   # shows one "=" calculation per operation -> full thought credit
            work = "; ".join(f"step {i + 1} = partial" for i in range(steps))
            return f"<think>{work}; total = {correct_answer}</think><answer>{correct_answer}</answer>"
        if kind == "lazy":   # correct, but almost no reasoning (no calculation steps)
            return f"<think>obviously.</think><answer>{correct_answer}</answer>"
        if kind == "wrong":  # reasoning present but the answer is incorrect
            return (
                "<think>I'll guess based on a rough estimate without checking."
                f"</think><answer>{wrong_answer}</answer>"
            )
        return f"the answer is probably {wrong_answer}"  # malformed: no tags

    def generate(self, prompt, num_samples, correct_answer=0, steps=1):
        """Sample `num_samples` outputs from the current archetype distribution.

        `steps` (the task's operation count) scales how much working the "good"
        archetype shows, mirroring the per-task explanation requirement.
        """
        probs = self._softmax()
        idxs = [self._rng.choices(range(len(ARCHETYPES)), weights=probs)[0]
                for _ in range(num_samples)]
        self._last = (idxs, probs)
        return [self._render(ARCHETYPES[i], correct_answer, steps) for i in idxs]

    def train_step(self, advantages):
        """One regularized policy-gradient update on the archetype logits; returns loss.

        loss = -mean(advantage * log p(chosen)) + kl_coef * KL(policy || reference)
        The PG gradient of log p is (one_hot(chosen) - p); the KL term pulls the
        distribution back toward the frozen initial one, so it cannot collapse.
        """
        idxs, probs = self._last
        n, k = len(advantages), len(ARCHETYPES)

        pg_loss = -sum(a * math.log(probs[i]) for a, i in zip(advantages, idxs)) / n
        kl = sum(probs[j] * (math.log(probs[j]) - self.ref_logp[j]) for j in range(k))
        loss = pg_loss + self.kl_coef * kl

        for j in range(k):
            # ascend the PG objective, descend the KL penalty.
            pg_grad = sum(a * ((1.0 if i == j else 0.0) - probs[j]) for a, i in zip(advantages, idxs)) / n
            kl_grad = probs[j] * ((math.log(probs[j]) - self.ref_logp[j]) - kl)
            self.logits[j] += self.lr * (pg_grad - self.kl_coef * kl_grad)
        return loss
