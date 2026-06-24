"""Rewards: score a single completion on format + accuracy + thought.

Maps directly to research_notes.txt:
    format    -> were the <think>/<answer> tags used correctly? (penalize misuse)
    accuracy  -> is the answer correct? (the accuracy checker)
    thought   -> how rich is the reasoning? (reinforced only when also correct)
"""

from dataclasses import dataclass

from parse_tags import parse_output


@dataclass
class RewardWeights:
    """Relative importance of each reward component (tune to taste)."""
    format: float = 1.0
    accuracy: float = 2.0
    thought: float = 1.0
    thought_target_words: int = 12  # reasoning counts as "rich" near this length


def max_reward(weights):
    """Largest reward a single output can earn (each component caps at 1.0)."""
    return weights.format + weights.accuracy + weights.thought


@dataclass
class ScoredOutput:
    """One group member: its text, reward breakdown, and group-relative advantage."""
    text: str
    correct: bool
    format_score: float
    accuracy_score: float
    thought_score: float
    reward: float
    advantage: float = 0.0  # filled in once the whole group is known


def score_output(text, task, weights):
    """Score a single completion -> ScoredOutput (advantage added later by the group)."""
    think, answer = parse_output(text)

    # Format: full credit for both tags, half if only the answer tag exists,
    # zero if the answer tag is missing entirely (nothing gradable).
    if answer is not None and think is not None:
        format_score = 1.0
    elif answer is not None:
        format_score = 0.5
    else:
        format_score = 0.0

    # Accuracy: run the task's own checker on the answer tag.
    correct = answer is not None and task.check(answer)
    accuracy_score = 1.0 if correct else 0.0

    # Thought: depth of reasoning, capped at 1.0, and gated on correctness so
    # verbose-but-wrong reasoning earns nothing ("complex thought WITH accuracy").
    word_count = len(think.split()) if think else 0
    depth = min(word_count / weights.thought_target_words, 1.0)
    thought_score = depth if correct else 0.0

    reward = (
        weights.format * format_score
        + weights.accuracy * accuracy_score
        + weights.thought * thought_score
    )
    return ScoredOutput(text, correct, format_score, accuracy_score, thought_score, reward)
