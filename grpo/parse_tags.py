"""Tag parsing: pull <think>/<answer> content out of a completion.

GRPO forces the model to "think out loud" inside <think>...</think> and give its
final answer inside <answer>...</answer>. These helpers recover those parts so the
reward function can grade format, reasoning, and correctness separately.
"""

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract(pattern, text):
    """Return the stripped content of the first matching tag, or None if absent."""
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def parse_output(text):
    """Split a completion into (think_content, answer_content); None where missing."""
    return _extract(_THINK_RE, text), _extract(_ANSWER_RE, text)
