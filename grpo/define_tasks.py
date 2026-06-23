"""Tasks: the "inputted tasks of a very specific nature" GRPO is tested on.

A Task is just a prompt plus a deterministic checker that decides whether an
answer string is correct. Because correctness lives on the task, you can plug in
ANY specific verifiable task type (arithmetic, "is it prime?", string puzzles)
without touching the GRPO machinery.

`equation_task("7 + 5")` parses a user-supplied numerical equation into a Task, so
equations can be passed straight from the command line.
"""

import re
from dataclasses import dataclass
from typing import Callable

# Matches "a op b" with op in + - *, allowing spaces and negative integers.
_EQUATION_RE = re.compile(r"^\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*$")


@dataclass
class Task:
    prompt: str                       # what the model is asked
    check: Callable[[str], bool]      # answer_text -> is it correct?
    label: str = ""                   # short description for printouts
    answer: int = 0                   # the expected answer (metadata for demos)


def arithmetic_task(a, b, op="+"):
    """Build a specific arithmetic task with a deterministic answer checker."""
    expected = {"+": a + b, "-": a - b, "*": a * b}[op]

    def check(answer_text):
        # Correct only if the answer parses to the exact expected integer.
        try:
            return int(answer_text.strip()) == expected
        except (ValueError, AttributeError):
            return False

    return Task(prompt=f"What is {a} {op} {b}?", check=check,
                label=f"{a} {op} {b} = {expected}", answer=expected)


def equation_task(expr):
    """Parse a numerical equation string (e.g. "7 + 5") into a Task."""
    match = _EQUATION_RE.match(expr)
    if not match:
        raise ValueError(f"Could not parse equation {expr!r}; expected 'a + b', 'a - b', or 'a * b'.")
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    return arithmetic_task(a, b, op)
