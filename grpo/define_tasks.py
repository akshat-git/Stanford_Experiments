"""Tasks: the "inputted tasks of a very specific nature" GRPO is tested on.

A Task is just a prompt plus a deterministic checker that decides whether an
answer string is correct. Because correctness lives on the task, you can plug in
ANY specific verifiable task type (arithmetic, "is it prime?", string puzzles)
without touching the GRPO machinery.

`equation_task("2 + 3 * 4")` parses a user-supplied numerical expression into a
Task. Expressions may chain several operators, so they take a couple of reasoning
steps (and follow standard operator precedence), not just one.
"""

import re
from dataclasses import dataclass
from typing import Callable

# A chain of integers joined by + - * (>= 1 operator), e.g. "7 + 5 - 3".
_EXPRESSION_RE = re.compile(r"^\s*-?\d+(\s*[+\-*]\s*-?\d+)+\s*$")


@dataclass
class Task:
    prompt: str                       # what the model is asked
    check: Callable[[str], bool]      # answer_text -> is it correct?
    label: str = ""                   # short description for printouts
    answer: int = 0                   # the expected answer (metadata for demos)


def arithmetic_task(a, b, op="+"):
    """Build a single-step arithmetic task with a deterministic answer checker."""
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
    """Parse a numerical expression into a (possibly multi-step) Task.

    Examples: "7 + 5" (one step) or "2 + 3 * 4" (a couple of steps, standard
    precedence). Only integers and the operators + - * are accepted.
    """
    if not _EXPRESSION_RE.match(expr):
        raise ValueError(f"Could not parse expression {expr!r}; use integers joined by + - *.")

    # Safe: the regex guarantees `expr` is only integers, spaces, and + - *.
    expected = eval(expr.strip(), {"__builtins__": {}}, {})

    def check(answer_text):
        try:
            return int(answer_text.strip()) == expected
        except (ValueError, AttributeError):
            return False

    text = expr.strip()
    return Task(prompt=f"What is {text}?", check=check, label=f"{text} = {expected}", answer=expected)
