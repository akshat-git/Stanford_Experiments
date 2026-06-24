"""Tasks: the "inputted tasks of a very specific nature" GRPO is tested on.

A Task is just a prompt plus a deterministic checker that decides whether an
answer string is correct. Because correctness lives on the task, you can plug in
ANY specific verifiable task type (arithmetic, "is it prime?", string puzzles)
without touching the GRPO machinery.

`equation_task("2 * (3 + 4)")` parses a user-supplied numerical expression into a
Task. Expressions may chain several operators and use parentheses, so they take a
varying number of reasoning steps (and follow standard operator precedence).
"""

import re
from dataclasses import dataclass
from typing import Callable

# Allowed characters: integers, spaces, the operators + - *, and parentheses.
_ALLOWED_RE = re.compile(r"^[0-9+\-*()\s]+$")


@dataclass
class Task:
    prompt: str                       # what the model is asked
    check: Callable[[str], bool]      # answer_text -> is it correct?
    label: str = ""                   # short description for printouts
    answer: int = 0                   # the expected answer (metadata for demos)
    steps: int = 1                    # operations required -> how many "=" a full solution shows


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
                label=f"{a} {op} {b} = {expected}", answer=expected, steps=1)


def equation_task(expr):
    """Parse a numerical expression into a (possibly multi-step) Task.

    Supports chained `+ - *` and parentheses with standard precedence, e.g.
    "2 + 3 * 4", "8 - 2 + 5 - 1", or "2 * (3 + 4)". Only integers and `+ - * ( )`
    are accepted.
    """
    text = expr.strip()
    if not text or not _ALLOWED_RE.match(text) or "**" in text.replace(" ", ""):
        raise ValueError(f"Could not parse expression {expr!r}; use integers, + - *, and ().")

    # Safe: only integers, spaces, + - *, and parentheses can reach eval.
    try:
        expected = eval(text, {"__builtins__": {}}, {})
    except SyntaxError:
        raise ValueError(f"Invalid expression {expr!r}.")
    if not isinstance(expected, int):
        raise ValueError(f"Expression {expr!r} must evaluate to an integer.")

    def check(answer_text):
        try:
            return int(answer_text.strip()) == expected
        except (ValueError, AttributeError):
            return False

    # One computation step per binary operator; a full solution shows this many "=".
    steps = max(len(re.findall(r"[+\-*]", text)), 1)
    return Task(prompt=f"What is {text}?", check=check, label=f"{text} = {expected}",
                answer=expected, steps=steps)
