"""Tasks: the "inputted tasks of a very specific nature" GRPO is tested on.

A Task is just a prompt plus a deterministic checker that decides whether an
answer string is correct. Because correctness lives on the task, you can plug in
ANY specific verifiable task type (arithmetic, "is it prime?", string puzzles)
without touching the GRPO machinery.

`equation_task("(12 / 4 + 3) ** 2 - 5")` parses any standard arithmetic expression
into a Task. There is no operator allow-list -- it accepts `+ - * / // % **` and
parentheses (arbitrary nesting) -- but it is evaluated with a SAFE AST walker, not
`eval`, so only arithmetic runs (no names, calls, or attribute access). The result
must be an integer so the deterministic checker can compare exactly.
"""

import ast
import operator
from dataclasses import dataclass
from typing import Callable

# The only AST node operators we evaluate -- everything else is rejected.
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    """Recursively evaluate a whitelisted arithmetic AST node (safe; no code exec)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression element")


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
    """Parse any standard arithmetic expression into a (multi-step) Task.

    Accepts `+ - * / // % **` and parentheses with standard precedence, e.g.
    "2 + 3 * 4", "3 * (4 + 2 * (5 - 1))", or "(12 / 4 + 3) ** 2 - 5". Evaluated with
    a safe AST walker (no `eval`), and required to be an integer.
    """
    text = expr.strip()
    try:
        tree = ast.parse(text, mode="eval")
        value = _eval_node(tree)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
        raise ValueError(f"Could not parse arithmetic expression {expr!r}.")

    if isinstance(value, float):
        if value.is_integer():
            value = int(value)

    def check(answer_text):
        try:
            return int(answer_text.strip()) == value
        except (ValueError, AttributeError):
            return False

    # One computation step per binary operator; a full solution shows this many "=".
    steps = max(sum(isinstance(n, ast.BinOp) for n in ast.walk(tree)), 1)
    return Task(prompt=f"What is {text}?", check=check, label=f"{text} = {value}",
                answer=value, steps=steps)
