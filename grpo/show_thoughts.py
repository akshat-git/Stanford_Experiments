"""Shared reporter for the GRPO demos: print each output's reasoning + prediction.

Lets the user watch how the policy's thinking and answers evolve pass over pass.
Writes to a given stream (the demos point this at a per-folder log file, keeping
the terminal clean) and uses ASCII markers only, so it prints on any console.
"""

import sys

from parse_tags import parse_output


def show_group(task, scored, max_think=70, stream=None):
    """Print every output in a group: correctness, predicted answer, reasoning."""
    out = stream if stream is not None else sys.stdout
    print(f"  {task.prompt}  (target {task.answer})", file=out)
    for i, s in enumerate(scored):
        think, answer = parse_output(s.text)
        mark = "+" if s.correct else "-"
        think_str = think or "(no <think> tag)"
        if len(think_str) > max_think:
            think_str = think_str[:max_think] + "..."
        print(f"    {mark} #{i} answer={str(answer):<6} adv={s.advantage:+.2f}  think: {think_str}",
              file=out)
