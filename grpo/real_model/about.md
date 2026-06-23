# GRPO Trainer — Live Policy Optimization

Working **Group Relative Policy Optimization (GRPO)** against a real transformer.
Drives the full RL loop: sample from a Hugging Face causal LM, grade against
verifiable tasks, backpropagate a genuine policy-gradient update.

## Goal

- Train a policy to externalize reasoning in `<think>…</think>` and commit to a verifiable result in `<answer>…</answer>`.
- Score not against an absolute reward but **relative to a sampled group** of *M* completions per prompt.
- Above-cohort completions → made more probable; below-cohort → less probable.
- The group *is* the baseline ⇒ no separate value/critic network.
- Here: convert the reward signal into gradients that reshape live weights — reward → parameter change made explicit and runnable.

## Architecture

- Algorithmic core = standalone modules one directory up (shared).
- This package adds only the model-backed generator (`generate_policy.py`) + two entry points (`grpo_test.py`, `demo.py`).
- Dependencies strictly acyclic, single-direction:

```
parse_tags.py  define_tasks.py  compute_advantages.py   (concept leaves)
        \          /
      score_rewards.py            (imports parse_tags.py)
            \
   generate_policy.py   plot_performance.py  show_thoughts.py   (model; plotting; reporting)
            \                 \            /
   grpo_test.py   demo.py      ← entry points (demo.py = multi-pass + plots + thoughts)
```

- Per-task pipeline: `generate()` → `score_output()` (per completion) → `compute_group_advantages()` → report reinforced (↑) / suppressed (↓) → **`train_step()`** (one gradient update).
- Across passes the gradient updates push the model toward high-reward behaviour ⇒ accuracy trends up (loss is noisy, as expected for REINFORCE).

## Algorithmic core (shared, one level up)

| Module | Key symbol | Notes |
|--------|-----------|-------|
| `parse_tags.py` | `parse_output(text)` | Extracts `<think>` / `<answer>` payloads; `None` per omitted tag. |
| `define_tasks.py` | `Task`, `arithmetic_task(a,b,op)`, `equation_task("7 + 5")` | `Task` = prompt + deterministic `check(answer)` + label + expected answer. `equation_task` parses a command-line equation string. Verification delegated to the task ⇒ any gradable domain swaps in untouched. |
| `score_rewards.py` | `score_output(text,task,weights)`, `RewardWeights` | Scalar reward from 3 criteria: **format** (tags well-formed), **accuracy** (`task.check`), **thought** (reasoning depth, credited *only if correct*). Weights set relative influence. |
| `compute_advantages.py` | `compute_group_advantages(group)` | The defining GRPO op: `advantage = (reward − μ) / σ` over the group. |
| `plot_performance.py` | `plot_performance(losses, accuracies, out)` | Saves the loss + accuracy curves to `performance_graphs.png` (matplotlib, lazy import). |
| `show_thoughts.py` | `show_group(task, scored)` | Prints each output's reasoning + prediction so thinking can be watched improving. |

## This package

| File | Key symbol | Notes |
|------|-----------|-------|
| `generate_policy.py` | `RealPolicy` | Wraps HF causal LM + tokenizer + Adam. `generate()`: *M* completions via stochastic decoding (ignores the answer — model must derive it). `train_step(advantages)`: objective `loss = −(advantage × Σ log p(token)).mean()`; recovers per-token log-probs, masks prompt + padding, backpropagates. |
| `grpo_test.py` | `run_grpo_on_task()`, `demo()` | Minimal single-pass entry point: a GRPO pass + the gradient update over a task set. Needs PyTorch + Transformers atop the shared core. |
| `demo.py` | `run()`, `parse_cli()` | Clean multi-pass demo. Accepts numerical equations on the command line, runs many GRPO passes, prints per-pass thoughts/predictions, and saves `performance_graphs.png`. |

## Usage

```
cd real_model
python demo.py                      # default equations
python demo.py "7 + 5" "12 - 4"     # your own equations
python demo.py --passes 30 "6 * 7"  # more passes
```

- First run downloads a small pretrained model (`distilgpt2`).
- `python grpo_test.py` runs the minimal single-pass version instead.
- Custom domains: build `Task(prompt, check, label, answer)` instances → pass to `run()`.
