# GRPO Trainer — Live Policy Optimization

Working **Group Relative Policy Optimization (GRPO)** against a real transformer
(`Qwen/Qwen2.5-0.5B-Instruct`). Drives the full RL loop: sample from the model,
grade against verifiable tasks, backpropagate a genuine policy-gradient update.

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
- Across passes the gradient updates push the model toward high-reward behaviour ⇒ **reward and accuracy trend up**.
- Stability: the update is regularized with a **KL penalty to a frozen reference (the initial model)** plus **length-normalized** log-probs and a gentle LR, so the policy holds near its competent start instead of collapsing (plain REINFORCE drifts off the pretrained manifold and "forgets" correct answers).
- Reading the graphs: the **loss** is the regularized surrogate `-(advantage × mean_logp) + β·KL`. Advantages are z-scored per group (mean 0), so the PG part is noisy and shrinks toward ~0 as the model converges, leaving the small KL floor; its sign/magnitude are not progress. Watch **reward/accuracy** for progress, not loss.

## Algorithmic core (shared, one level up)

| Module | Key symbol | Notes |
|--------|-----------|-------|
| `parse_tags.py` | `parse_output(text)` | Extracts `<think>` / `<answer>` payloads; `None` per omitted tag. |
| `define_tasks.py` | `Task`, `arithmetic_task(a,b,op)`, `equation_task("7 + 5")` | `Task` = prompt + deterministic `check(answer)` + label + expected answer. `equation_task` parses a command-line equation string. Verification delegated to the task ⇒ any gradable domain swaps in untouched. |
| `score_rewards.py` | `score_output(text,task,weights)`, `RewardWeights` | Scalar reward from 3 criteria: **format** (tags well-formed), **accuracy** (`task.check`), **thought** (reasoning depth, credited *only if correct*). Weights set relative influence. |
| `compute_advantages.py` | `compute_group_advantages(group)` | The defining GRPO op: `advantage = (reward − μ) / σ` over the group. |
| `plot_performance.py` | `plot_performance(losses, rewards, accuracies, out)` | Saves loss + reward + accuracy curves to `performance_graphs.png` (matplotlib, lazy import). |
| `show_thoughts.py` | `show_group(task, scored)` | Prints each output's reasoning + prediction so thinking can be watched improving. |

## This package

| File | Key symbol | Notes |
|------|-----------|-------|
| `generate_policy.py` | `RealPolicy` | Wraps `Qwen/Qwen2.5-0.5B-Instruct` + a frozen reference copy + tokenizer + Adam; prompts via the chat template with a one-shot tag example. `generate()`: *M* completions via stochastic decoding (ignores the answer — model must derive it). `train_step(advantages)`: objective `loss = −(advantage × mean_logp) + β·KL(policy‖reference)` — length-normalized log-probs + a KL leash (k3 estimator) so the policy can't collapse; masks prompt + padding, backpropagates. |
| `grpo_test.py` | `run_grpo_on_task()`, `demo()` | Minimal single-pass entry point: a GRPO pass + the gradient update over a task set. Needs PyTorch + Transformers atop the shared core. |
| `demo.py` | `run()`, `parse_cli()` | Clean multi-pass demo. Accepts numerical equations on the command line, runs many GRPO passes, prints per-pass thoughts/predictions, and saves `performance_graphs.png`. |

## Usage

```
cd real_model
python demo.py                      # default equations
python demo.py "7 + 5" "12 - 4"     # your own equations
python demo.py --passes 60 "6 * 7"  # more passes
python demo.py --group 24 "6 * 7"   # more outputs sampled per task
```

- More outputs per group (`--group`) raises the chance a *correct* sample appears for GRPO to reinforce; more passes (`--passes`) gives more reinforcement opportunities.
- First run downloads the model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB).
- `python grpo_test.py` runs the minimal single-pass version instead.
- Custom domains: build `Task(prompt, check, label, answer)` instances → pass to `run()`.
