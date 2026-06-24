# GRPO Reference Implementation — Simulation Harness

Local reference for **Group Relative Policy Optimization (GRPO)**, driven
by a deterministic simulated generator. Isolates the *algorithm's mechanics* from
any model

## Goal

- Teach a policy to externalize reasoning in `<think>…</think>` and commit to a verifiable result in `<answer>…</answer>`.
- Score not against an absolute reward but **relative to a sampled group** of *M* completions per prompt.
- Above-cohort completions → reinforced; below-cohort → suppressed.
- The group *is* the baseline ⇒ no separate value/critic network.
- Here: maximal pedagogical fidelity — every stage legible, every reward reproducible and unambiguous.

## Architecture

- Algorithmic core = standalone modules one directory up (shared).
- This package adds only the generator (`generate_policy.py`) + two entry points (`grpo_test.py`, `demo.py`).
- Dependencies strictly acyclic, single-direction:

```
parse_tags.py  define_tasks.py  compute_advantages.py   (concept leaves)
        \          /
      score_rewards.py            (imports parse_tags.py)
            \
   generate_policy.py   plot_performance.py  show_thoughts.py   (generator; plotting; reporting)
            \                 \            /
   grpo_test.py   demo.py      ← entry points (demo.py = multi-pass + plots + thoughts)
```

- Per-task pipeline: `generate()` → `score_output()` (per completion) → `compute_group_advantages()` → `train_step()` (update policy) → report reinforced (↑) / suppressed (↓).
- Across passes the archetype distribution shifts toward high-reward behaviour ⇒ **reward and accuracy climb**.
- Stability: the update is regularized with a **KL penalty to the frozen initial distribution**, so the policy settles near a leashed optimum instead of collapsing.
- Reading the graphs: the **loss** is the regularized surrogate `-(advantage × logp) + β·KL`. Advantages are z-scored per group (mean 0), so the PG part shrinks to ~0 at convergence, leaving the small KL floor — expected, not a failure. Watch **reward/accuracy** for progress, not loss.

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
| `generate_policy.py` | `MockPolicy` | A *learnable* categorical policy over 4 archetypes — *competent / lazy / incorrect / malformed*. `generate()` samples them; `train_step(advantages)` applies a categorical policy-gradient update **plus a KL penalty to the frozen initial distribution**, so it improves across passes without collapsing — mirroring the real model's GRPO leash at the archetype level. |
| `grpo_test.py` | `run_grpo_on_task()`, `demo()` | Minimal single-pass entry point: generate → score → normalize → report. Standard library only. |
| `demo.py` | `run()`, `parse_cli()` | Clean multi-pass demo. Accepts numerical equations on the command line, runs many GRPO passes, prints per-pass thoughts/predictions, and saves `performance_graphs.png`. Needs matplotlib. |

## Usage

```
cd test
python demo.py                      # default equations
python demo.py "7 + 5" "12 - 4"     # your own equations
python demo.py --passes 100 "6 * 7" # more passes
python demo.py --group 24 "6 * 7"   # more outputs sampled per task
```

- More outputs per group (`--group`) raises the chance a high-reward sample appears for the policy to reinforce; more passes (`--passes`) gives more reinforcement opportunities.
- `python grpo_test.py` runs the minimal single-pass version instead.
- Custom domains: build `Task(prompt, check, label, answer)` instances → pass to `run()`.
