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
- This package adds the model-backed generator (`generate_policy.py`), the user-editable `config.py`, and two entry points (`grpo_test.py`, `demo.py`).
- Dependencies strictly acyclic, single-direction:

```
parse_tags.py  define_tasks.py  compute_advantages.py   (concept leaves)
        \          /
      score_rewards.py            (imports parse_tags.py)
            \
   generate_policy.py  plot_performance.py  show_thoughts.py  config.py   (model; plotting; reporting; settings)
            \                 \            /                   /
   grpo_test.py   demo.py      ← entry points (demo.py reads config.py: multi-pass + plots + thoughts)
```

- Per-task pipeline: `generate()` → `score_output()` (per completion) → `compute_group_advantages()` → report reinforced (↑) / suppressed (↓) → **`train_step()`** (one gradient update).
- Across passes the gradient updates push the model toward high-reward behaviour ⇒ **reward and accuracy trend up**.
- Stability: the update is regularized with a **KL penalty to a frozen reference (the initial model)** plus **length-normalized** log-probs and a gentle LR, so the policy holds near its competent start instead of collapsing (plain REINFORCE drifts off the pretrained manifold and "forgets" correct answers). Advantages are also zeroed for any group with no correct output, so a wrong consensus is never reinforced.
- Reading the graphs: the **loss** is the regularized surrogate `-(advantage × mean_logp) + β·KL`. Advantages are z-scored per group (mean 0), so the PG part is noisy and shrinks toward ~0 as the model converges, leaving the small KL floor; its sign/magnitude are not progress. Watch **reward/accuracy** for progress, not loss.

## Algorithmic core (shared, one level up)

| Module | Key symbol | Notes |
|--------|-----------|-------|
| `parse_tags.py` | `parse_output(text)` | Extracts `<think>` / `<answer>` payloads; `None` per omitted tag. |
| `define_tasks.py` | `Task`, `arithmetic_task(a,b,op)`, `equation_task("(12 / 4 + 3) ** 2 - 5")` | `Task` = prompt + `check` + label + answer + `steps`. `equation_task` accepts **any standard arithmetic** (`+ - * / // % **`, nested parentheses) via a **safe AST walker** (no `eval`, so no code execution); the result must be an integer. Verification delegated to the task ⇒ any gradable domain swaps in untouched. |
| `score_rewards.py` | `score_output(text,task,weights)`, `RewardWeights` | Scalar reward from 3 criteria: **format** (tags well-formed), **accuracy** (`task.check`), **thought** (shown `=` calculation steps **scaled to the task's operation count** `task.steps`, credited *only if correct* — so harder problems need fuller working). Weights set relative influence. |
| `compute_advantages.py` | `compute_group_advantages(group)` | The defining GRPO op: `advantage = (reward − μ) / σ` over the group. Zeros the advantages when **no** output is correct, so GRPO can't converge onto a wrong answer (only the KL term acts). |
| `plot_performance.py` | `plot_performance(losses, rewards, accuracies, out, max_reward=)` | Saves loss + reward + accuracy curves to `performance_graphs.png`; the reward panel draws the **max-possible-reward** ceiling line under the reward points; accuracy is shown as a percentage (matplotlib, lazy import). |
| `show_thoughts.py` | `show_group(task, scored, stream=)`, `report_final_modes(tasks, answers)` | `show_group` writes each output's reasoning + prediction to a stream (the demos point it at `run_log.txt`); `report_final_modes` prints each task's modal (most common) final answer vs its target. |

## This package

| File | Key symbol | Notes |
|------|-----------|-------|
| `generate_policy.py` | `RealPolicy` | Wraps `Qwen/Qwen2.5-0.5B-Instruct` + a frozen bf16 reference + tokenizer + Adam; prompts via the chat template with three few-shot examples that show ONLY explicit `a op b = result` steps (no prose) so it follows precedence/parentheses and shows its working instead of narrating rules. Runs bf16 mixed precision (fp32 weights, bf16 compute), a `logsumexp` log-prob, and **micro-batches** generation + the training step (gradient accumulation) so peak memory stays low while keeping the full group. `generate()`: *M* completions via stochastic decoding (ignores the answer — model must derive it). `eval_generate()`: read-only generalization probe — samples completions with the training cache snapshotted/restored, so probing held-out tasks never perturbs the next `train_step`. `train_step(advantages)`: objective `loss = −(advantage × mean_logp) + β·KL(policy‖reference)` — length-normalized log-probs + a KL leash (k3 estimator) so the policy can't collapse; masks prompt + padding, backpropagates. |
| `grpo_test.py` | `run_grpo_on_task()`, `demo()` | Minimal single-pass entry point: a GRPO pass + the gradient update over a task set. Needs PyTorch + Transformers atop the shared core. |
| `demo.py` | `run()`, `evaluate_generalization()`, `parse_cli()` | Clean multi-pass demo. Reads `config.py`, runs many GRPO passes, writes per-pass thoughts/predictions to `run_log.txt` (terminal shows only loss/reward/accuracy), prints a final per-task mode-of-answers summary, and saves `performance_graphs.png`. Every `EVAL_EVERY` passes it runs a read-only **generalization probe** on the held-out `TEST_EQUATIONS` and overlays each held-out curve (dashed) on the accuracy panel. CLI equations are **added** to config's set; `--passes`/`--group` override. |
| `config.py` | `EQUATIONS`, `TEST_EQUATIONS`, `EVAL_EVERY`, `PASSES`, `GROUP_SIZE`, `MODEL_NAME`, `LEARNING_RATE`, `KL_COEF`, `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_K`, `MICRO_BATCH` | **The single editable surface** — tasks + hyperparameters. `TEST_EQUATIONS` are held-out (never trained on) and probed every `EVAL_EVERY` passes. `PASSES` is half the test folder's; lower `MICRO_BATCH` if you hit CUDA OOM. |

## Usage

```
cd real_model
python demo.py                          # default (multi-step) equations
python demo.py "7 + 5 - 3" "2 + 3 * 4"  # ADD these to config's equations
python demo.py --passes 60 "6 * 7 - 1"  # more passes (also adds the equation)
python demo.py --group 24 "6 * 7 - 1"   # more outputs sampled per task
```

- **Edit `config.py`** to change the tasks or tuning — it is the single editable surface. CLI equations are *added* to that set, not substituted.
- Outputs: per-pass thoughts go to `run_log.txt`; a final per-task mode-of-answers summary prints to the terminal (alongside per-pass loss/reward/accuracy); curves to `performance_graphs.png`.
- More outputs per group (`--group`) raises the chance a *correct* sample appears for GRPO to reinforce; more passes (`--passes`) gives more reinforcement opportunities.
- First run downloads the model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB).
- Low on CUDA memory? It already runs bf16 mixed precision, keeps the reference in bf16, avoids materializing the full-vocab softmax, and **micro-batches** generation + training (so peak memory tracks `MICRO_BATCH`, not `GROUP_SIZE`). If still OOM, lower `MICRO_BATCH` in `config.py` first (keeps the group/results), then `MAX_NEW_TOKENS`. For speed, generation dominates (`GROUP_SIZE × MAX_NEW_TOKENS` sequential steps), so lower those.
- `python grpo_test.py` runs the minimal single-pass version instead.
- Custom domains: build `Task(prompt, check, label, answer)` instances → pass to `run()`.
