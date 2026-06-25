# Stanford_Experiments
Assorted scripts for Bayesian experiments, counterfactual tooling, and transformer model experiments.


## Bayesian Experiments
- `bayesian_singlegradient_toy.py` is a small toy script for the single-gradient Bayesian experiment path.
- `bayesian_sinusoidal.py` is the main Bayesian sinusoidal experiment script.
- `bayesian_sinusoidal_toy.py` is the lighter toy variant of the sinusoidal Bayesian setup.

- `output.txt` captures saved console output or experiment notes for the Bayesian folder.


## Cola Sleep
- `about.txt` is a short note or description file for the cola_sleep folder.


## Counterfactuals
- `evaluate.py` runs evaluation for the counterfactual workflow.
- `form_dataset.py` builds the dataset used by the counterfactual scripts.
- `load_meta.py` loads metadata needed by the counterfactual pipeline.
- `setup_models.py` creates or configures the models used in the counterfactual workflow.
- `train.py` is the main training entry point for the counterfactual models.


## Transformer
- `compare.py` is the orchestration entry point that runs both transformer variants on the same data and saves comparison plots.

- `create_transformer.py` is the parallel file for the model that is created from scratch in PyTorch.
- `import_transformer.py` is the parallel file for the model that is imported from a third-party library and then trained.

- `training_utils.py` holds the shared functions for device selection, data loading, training, evaluation, and plotting.

- `fake_data.py` provides the synthetic text source used for fast local transformer runs.
- `real_data.py` provides the real-text source used when the dataset dependency is available.

- `create_transformer_curves.png` stores the training curves for the from-scratch transformer run.
- `import_transformer_curves.png` stores the training curves for the imported transformer run.
- `model_comparison_fake.png` stores the side-by-side comparison plot for the fake-data run.

- `scratch_transformer_model.pt` and `pretrained_transformer_model.pt` are the saved model checkpoints for the two parallel transformer paths.


## GRPO
An intuitive implementation of Group Relative Policy Optimization (force `<think>`/`<answer>` tags, sample a group of outputs per task, reward each relative to the group average). The four concept modules live loose in `grpo/` and are shared by two parallel folders: `test/` (simulated policy) and `real_model/` (real torch+transformers model).

- `research_notes.txt` is the plain-language description of the GRPO concept the code implements.

- `parse_tags.py` pulls the `<think>`/`<answer>` contents out of a model output (`parse_output`).
- `define_tasks.py` defines a `Task` (prompt + answer checker + label + answer + step count) plus `arithmetic_task` and `equation_task`, which accepts any standard arithmetic expression (`+ - * / // % **`, nested parentheses, e.g. `"(12 / 4 + 3) ** 2 - 5"`) via a safe AST walker (no `eval` / code execution) and requires an integer result.
- `score_rewards.py` scores one output on format + accuracy + thought (`score_output`, `RewardWeights`); thought rewards the number of shown `=` calculation steps (only when correct), so the model shows its working rather than padding with prose.
- `compute_advantages.py` performs the core GRPO step, turning rewards into group-relative advantages (`compute_group_advantages`); zeros them for a group with no correct output so GRPO can't converge onto a wrong answer.
- `plot_performance.py` saves loss + reward + accuracy curves to `performance_graphs.png` (`plot_performance`, used by both demos). Loss is the regularized GRPO surrogate (PG term + KL, settles near the KL floor); mean reward is the actual progress signal, drawn against the max-possible-reward ceiling line (`score_rewards.max_reward`); accuracy is shown as a percentage.
- `show_thoughts.py` writes each output's reasoning and prediction per pass to a stream (the demos point it at `run_log.txt`) via `show_group`, and `report_final_modes` prints each task's modal (most common) final answer vs its target.

- `test/generate_policy.py` is the simulated generator (`MockPolicy`) — a learnable distribution over output archetypes with `generate`, a read-only `eval_generate` (the generalization probe), and a KL-regularized `train_step`.
- `test/config.py` is the user-editable settings file (tasks + hyperparameters); `TEST_EQUATIONS` are held-out and probed every `EVAL_EVERY` passes; `PASSES` is double the real folder's.
- `test/grpo_test.py` is the minimal single-pass entry point for the simple, dependency-free demo (`run_grpo_on_task`, `demo`).
- `test/demo.py` is the clean multi-pass demo: reads `config.py` (tunable via `--passes`/`--group`; CLI equations are added to the set), writes per-pass thoughts to `run_log.txt`, prints loss/reward/accuracy + a final per-task mode-of-answers summary, runs a read-only generalization probe on the held-out `TEST_EQUATIONS` every `EVAL_EVERY` passes (overlaid on the accuracy panel), and saves `performance_graphs.png` (needs matplotlib).
- `test/about.md` is the standalone README for the simulation harness.

- `real_model/generate_policy.py` is the real generator (`RealPolicy`) wrapping `Qwen/Qwen2.5-0.5B-Instruct` (chat template + three few-shot examples showing only explicit `a op b = result` steps), with `generate` and a `train_step` GRPO update stabilized by a KL leash to a frozen reference, length-normalized log-probs, and a gentle LR (so it doesn't collapse). Runs bf16 mixed precision (fp32 weights, bf16 compute), a bf16 reference, a `logsumexp` log-prob, and micro-batched generation + training (gradient accumulation) to cut VRAM (peak tracks `MICRO_BATCH`, not `GROUP_SIZE`). A read-only `eval_generate` (training cache restored) serves the generalization probe.
- `real_model/config.py` is the user-editable settings file (tasks + model/training hyperparameters); `TEST_EQUATIONS` are held-out and probed every `EVAL_EVERY` passes; `PASSES` is half the test folder's.
- `real_model/grpo_test.py` is the minimal single-pass entry point for the live training loop (`run_grpo_on_task`, `demo`).
- `real_model/demo.py` is the clean multi-pass demo: reads `config.py` (tunable via `--passes`/`--group`; CLI equations are added to the set), writes per-pass thoughts to `run_log.txt`, prints loss/reward/accuracy + a final per-task mode-of-answers summary, runs a read-only generalization probe on the held-out `TEST_EQUATIONS` every `EVAL_EVERY` passes (overlaid on the accuracy panel), and saves `performance_graphs.png` (needs matplotlib).
- `real_model/about.md` is the standalone README for the live trainer.
