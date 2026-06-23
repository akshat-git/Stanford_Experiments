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
