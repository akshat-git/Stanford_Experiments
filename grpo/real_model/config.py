"""User-editable configuration for the real-model GRPO demo.

This is the single place to change the tasks or tuning. Edit the values below;
demo.py reads them (nothing else needs editing).
"""

# --- Tasks ---------------------------------------------------------------- #
# Numerical expressions to train on. Multi-step is fine (chained + - *, standard
# precedence), e.g. "2 + 3 * 4". One Task is built per expression.
# A hard gradient: precedence + long chain, two parenthesised groups, and nested parentheses.
EQUATIONS = [
    "12 + 3 * 4 - 5",            # = 19
    "2 * (3 + 4) - 6",          # = 8
    "(8 - 3) * (2 + 1) + 4",    # = 19
    "3 * (4 + 2 * (5 - 1))",    # = 36  (nested parentheses)
]

# --- Training loop -------------------------------------------------------- #
PASSES = 15          # number of GRPO passes over the task set
GROUP_SIZE = 12      # completions sampled per task per pass (drives most of the compute)

# --- Policy / model hyperparameters --------------------------------------- #
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LEARNING_RATE = 3e-6
KL_COEF = 0.2        # KL leash to the frozen reference (the anti-collapse constraint)
MAX_NEW_TOKENS = 48
TEMPERATURE = 1.0
TOP_K = 50
MICRO_BATCH = 4      # sequences processed on the GPU at once (lower if out of memory)
