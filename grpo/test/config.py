"""User-editable configuration for the simulated GRPO demo.

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
PASSES = 100          # double the real folder's passes (this demo is instant)
GROUP_SIZE = 50      # completions sampled per task per pass

# --- Policy hyperparameters (categorical archetype policy) ---------------- #
SEED = 42
LEARNING_RATE = 0.2
KL_COEF = 0.1        # KL leash to the initial distribution (prevents collapse)
INIT_WEIGHTS = (0.3, 0.25, 0.25, 0.2)  # P(good, lazy, wrong, malformed) at start
