"""User-editable configuration for the simulated GRPO demo.

This is the single place to change the tasks or tuning. Edit the values below;
demo.py reads them (nothing else needs editing).
"""

# --- Tasks ---------------------------------------------------------------- #
# Numerical expressions to train on. Multi-step is fine (chained + - *, standard
# precedence), e.g. "2 + 3 * 4". One Task is built per expression.
# Complex multi-step expressions exercising the full operator set: powers (**),
# division (/), modulo (%), and floor division (//), with parentheses.
EQUATIONS = [
    "2 ** 5 - 3 * (4 + 1)",             # = 17  (power)
    "(12 / 4 + 3) * 2 - 5",             # = 7   (division)
    "100 % 7 + 2 ** 3 * (5 - 2)",       # = 26  (modulo + power)
    "(6 + 4) * 3 // 4 + 2 ** 2",        # = 11  (floor division + power)
]

# --- Generalization probe ------------------------------------------------- #
# Held-out expressions the policy is NEVER trained on. Every EVAL_EVERY passes
# the current policy is evaluated on these (read-only -- no weight update, RNG
# snapshot/restored) and their accuracy is overlaid on the training-accuracy
# panel, so you can see whether what the policy learned generalizes.
TEST_EQUATIONS = [
    "3 * 4 + 2 ** 3 - 5",         # = 15
    "(20 - 6) // 3 + 2 * 4",      # = 12
    "50 % 9 + 3 ** 2 * 2",        # = 23
    "(8 + 4) / 3 * 2 + 1",        # = 9   (division)
    "2 ** 4 - 10 % 4 + 3",        # = 17
]
EVAL_EVERY = 3       # run the generalization probe after every N training passes

# --- Training loop -------------------------------------------------------- #
PASSES = 100          # double the real folder's passes (this demo is instant)
GROUP_SIZE = 50      # completions sampled per task per pass

# --- Policy hyperparameters (categorical archetype policy) ---------------- #
SEED = 42
LEARNING_RATE = 0.2
KL_COEF = 0.1        # KL leash to the initial distribution (prevents collapse)
INIT_WEIGHTS = (0.3, 0.25, 0.25, 0.2)  # P(good, lazy, wrong, malformed) at start
