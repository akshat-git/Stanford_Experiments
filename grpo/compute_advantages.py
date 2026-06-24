"""The GRPO step: turn a group's rewards into RELATIVE advantages.

This is the heart of Group Relative Policy Optimization. Instead of an absolute
reward, each output is judged against the average of its own group:

    advantage = (reward - group_mean) / group_std

Above-average outputs get a positive advantage (reinforce); below-average get a
negative one (suppress). The group itself is the baseline, so no separate
value/critic network is needed. Works on any objects with a `.reward` attribute,
a `.correct` flag, and a settable `.advantage`.
"""

import statistics


def compute_group_advantages(scored, eps=1e-6):
    """Set each output's advantage = (reward - group_mean) / group_std (in place).

    Guard against collapsing onto a wrong answer: if NO output in the group is
    correct, there is nothing good to reinforce. Reinforcing the relatively
    "best" wrong output (e.g. the best-formatted one) is exactly what drives the
    policy to a confident wrong solution, so in that case we zero the advantages
    and let only the KL term act (pulling back toward the reference).

    Returns (mean, std) for reporting.
    """
    rewards = [s.reward for s in scored]
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)  # population std over the group
    any_correct = any(getattr(s, "correct", False) for s in scored)
    for s in scored:
        s.advantage = (s.reward - mean) / (std + eps) if any_correct else 0.0
    return mean, std
