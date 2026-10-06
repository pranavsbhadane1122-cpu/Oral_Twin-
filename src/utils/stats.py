"""Small statistics helpers used wherever a proportion is reported.

Every rate in this project is reported with an interval. At the sample sizes
here - 449 test images, and only 19 of them OCA - a point estimate on its own
invites a confidence the data does not support.
"""

import math


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a proportion.

    Preferred over the normal approximation because it behaves sensibly at the
    extremes, which matters when a cell holds 19 images and the estimate may sit
    at or near 1.0.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def format_rate(successes, total, decimals=3):
    """'0.842 [0.790, 0.884] (123/146)'."""
    p, low, high = wilson_interval(successes, total)
    return (f"{p:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}] "
            f"({successes}/{total})")


def interval_width(successes, total):
    _, low, high = wilson_interval(successes, total)
    return high - low
