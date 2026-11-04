"""Measured performance, bound to the score it describes.

THE RULE THIS MODULE ENFORCES: a track score cannot exist without the
performance that produced it. Not "should carry" - cannot be constructed
without. Same discipline as ComparisonResult.Refused carrying no numeric
field: if the type makes the wrong thing unrepresentable, no screen can render
it by accident and no later edit can quietly drop it.

Every figure is READ from models/logs/track_remeasurement.json, the output of
scripts/remeasure_tracks.py. None is typed here. A number typed into code
beside the number it is supposed to equal is two numbers, and two numbers
diverge.

EXTERNAL VALIDATION IS AN ENUM, NOT AN OPTIONAL FIELD. Tracks A and B have no
external validation and cannot have one with the data this project holds -
that is a fact about the evidence, not a missing value. `None` would say
"unknown"; NoneExists says "measured to be impossible, here is why", and
renders as a sentence a reader can act on.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from src.utils.config import PROJECT_ROOT

MEASUREMENT_FILE = "track_remeasurement.json"

# Below this, a point estimate on its own is not an honest summary: the
# interval is wider than most readers would assume. Track A's three classes
# (n = 18, 13, 21) all fall here.
SMALL_SAMPLE = 100


def measurement_path(cfg):
    return PROJECT_ROOT / cfg["paths"]["models"] / "logs" / MEASUREMENT_FILE


@lru_cache(maxsize=1)
def _load(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Track performance is read from the "
            f"re-measurement run, never typed. Run "
            f"python scripts/remeasure_tracks.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def measurements(cfg):
    return _load(str(measurement_path(cfg)))


@dataclass(frozen=True)
class Rate:
    """A measured figure, with whatever counts and interval genuinely exist.

    The counts are not decoration. A sensitivity of 0.072 means something
    different when it is 9 of 125 than when it is 72 of 1000, and only the
    first is this project's figure.

    BUT NOT EVERY FIGURE IS A PROPORTION OF SUCCESSES. A mean Dice over 449
    images is not "163 of 449"; writing it that way invents a count of
    successes that was never computed. Such figures pass counts=None and a
    [basis] describing what the number actually is.

    The interval is likewise optional and NEVER fabricated. Below
    SMALL_SAMPLE it is mandatory, and constructing a small-sample Rate
    without one raises rather than quietly printing a bare point estimate.
    """

    value: float
    denominator: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    numerator: Optional[int] = None
    basis: Optional[str] = None

    def __post_init__(self):
        if self.denominator <= 0:
            raise ValueError("a rate needs a denominator")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"rate out of range: {self.value}")
        if self.small_sample and (self.ci_low is None or self.ci_high is None):
            raise ValueError(
                f"n={self.denominator} is below {SMALL_SAMPLE}, so this figure "
                f"must carry its interval. A bare point estimate on a small "
                f"sample reads as more certain than it is."
            )

    @property
    def small_sample(self) -> bool:
        return self.denominator < SMALL_SAMPLE

    @property
    def has_interval(self) -> bool:
        return self.ci_low is not None and self.ci_high is not None

    def render(self, always_interval: bool = False) -> str:
        text = f"{self.value:.3f}"
        if self.has_interval and (self.small_sample or always_interval):
            text += f" [{self.ci_low:.3f}, {self.ci_high:.3f}]"
        if self.numerator is not None:
            text += f" ({self.numerator} of {self.denominator})"
        elif self.basis:
            text += f" ({self.basis})"
        return text

    @staticmethod
    def from_json(entry) -> "Rate":
        return Rate(value=entry["value"], ci_low=entry["ci"][0],
                    ci_high=entry["ci"][1], numerator=entry["numerator"],
                    denominator=entry["denominator"])


# --------------------------------------------------------------------------
# External validation: an enum of two cases, neither of them empty
# --------------------------------------------------------------------------

class ExternalValidation:
    """Base. Not instantiable - use Measured or NoneExists."""

    def __init__(self):
        if type(self) is ExternalValidation:
            raise TypeError(
                "ExternalValidation is an enum of Measured | NoneExists. "
                "There is no neutral or empty case: a track either has an "
                "external figure or has a stated reason it cannot."
            )

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Measured(ExternalValidation):
    sensitivity: Rate
    set_name: str
    class_balance: str

    def __post_init__(self):
        if not self.set_name.strip():
            raise ValueError("Measured needs the name of the set it was measured on")
        if not self.class_balance.strip():
            raise ValueError(
                "Measured needs its class balance. An external rate without "
                "its base rate is how 0.915 accuracy on a 94.5% negative set "
                "came to be quoted as a result."
            )

    def render(self) -> str:
        return (f"Tested on {self.set_name}, which was never used in training: "
                f"sensitivity {self.sensitivity.render(always_interval=True)}. "
                f"{self.class_balance}")


@dataclass(frozen=True)
class NoneExists(ExternalValidation):
    reason: str

    def __post_init__(self):
        if not self.reason.strip():
            raise ValueError(
                "NoneExists needs a reason. The whole point of this case is "
                "that it says why, rather than leaving a gap a reader fills in "
                "optimistically."
            )

    def render(self) -> str:
        return f"No external testing exists for this. {self.reason}"


@dataclass(frozen=True)
class TrackPerformance:
    """Everything that must travel with a track score."""

    internal: Tuple[Tuple[str, Rate], ...]
    external: ExternalValidation
    note: Optional[str] = None

    def __post_init__(self):
        if not self.internal:
            raise ValueError("a track must report at least one internal figure")
        if not isinstance(self.external, (Measured, NoneExists)):
            raise TypeError(
                "external must be Measured or NoneExists - never None, never "
                "an empty string. An absent external figure is a finding, and "
                "it has to be stated as one."
            )

    def render_lines(self):
        lines = ["Measured performance of this score:"]
        for label, rate in self.internal:
            lines.append(f"  {label}: {rate.render()}")
        lines.append(f"  {self.external.render()}")
        if self.note:
            lines.append(f"  {self.note}")
        return lines
