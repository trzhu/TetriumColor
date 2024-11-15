from dataclasses import dataclass
from enum import Enum
import numpy.typing as npt


class ColorTestResult(Enum):
    kSuccess = 1
    kFailure = 0


class TestType(Enum): # should be some sort of gui opening selection
    Screening = 0   # Maximal Metamers
    Targeted = 1    # Targeted Quest Along N=1 Axis, the Q-axis
    InDepth = 2     # In Depth Quest Along N Hue Directions


@dataclass
class TetraColor:
    RGB: npt.ArrayLike
    OCV: npt.ArrayLike


@dataclass
class PlateColor:
    shape: TetraColor
    background: TetraColor