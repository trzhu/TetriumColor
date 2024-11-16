import numpy.typing as npt
from typing import List

from dataclasses import dataclass
from enum import Enum



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

# TODO: maybe implement this in a different module and have some general functions to compute on this data structure
@dataclass
class ColorSpaceTransform:
    ConeToDisp: npt.ArrayLike
    MaxBasisToDisp: npt.ArrayLike
    HeringToDisp: npt.ArrayLike
    MetamericAxis: int
    DisplayBasis: List