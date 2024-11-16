import numpy.typing as npt
from typing import List

from dataclasses import dataclass
from enum import Enum



class ColorTestResult(Enum):
    kSuccess = 1
    kFailure = 0


class TestType(Enum): # should be some sort of gui opening selection
    screening = 0   # Maximal Metamers
    targeted = 1    # Targeted Quest Along N=1 Axis, the Q-axis
    in_depth = 2     # In Depth Quest Along N Hue Directions


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
    """
    A dataclass to hold the color space transform data
    Attributes: 
        cone_to_disp: The cone to display transform matrix
        maxbasis_to_disp: The maxbasis to display transform matrix
        hering_to_disp: The hering to display transform matrix
        metameric_axis: The axis of the metameric transform
        display_basis: The indices of the LED in RGVOCV order
    """
    cone_to_disp: npt.ArrayLike
    maxbasis_to_disp: npt.ArrayLike
    hering_to_disp: npt.ArrayLike
    metameric_axis: int
    display_basis: List