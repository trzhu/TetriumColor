from typing import List
import numpy as np
import numpy.typing as npt

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor


def ConvertColorsToPlateColors(colors: npt.NDArray, prev_gray_point: npt.NDArray, transform: ColorSpaceTransform) -> List[PlateColor]:
    """
    Nx4 Array in Display Space Coordinates, transform into PlateColor

    :param colors: Nx4 Array in Display Space Coordinates
    :param transform: ColorSpaceTransform to use for the conversion to a Plate Color
    """
    mat: npt.NDArray = Map4DTo6D(colors, transform)
    gray_point: npt.NDArray = Map4DTo6D(prev_gray_point, transform)[0]

    plate_colors: List[PlateColor] = []
    for i in range(mat.shape[0]):
        plate_colors += [PlateColor(TetraColor(mat[i][:3], mat[i][3:]),
                                    TetraColor(gray_point[:3], gray_point[3:]))]
    return plate_colors


def Map4DTo6D(colors: npt.NDArray, transform: ColorSpaceTransform) -> npt.NDArray:
    """
    Nx4 Array in Display Space Coordinates, transform into 6D Array

    :param colors: Nx4 Array in Display Space Coordinates
    :param transform: ColorSpaceTransform to use for the conversion to a Plate Color
    """
    mat = np.zeros((colors.shape[0], 6))
    for i, mapped_idx in enumerate(transform.display_basis):
        # Multiply the color by the white point weight
        # All colors are [0, 1] inside of the color space to make things "nice"
        # But when we need to transform to a display weight, we need to rescale them back
        # in their dynamic range -- need to double check that this is right theoretically!
        mat[:, mapped_idx] = colors[:, i] * transform.white_weights[i]
    return mat


def ConvertMetamersToPlateColors(colors: npt.NDArray, transform: ColorSpaceTransform) -> List[PlateColor]:
    """
    Nx4 Array in Display Space Coordinates, transform into PlateColor

    :param colors: Nx4 Array in Display Space Coordinates
    :param transform: ColorSpaceTransform to use for the conversion to a Plate Color
    """
    mat = Map4DTo6D(colors, transform)

    six_color_array = mat.reshape((colors.shape[0]//2, 2, 6))
    plate_colors: List[PlateColor] = []
    for i in range(six_color_array.shape[0]):
        plate_colors += [PlateColor(TetraColor(six_color_array[i][0][:3], six_color_array[i][0][3:]),
                                    TetraColor(six_color_array[i][1][:3], six_color_array[i][1][3:]))]
    return plate_colors
