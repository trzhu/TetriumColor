from typing import List
import numpy as np
import numpy.typing as npt
import math

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor


def SampleAnglesEqually(samples, dim) -> npt.NDArray:
    """
    For a given dimension, sample the sphere equally
    """
    if dim == 2:
        return SampleCircle(samples)
    elif dim == 3:
        return SampleFibonacciSphere(samples)
    else:
        raise NotImplementedError("Only 2D and 3D Spheres are supported")


def SampleCircle(samples=1000) -> npt.NDArray:
    return np.array([[2 * math.pi * (i / float(samples)) for i in range(samples)]]).T


def SampleFibonacciSphere(samples=1000) -> npt.NDArray:
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # stupid but i do not care right now
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arccos(z / r)
        theta = np.arctan2(y, x)
        points.append((theta, phi))

    return np.array(points)


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
        mat[:, mapped_idx] = colors[:, i]
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
