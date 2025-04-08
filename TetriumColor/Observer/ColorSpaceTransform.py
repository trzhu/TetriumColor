import numpy as np

import numpy.typing as npt
from typing import List
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.models import RGB_COLOURSPACE_BT709
from colour import XYZ_to_RGB, wavelength_to_XYZ

from . import Observer, Spectra, MaxBasisFactory
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform


def GetParalleletopeBasis(observer: Observer, led_spectrums: List[Spectra]):
    disp = observer.observe_spectras(led_spectrums)
    intensities = disp.T
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    return (intensities@np.diag(white_weights)).T


def GetsRGBfromWavelength(wavelength):
    try:
        return XYZ_to_RGB(wavelength_to_XYZ(wavelength), "sRGB")
    except Exception as e:
        return np.array([0, 0, 0])


def GetConeTosRGBPrimaries(observer: Observer, metameric_axis: int = 2):
    if observer.dimension > 3:
        subset = [i for i in range(4) if i != metameric_axis]

    xyz_cmfs = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].values
    white_pt = np.diag(observer.get_whitepoint(wavelengths=observer.wavelengths)[subset])

    M_XYZ_to_RGB = RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB
    M_Cone_To_Primaries = M_XYZ_to_RGB@(xyz_cmfs.T@np.linalg.pinv(observer.sensor_matrix[subset]) @ white_pt / 100)

    if observer.dimension > 3:
        M_Cone_To_Primaries = np.insert(M_Cone_To_Primaries, metameric_axis, 0, axis=1)
    return M_Cone_To_Primaries


def GetColorSpaceTransformTosRGB(observer: Observer, metameric_axis: int = 2,
                                 subset_leds: List[int] = [0, 1, 2, 3]) -> ColorSpaceTransform:
    """ONLY WORKS FOR 3D OBSERVERS. ColorSpaceTransform Object for Observer to sRGB

    Args:
        observer (Observer): observer object
        metameric_axis (int, optional): axis along metamers. Defaults to 2.
        subset_leds (List[int], optional): subset_led. Defaults to [0, 1, 2, 3].

    Returns:
        ColorSpaceTransform: ColorSpaceTransform object
    """
    # ONLY WORKS FOR 3D in general because that transform is only defined for 3 dimensions
    max_basis = MaxBasisFactory.get_object(observer, verbose=False)

    M_Cone_To_Primaries = GetConeTosRGBPrimaries(observer, metameric_axis)

    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        subset_leds,
        np.ones(observer.dimension),
        M_Cone_To_Primaries
    )


def GetColorSpaceTransformWODisplay(observer: Observer, metameric_axis: int = 2) -> ColorSpaceTransform:
    """Given an observer and display primaries, return the ColorSpaceTransform

    Args:
        observer (Observer): Observer object
        metameric_axis: axis to be metameric over
    Returns:
         ColorSpaceTransform
    """
    max_basis = MaxBasisFactory.get_object(observer, verbose=False)

    M_Cone_To_Primaries = np.eye(observer.dimension)  # dummy matrix
    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        [i for i in range(observer.dimension)],  # dummy LEDs
        np.ones(observer.dimension),  # dummy white point
        GetConeTosRGBPrimaries(observer, metameric_axis)
    )


def GetColorSpaceTransform(observer: Observer, display_primaries: List[Spectra] | npt.NDArray,
                           scaling_factor: float = 1000, metameric_axis: int = 2,
                           subset_leds: List[int] = [0, 1, 2, 3]) -> ColorSpaceTransform:
    """Given an observer and display primaries, return the ColorSpaceTransform

    Args:
        observer (Observer): Observer object
        display_primaries (List[Spectra]): List of Spectra objects representing the display primaries
        scaling_factor (float, optional): factor to scale the display primaries by -- they are pretty low by default. Defaults to 1000.

    Returns:
        _type_: ColorSpaceTransform
    """
    max_basis = MaxBasisFactory.get_object(observer, verbose=False)
    disp = observer.observe_spectras(display_primaries)

    intensities = disp.T * scaling_factor
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    rescaled_white_weights = white_weights / np.max(white_weights)
    new_intensities = intensities * white_weights

    M_Cone_To_Primaries = np.linalg.inv(new_intensities)  # something is fucked
    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        subset_leds,
        rescaled_white_weights,
        GetConeTosRGBPrimaries(observer, metameric_axis)
    )


def GetMaxBasisToDisplayTransform(color_space_transform: ColorSpaceTransform) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate from a 4x4 matrix that converts from RYGB to display primaries, a two 4x3 matrices
    that represent the conversion from max basis to display primaries in order to interface with Tetrium Paint Program

    Args:
        color_space_transform (ColorSpaceTransform): the colorspace transform to use.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: a tuple of 4x3 matrices that converts from max basis to display primaries
    """
    disp_weights = np.identity(4) * color_space_transform.white_weights
    # go from bgyr to rygb
    rygb_to_rgbo = disp_weights @ np.flip(color_space_transform.maxbasis_to_disp, axis=1)

    rygb_to_rgb = np.zeros((3, 4))
    rygb_to_rgb = rygb_to_rgbo[:3]

    rygb_to_ocv = np.zeros((3, 4))
    rygb_to_ocv[0] = rygb_to_rgbo[3:]

    return rygb_to_rgb.T, rygb_to_ocv.T
