"""
Goal: Sample Directions in Color Space. 
3.5 -- probably want to precompute the bounds on each direction so quest doesn't try to keep testing useless saturations
"""
import numpy.typing as npt
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform

import numpy as np

#TODO: Implement the above, also generate all of the precomputed matrices. We also need to account for S-cone noise somewhere, still not applied.

def __convertPolarToCartesian(VSH: npt.ArrayLike) -> npt.ArrayLike:
    """
    Convert Polar to Cartesian Coordinates
    Args:
        HSV (npt.ArrayLike, N x 3): The HSV coordinates that we want to transform. Value is the same but Hue and Saturation are transformed
    """
    V, S, H = VSH[:, 0], VSH[:, 1], VSH[:, 2]
    return np.array([S * np.cos(H), S * np.sin(H), V])


def __convertCartesianToPolar(VCC: npt.ArrayLike) -> npt.ArrayLike:
    """
    Convert Cartesian to Polar Coordinates (VSH)
    Args:
        Cartesian (npt.ArrayLike, N x 3): The Cartesian coordinates that we want to transform
    """
    x, y, z = VCC[:, 0], VCC[:, 1], VCC[:, 2]
    return np.array([x, np.sqrt(y**2 + z**2), np.arctan2(z, y)])


def __convertSphericalToCartesian(thetaPhiRValue: npt.ArrayLike) -> npt.ArrayLike:
    """
    Convert Spherical Coordinates (VSHH) to Cartesian Coordinates
    Args:
        thetaPhiRValue (npt.ArrayLike, N x 4): The Theta, Phi, Radius and Value ordered in a columnar fashion 
    """
    value, r, phi, theta = thetaPhiRValue[:, 0], thetaPhiRValue[:, 1], thetaPhiRValue[:, 2], thetaPhiRValue[:, 3]
    return np.array([value, r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])


def __convertCartesianToSpherical(Cartesian: npt.ArrayLike) -> npt.ArrayLike:
    """
    Convert Cartesian Coordinates to Spherical Coordinates
    Args:
        Cartesian (npt.ArrayLike, N x 4): The Cartesian coordinates that we want to transform
    """
    v, x, y, z = Cartesian[:, 0], Cartesian[:, 1], Cartesian[:, 2], Cartesian[:, 3]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    return np.array([v, r, theta, phi])


def convertHSVToHering(HSV: npt.ArrayLike) -> npt.ArrayLike:
    """
    Converts from HSV to the Max Basis. Returns an Nxdim array of points in the Max Basis
    Args:
        HSV (npt.ArrayLike, Nxdim): The HSV coordinates that we want to transform
    """
    if HSV.shape[1] == 4:
        return __convertSphericalToCartesian(HSV)
    elif HSV.shape[1] == 3:
        return __convertPolarToCartesian(HSV)
    else:
        raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

def convertHeringToHSV(Hering: npt.ArrayLike) -> npt.ArrayLike:
    """
    Converts from the Hering Basis to HSV. Returns an Nxdim array of points in HSV #TODO: decide on ordering.
    Args:
        Hering (npt.ArrayLike, Nxdim): The Max Basis coordinates that we want to transform
    """
    if Hering.shape[1] == 4:
        return np.vstack(Hering[:, 0], __convertCartesianToSpherical(Hering[:, 1:]))
    elif Hering.shape[1] == 3:
        return np.vstack(Hering[:, 0], __convertCartesianToPolar(Hering[:, 1:]))
    else:
        raise NotImplementedError("Not implemented for dimensions other than 3 or 4")


# TODO: @Jess make sure ordering of HSV is right when the matrices are formed
def HSVtoDisplaySpace(HSV: npt.ArrayLike, M_HeringToDisp: npt.ArrayLike) -> npt.ArrayLike:
    """
    Transform from HSV to Display Space, returns an Nxdim array of HSV points in Display Space
    Args:
        HSV (npt.ArrayLike): The HSV coordinates that we want to transform
        M_HSVToDisplay (npt.ArrayLike): The Transform from HSV to Display Space
    """
    return np.clip((M_HeringToDisp@convertHSVToHering(HSV).T).T, 0, 1)


def SampleAlongDirection(VSH: npt.ArrayLike, stepSize: float) -> npt.ArrayLike:
    """
    Sample saturation along the given HSV coordinate
    Args:
        HSV (npt.ArrayLike): The HSV coordinates to sample from
        stepSize (float): The step size to sample at
    """
    # TODO: settle on ordering. I think it should be VSHH because H can always be more dims, less casing can happen for different dimensions
    maxS = VSH[:, 1]
    sampleSat = np.arange(0, maxS, stepSize)
    samplesAlongH = np.repeat(VSH, len(sampleSat), axis=0)
    samplesAlongH[:, 1] = sampleSat
    return samplesAlongH


def computeMetamericAxisInHering(colorSpaceTransform: ColorSpaceTransform) -> npt.ArrayLike:
    """
    Get Metameric Axis in Hering Space
    """
    metamericAxis = np.zeros(colorSpaceTransform.cone_to_disp.shape[0])
    metamericAxis[colorSpaceTransform.metameric_axis] = 1
    direction = np.dot(colorSpaceTransform.cone_to_disp, metamericAxis)
    normalized_direction = direction / np.linalg.norm(direction)
    return np.linalg.inv(colorSpaceTransform.hering_to_disp)@normalized_direction


def getMetamericSteps(colorSpaceTransform: ColorSpaceTransform, stepSize:float) -> npt.ArrayLike:
    """
    Get the Metameric Steps for the given ColorSpaceTransform
    Args:
        colorSpaceTransform (ColorSpaceTransform): The ColorSpaceTransform to get the Metameric Steps for
    """
    metamericAxis = computeMetamericAxisInHering(colorSpaceTransform)
    hsv_hering = convertHeringToHSV(metamericAxis)
    hsv_samples = SampleAlongDirection(hsv_hering, stepSize)
    return HSVtoDisplaySpace(hsv_samples, colorSpaceTransform.hering_to_disp)