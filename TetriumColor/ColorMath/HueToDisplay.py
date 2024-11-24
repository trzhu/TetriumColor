"""
Goal: Sample Directions in Color Space.
3.5 -- probably want to precompute the bounds on each direction so quest doesn't try to keep testing useless saturations
"""
import numpy.typing as npt
from typing import List

from tqdm import tqdm
import numpy as np

from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation
import TetriumColor.ColorMath.ColorMathUtils as ColorMathUtils
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor


def __convertPolarToCartesian(SH: npt.NDArray) -> npt.NDArray:
    """
    Convert Polar to Cartesian Coordinates
    Args:
        SH (npt.ArrayLike, N x 2): The SH coordinates that we want to transform. Saturation and Hue are transformed
    """
    S, H = SH[:, 0], SH[:, 1]
    return np.array([S * np.cos(H), S * np.sin(H)]).T


def __convertCartesianToPolar(CC: npt.NDArray) -> npt.NDArray:
    """
    Convert Cartesian to Polar Coordinates (SH)
    Args:
        Cartesian (npt.ArrayLike, N x 2): The Cartesian coordinates that we want to transform
    """
    x, y = CC[:, 0], CC[:, 1]
    rTheta = np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)]).T
    rTheta[:, 1] = np.where(rTheta[:, 1] < -1e-9, rTheta[:, 1] + 2 * np.pi, rTheta[:, 1])  # Ensure θ is in [0, 2π]
    return rTheta


def __convertSphericalToCartesian(rPhiTheta: npt.NDArray) -> npt.NDArray:
    """
    Convert Spherical Coordinates (VSHH) to Cartesian Coordinates
    Args:
        thetaPhiRValue (npt.ArrayLike, N x 4): The Theta, Phi, Radius and Value ordered in a columnar fashion
    """
    r, phi, theta = rPhiTheta[:, 0], rPhiTheta[:, 1], rPhiTheta[:, 2]
    return np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)]).T


def __convertCartesianToSpherical(cartesian: npt.NDArray) -> npt.NDArray:
    """
    Convert Cartesian Coordinates to Spherical Coordinates
    Args:
        Cartesian (npt.ArrayLike, N x 4): The Cartesian coordinates that we want to transform

    Returns:
        npt.ArrayLike: The Spherical Coordinates in the form of R Theta Phi as Nx3 Matrix
    """
    x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    return np.array([r, theta, phi]).T


def ConvertVSHToHering(vsh: npt.NDArray) -> npt.NDArray:
    """
    Converts from HSV to the Max Basis. Returns an Nxdim array of points in the Max Basis
    Args:
        HSV (npt.ArrayLike, Nxdim): The HSV coordinates that we want to transform
    """
    if vsh.shape[1] == 4:
        return np.hstack([vsh[:, [0]], __convertSphericalToCartesian(vsh[:, 1:])])
    elif vsh.shape[1] == 3:
        return np.hstack([vsh[:, [0]], __convertPolarToCartesian(vsh[:, 1:])])
    else:
        raise NotImplementedError(
            "Not implemented for dimensions other than 3 or 4")


def ConvertHeringToVSH(hering: npt.NDArray) -> npt.NDArray:
    """
    Converts from the Hering Basis to HSV. Returns an Nxdim array of points in HSV #TODO: decide on ordering.
    Args:
        Hering (npt.ArrayLike, Nxdim): The Max Basis coordinates that we want to transform
    """
    if hering.shape[1] == 4:
        return np.hstack([hering[:, [0]], __convertCartesianToSpherical(hering[:, 1:])])
    elif hering.shape[1] == 3:
        return np.hstack([hering[:, [0]], __convertCartesianToPolar(hering[:, 1:])])
    else:
        raise NotImplementedError("Not implemented for dimensions other than 3 or 4")


def ConvertVSHtoTetraColor(vsh: npt.NDArray, color_space_transform: ColorSpaceTransform) -> List[TetraColor]:
    """
    Convert VSH to TetraColor
    Args:
        vsh (npt.NDArray): The VSH coordinates to convert
    """
    hering = ConvertVSHToHering(vsh)
    disp = (color_space_transform.hering_to_disp@hering.T).T
    six_d_color = ColorMathUtils.Map4DTo6D(disp, color_space_transform)
    return [TetraColor(six_d_color[i, :3], six_d_color[i, 3:]) for i in range(six_d_color.shape[0])]


def ConvertVSHToPlateColor(vsh: npt.NDArray, luminance: float, color_space_transform: ColorSpaceTransform) -> PlateColor:
    """
    Convert VSH to PlateColor
    Args:
        vsh (npt.NDArray): The VSH coordinates to convert
        luminance (float): The luminance value of the plane
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to use for the conversion
    """
    pair_colors = np.concatenate([vsh[np.newaxis, :], np.array([luminance, 0, 0, 0])[np.newaxis, :]])
    hering = ConvertVSHToHering(pair_colors)
    disp = (color_space_transform.hering_to_disp@hering.T).T
    six_d_color = ColorMathUtils.Map4DTo6D(disp, color_space_transform)
    return PlateColor(TetraColor(six_d_color[0][:3], six_d_color[0][3:]), TetraColor(six_d_color[1][:3], six_d_color[1][3:]))


def SampleAlongDirection(vsh: npt.ArrayLike, step_size: float, max_saturation: float) -> npt.ArrayLike:
    """
    Sample saturation along the given HSV coordinate
    Args:
        HSV (npt.ArrayLike): The HSV coordinates to sample from
        stepSize (float): The step size to sample at
    """
    sample_sat = np.arange(0, max_saturation, step_size)
    samples_along_h = np.repeat(vsh, len(sample_sat), axis=0)
    samples_along_h[:, 1] = sample_sat
    return samples_along_h


def FindMaxSaturationForVSH(vsh: npt.NDArray, color_space_transform: ColorSpaceTransform) -> tuple[float, float]:
    cartesian = (color_space_transform.hering_to_disp@ConvertVSHToHering(vsh).T).T
    max_sat_pt_in_display = np.array([FindMaximalSaturation(cartesian[0], np.eye(color_space_transform.dim))])

    # convert display points back to VSH, and set parameters
    invMat = np.linalg.inv(color_space_transform.hering_to_disp)
    max_sat_per_angle = ConvertHeringToVSH((invMat@max_sat_pt_in_display.T).T)[0]
    return tuple([max_sat_per_angle[0], max_sat_per_angle[1]])


def GenerateGamutLUT(all_vshh: npt.NDArray, color_space_transform: ColorSpaceTransform) -> dict:
    """
    Generate a Look-Up Table for the Gamut of the Given ColorSpaceTransform
    Args:
        all_vshh (npt.NDArray): The VSHH points to generate the LUT for
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to generate the LUT for
    """
    dim = color_space_transform.cone_to_disp.shape[0]
    all_cartesian_points = (color_space_transform.hering_to_disp@ConvertVSHToHering(all_vshh).T).T

    # get max sat points for each hue direction
    map_angle_to_sat = {}
    pts = []
    for pt in tqdm(all_cartesian_points):
        pts += [FindMaximalSaturation(pt, np.eye(dim))]  # paralletope is the unit cube..? yes.
    max_sat_cartesian_per_angle = np.array(pts)

    # convert display points back to VSH, and set parameters
    invMat = np.linalg.inv(color_space_transform.hering_to_disp)
    max_sat_per_angle = ConvertHeringToVSH((invMat@max_sat_cartesian_per_angle.T).T)
    for angle, sat in zip(all_vshh[:, 2:], max_sat_per_angle):
        map_angle_to_sat[tuple(angle)] = tuple([sat[0], sat[1]])
    return map_angle_to_sat


def SolveForBoundary(L: float, max_L: float, lum_cusp: float, sat_cusp: float) -> float:
    """
    Solve for the boundary of the gamut
    Args:
        L (float): The Luminance Value to solve for
        max_L (float): The Maximum Luminance Value
        lum_cusp (float): The Luminance Value at the Cusp
        sat_cusp (float): The Saturation Value at the Cusp

    Returns:
        float: The Saturation Value that corresponds to the boundary point at L
    """
    # get the cusp point for the given angle -- either presolved or solved on the fly atm
    if L > lum_cusp:
        slope = -(max_L - lum_cusp) / sat_cusp
        return (L - max_L) / (slope)
    else:
        slope = lum_cusp / sat_cusp
        return L / slope


def GetEquiluminantPlane(luminance: float, color_space_transform: ColorSpaceTransform, map_angle_sat: dict) -> dict:
    """Get the saturation plane for the given VSHH points

    Args:
        luminance (float): luminance value
        color_space_transform (ColorSpaceTransform): color space transform object
        map_angle_sat (dict): dictionary that solves for the boundary of the gamut

    Returns:
        map_angle_lum_sat(dict): Mapping from angle to constant luminance varying saturation
    """
    map_angle_lum_sat = {}
    max_L = (np.linalg.inv(color_space_transform.hering_to_disp) @
             np.ones(color_space_transform.cone_to_disp.shape[0]))[0]
    for angle, (lum_cusp, sat_cusp) in map_angle_sat.items():
        sat = SolveForBoundary(luminance, max_L, lum_cusp, sat_cusp)
        map_angle_lum_sat[angle] = (luminance, sat)
    return map_angle_lum_sat


def RemapGamutPoints(VSHH: npt.NDArray, color_space_transform: ColorSpaceTransform, map_angle_sat: dict) -> npt.NDArray:
    """Given a set of VSHH points, remap the saturation values to be within the gamut

    Args:
        VSHH (npt.NDArray): value, saturation, hue(dim-2)
        color_space_transform (ColorSpaceTransform): color space transform object
        map_angle_sat (dict): dictionary that solves for the boundary of the gamut

    Returns:
        npt.NDArray: Remapped VSHH
    """
    max_L = (np.linalg.inv(color_space_transform.hering_to_disp) @
             np.ones(color_space_transform.cone_to_disp.shape[0]))[0]
    for i in range(len(VSHH)):
        angle = tuple(VSHH[i, 2:])
        if angle not in map_angle_sat:
            lum_cusp, sat_cusp = FindMaxSaturationForVSH(np.array([[0, 1, *angle]]), color_space_transform)
        else:
            lum_cusp, sat_cusp = map_angle_sat[angle]
        sat = SolveForBoundary(VSHH[i][0], max_L, lum_cusp, sat_cusp)
        VSHH[i, 1] = min(sat, VSHH[i][1])
    return VSHH


def SampleHueManifold(luminance: float, saturation: float, dim: int, num_points: int) -> npt.NDArray:
    """
    Generate a sphere of hue values
    Args:
        luminance (float): The luminance value to generate the sphere at
        saturation (float): The saturation value to generate the sphere at
    """
    all_angles = ColorMathUtils.SampleAnglesEqually(num_points, dim-1)
    all_vshh = np.zeros((len(all_angles), dim))
    all_vshh[:, 0] = luminance
    all_vshh[:, 1] = saturation
    all_vshh[:, 2:] = all_angles
    return all_vshh


def GetMetamericAxisInVSH(color_space_transform: ColorSpaceTransform) -> npt.NDArray:
    """
    Get the Metameric Axis in VSH
    Args:
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to get the Metameric Axis for
    """
    metameric_axis = np.zeros(color_space_transform.cone_to_disp.shape[0])
    metameric_axis[color_space_transform.metameric_axis] = 1
    direction = np.dot(color_space_transform.cone_to_disp, metameric_axis)
    normalized_direction = direction / np.linalg.norm(direction)  # return normalized direction
    return ConvertHeringToVSH(normalized_direction[np.newaxis, :])
