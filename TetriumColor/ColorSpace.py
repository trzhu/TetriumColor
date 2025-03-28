import numpy as np
import numpy.typing as npt
from typing import List
from enum import Enum

from TetriumColor.Observer import Observer
from TetriumColor.Observer.Spectra import Spectra
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor, PlateColor
from TetriumColor.Observer.DisplayObserverSensitivity import (
    GetColorSpaceTransform, GetColorSpaceTransformTosRGB, GetColorSpaceTransformWODisplay
)
import TetriumColor.ColorMath.Geometry as Geometry
import TetriumColor.ColorMath.Conversion as Conversion


class ColorSpaceType(Enum):
    VSH = "vsh"  # Value-Saturation-Hue
    HERING = "hering"  # Hering opponent color space
    MAXBASIS = "maxbasis"  # Display space (RYGB)
    CONE = "cone"  # Cone responses (SMQL)
    RGB_OCV = "rgb_ocv"  # RGB/OCV 6D representation
    sRGB = "srgb"  # sRGB display

    def __str__(self):
        return self.value


class ColorSpace:
    """
    A class that represents a color space, combining an observer model with a display.

    This class encapsulates the functionality of ColorSpaceTransform and provides methods
    for sampling colors and transforming between different color spaces.
    """

    def __init__(self, observer: Observer,
                 display: List[Spectra] | str | None = None,
                 scaling_factor: float = 1000,
                 metameric_axis: int = 2,
                 subset_leds: List[int] | None = None):
        """
        Initialize a ColorSpace with an observer and optional display.

        Parameters:
            observer (Observer): The observer model
            display (List[Spectra] or str, optional): Either a list of display primary spectra or
                                                    a string identifying a predefined display
            scaling_factor (float, optional): Scaling factor for the display primaries
            metameric_axis (int, optional): Axis to be metameric over
            subset_leds (List[int], optional): Subset of LEDs to use
        """
        self.observer = observer
        self.metameric_axis = metameric_axis
        self.subset_leds = subset_leds or [0, 1, 2, 3]

        self.max_L = (np.linalg.inv(self.transform.hering_to_disp) @
                      np.ones(self.transform.cone_to_disp.shape[0]))[0]

        if display is None:
            # Create a default display transformation without specific primaries
            self.transform: ColorSpaceTransform = GetColorSpaceTransformWODisplay(observer, metameric_axis)
        elif isinstance(display, str):
            # Handle predefined displays
            if display.lower() == 'srgb':
                if observer.dimension != 3:
                    raise ValueError("sRGB display only supported for 3D observers")
                self.transform: ColorSpaceTransform = GetColorSpaceTransformTosRGB(
                    observer, metameric_axis, self.subset_leds)
            else:
                raise ValueError(f"Unknown predefined display: {display}")
        else:
            # Display primaries provided
            self.transform: ColorSpaceTransform = GetColorSpaceTransform(
                observer, display, scaling_factor, metameric_axis, self.subset_leds
            )

        # Store the dimensionality of the color space
        self.dim = self.transform.dim

    def _get_transform_chrom_to_metameric_dir(self) -> npt.NDArray:
        """
        Get the transformation matrix from chromatic coordinates to metameric direction.

        Returns:
            npt.NDArray: Transformation matrix
        """
        normalized_direction = self.get_metameric_axis_in(ColorSpaceType.HERING)
        return Geometry.RotateToZAxis(normalized_direction[1:])

    def get_metameric_axis_in(self, color_space_type: ColorSpaceType) -> npt.NDArray:
        """
        Get the metameric axis in display space.

        Returns:
            npt.NDArray: Normalized direction of the metameric axis
        """
        metameric_axis = np.zeros(self.dim)
        metameric_axis[self.transform.metameric_axis] = 1

        direction = self.convert(metameric_axis, ColorSpaceType.CONE, color_space_type)
        if color_space_type == ColorSpaceType.VSH:
            normalized_direction = direction
            normalized_direction[1] = 1.0  # make saturation 1
        else:
            normalized_direction = direction / np.linalg.norm(direction)
        return normalized_direction

    def _find_maximal_saturation(self, hue_direction: npt.NDArray) -> npt.NDArray:
        """
        Find the point with maximal saturation in the given hue direction.

        Parameters:
            hue_direction (npt.NDArray): Hue direction vector

        Returns:
            npt.NDArray: Point with maximal saturation
        """
        from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation
        result = FindMaximalSaturation(hue_direction, np.eye(self.dim))
        if result is None:
            raise ValueError("Failed to find maximal saturation point")
        else:

            return result

    def _vsh_to_hering(self, vsh: npt.NDArray) -> npt.NDArray:
        """
        Convert from Value-Saturation-Hue to Hering opponent space.

        Parameters:
            vsh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Points in Hering space
        """
        if vsh.shape[1] == 4:
            return np.hstack([vsh[:, [0]], Geometry.ConvertSphericalToCartesian(vsh[:, 1:])])
        elif vsh.shape[1] == 3:
            return np.hstack([vsh[:, [0]], Geometry.ConvertPolarToCartesian(vsh[:, 1:])])
        else:
            raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

    def _hering_to_vsh(self, hering: npt.NDArray) -> npt.NDArray:
        """
        Convert from Hering opponent space to Value-Saturation-Hue.

        Parameters:
            hering (npt.NDArray): Points in Hering space

        Returns:
            npt.NDArray: Points in VSH space
        """
        if hering.shape[1] == 4:
            return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToSpherical(hering[:, 1:])])
        elif hering.shape[1] == 3:
            return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToPolar(hering[:, 1:])])
        else:
            raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

    def _solve_for_boundary(self, L: float, max_L: float, lum_cusp: float, sat_cusp: float) -> float:
        """
        Solve for the boundary of the gamut at a given luminance.

        Parameters:
            L (float): Luminance value to solve for
            max_L (float): Maximum luminance value
            lum_cusp (float): Luminance value at the cusp
            sat_cusp (float): Saturation value at the cusp

        Returns:
            float: Saturation value at the boundary
        """
        if L >= lum_cusp:
            slope = -(max_L - lum_cusp) / sat_cusp
            return (L - max_L) / (slope)
        else:
            slope = lum_cusp / sat_cusp
            return L / slope

    def sample_hue_manifold(self, luminance: float, saturation: float, num_points: int) -> npt.NDArray:
        """
        Sample hue directions at a given luminance and saturation.

        Parameters:
            luminance (float): Luminance value
            saturation (float): Saturation value
            num_points (int): Number of points to sample

        Returns:
            npt.NDArray: Array of sampled points in VSH space
        """
        all_angles = Geometry.SampleAnglesEqually(num_points, self.dim-1)
        all_vshh = np.zeros((len(all_angles), self.dim))
        all_vshh[:, 0] = luminance
        all_vshh[:, 1] = saturation
        all_vshh[:, 2:] = all_angles
        return all_vshh

    def sample_equiluminant_plane(self,
                                  luminance: float,
                                  num_points: int = 100,
                                  remap_to_gamut: bool = True) -> npt.NDArray:
        """
        Sample points on an equiluminant plane.

        Parameters:
            luminance (float): Luminance value for the plane
            num_points (int): Number of points to sample
            remap_to_gamut (bool): Whether to remap points to be within the gamut

        Returns:
            npt.NDArray: Array of sampled points in VSH space
        """
        # Sample hue directions
        vshh = self.sample_hue_manifold(luminance, 1.0, num_points)

        # Remap to gamut if requested
        if remap_to_gamut:
            vshh = self.remap_to_gamut(vshh)

        return vshh

    def _solve_for_cusp(self, angle):
        # Compute the cusp point dynamically if not in the LUT
        hue_cartesian = self._vsh_to_hering(np.array([[0, 1, *angle]]))
        max_sat_point = self._find_maximal_saturation(
            (self.transform.hering_to_disp @ hue_cartesian.T).T[0]
        )
        max_sat_hering = np.linalg.inv(self.transform.hering_to_disp) @ max_sat_point
        max_sat_vsh = self._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
        lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]
        return lum_cusp, sat_cusp

    def remap_to_gamut(self, vshh: npt.NDArray) -> npt.NDArray:
        """
        Remap points to be within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Remapped points that are in gamut
        """
        # Ensure the cubemap is generated
        remapped_vshh = vshh.copy()

        # Remap each point
        for i in range(len(remapped_vshh)):
            angle = tuple(remapped_vshh[i, 2:])

            # Get cusp values by interpolating from the cubemap
            lum_cusp, sat_cusp = self._solve_for_cusp(angle)

            # Calculate the maximum saturation at the given luminance
            sat_max = self._solve_for_boundary(remapped_vshh[i, 0], self.max_L, lum_cusp, sat_cusp)

            # Clamp the saturation to the maximum
            remapped_vshh[i, 1] = min(sat_max, remapped_vshh[i, 1])

        return remapped_vshh

    def max_sat_at_luminance(self, luminance: float, angles: List[tuple[float, float]] | tuple[float, float]) -> float | List[float]:
        """
        Get the maximum saturation at a given luminance.

        Parameters:
            luminance (float): Luminance value

        Returns:
            float: Maximum saturation at the given luminance
        """
        # Ensure the cubemap is generated
        isOneD = False
        if isinstance(angles, tuple):
            isOneD = True
            angles = [angles]
        sat_maxes = []
        for angle in angles:
            # Get cusp values by interpolating from the cubemap
            lum_cusp, sat_cusp = self._solve_for_cusp(angle)
            # Calculate the maximum saturation at the given luminance
            sat_maxes += [self._solve_for_boundary(luminance, self.max_L, lum_cusp, sat_cusp)]

        if isOneD:
            return sat_maxes[0]
        else:
            return sat_maxes

    def is_in_gamut(self, points: npt.NDArray, color_space_type: ColorSpaceType) -> bool:
        """
        Check if points are within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Boolean array indicating if each point is in gamut
        """
        # Get the remapped points
        display_basis = self.convert(points, color_space_type, ColorSpaceType.RGB_OCV)
        in_gamut = np.all((display_basis >= 0) & (display_basis <= 1), axis=1)

        return in_gamut

    def convert(self, points: npt.NDArray,
                from_space: str | ColorSpaceType,
                to_space: str | ColorSpaceType) -> npt.NDArray:
        """
        Transform points from one color space to another.

        Parameters:
            points (npt.NDArray): Points to transform
            from_space (str or ColorSpaceType): Source color space
            to_space (str or ColorSpaceType): Target color space

        Returns:
            npt.NDArray: Transformed points
        """
        # Convert string to enum if necessary
        if isinstance(from_space, str):
            from_space = ColorSpaceType(from_space)
        if isinstance(to_space, str):
            to_space = ColorSpaceType(to_space)

        # If source and target are the same, return the input
        if from_space == to_space:
            return points

        # Handle sRGB as target space (to_space)
        if to_space == ColorSpaceType.sRGB:
            # Convert to cone space first, then to sRGB
            cone_points = self.convert(points, from_space, ColorSpaceType.CONE)
            return (self.transform.cone_to_sRGB @ cone_points.T).T

        # Handle sRGB as source space (from_space)
        if from_space == ColorSpaceType.sRGB:
            if self.transform.dim != 3:
                raise ValueError("sRGB color space is only defined for 3D color spaces")
            # Convert from sRGB to cone space, then proceed with normal conversions
            cone_points = (np.linalg.inv(self.transform.cone_to_sRGB) @ points.T).T
            return self.convert(cone_points, ColorSpaceType.CONE, to_space)

        # Define transformation paths
        if from_space == ColorSpaceType.VSH:
            if to_space == ColorSpaceType.HERING:
                return self._vsh_to_hering(points)
            elif to_space == ColorSpaceType.MAXBASIS:
                hering = self._vsh_to_hering(points)
                return (self.transform.hering_to_disp @ hering.T).T
            elif to_space == ColorSpaceType.CONE:
                hering = self._vsh_to_hering(points)
                display = (self.transform.hering_to_disp @ hering.T).T
                return (np.linalg.inv(self.transform.cone_to_disp) @ display.T).T
            elif to_space == ColorSpaceType.RGB_OCV:
                hering = self._vsh_to_hering(points)
                display = (self.transform.hering_to_disp @ hering.T).T
                return Conversion.Map4DTo6D(display, self.transform)

        elif from_space == ColorSpaceType.HERING:
            if to_space == ColorSpaceType.VSH:
                return self._hering_to_vsh(points)
            elif to_space == ColorSpaceType.MAXBASIS:
                return (self.transform.hering_to_disp @ points.T).T
            elif to_space == ColorSpaceType.CONE:
                display = (self.transform.hering_to_disp @ points.T).T
                return (np.linalg.inv(self.transform.cone_to_disp) @ display.T).T
            elif to_space == ColorSpaceType.RGB_OCV:
                display = (self.transform.hering_to_disp @ points.T).T
                return Conversion.Map4DTo6D(display, self.transform)

        elif from_space == ColorSpaceType.MAXBASIS:
            if to_space == ColorSpaceType.VSH:
                hering = (np.linalg.inv(self.transform.hering_to_disp) @ points.T).T
                return self._hering_to_vsh(hering)
            elif to_space == ColorSpaceType.HERING:
                return (np.linalg.inv(self.transform.hering_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.CONE:
                return (np.linalg.inv(self.transform.cone_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.RGB_OCV:
                return Conversion.Map4DTo6D(points, self.transform)

        elif from_space == ColorSpaceType.CONE:
            if to_space == ColorSpaceType.VSH:
                display = (self.transform.cone_to_disp @ points.T).T
                hering = (np.linalg.inv(self.transform.hering_to_disp) @ display.T).T
                return self._hering_to_vsh(hering)
            elif to_space == ColorSpaceType.HERING:
                display = (self.transform.cone_to_disp @ points.T).T
                return (np.linalg.inv(self.transform.hering_to_disp) @ display.T).T
            elif to_space == ColorSpaceType.MAXBASIS:
                return (self.transform.cone_to_disp @ points.T).T
            elif to_space == ColorSpaceType.RGB_OCV:
                display = (self.transform.cone_to_disp @ points.T).T
                return Conversion.Map4DTo6D(display, self.transform)

        elif from_space == ColorSpaceType.RGB_OCV:
            if to_space == ColorSpaceType.MAXBASIS:
                return Conversion.Map6DTo4D(points, self.transform)
            else:
                # First convert to DISPLAY, then to the target space
                display = Conversion.Map6DTo4D(points, self.transform)
                return self.convert(display, ColorSpaceType.MAXBASIS, to_space)

        # If we reach here, the transformation is not defined
        raise ValueError(f"Transformation from {from_space} to {to_space} not implemented")

    def to_tetra_color(self, vsh_points: npt.NDArray) -> List[TetraColor]:
        """
        Convert VSH points to TetraColor objects.

        Parameters:
            vsh_points (npt.NDArray): Points in VSH space

        Returns:
            List[TetraColor]: List of TetraColor objects
        """
        # Convert to RGB_OCV space
        six_d_color = self.convert(vsh_points, ColorSpaceType.VSH, ColorSpaceType.RGB_OCV)

        # Create TetraColor objects
        return [TetraColor(six_d_color[i, :3], six_d_color[i, 3:])
                for i in range(six_d_color.shape[0])]

    def to_plate_color(self, vsh_point: npt.NDArray, background_luminance: float = 0.5) -> PlateColor:
        """
        Create a PlateColor object from a VSH point and a background luminance.

        Parameters:
            vsh_point (npt.NDArray): Point in VSH space for the foreground
            background_luminance (float): Luminance value for the background

        Returns:
            PlateColor: PlateColor object with foreground and background
        """
        # Create a background point with the same hue but different luminance
        background_vsh = np.array([background_luminance, 0.0, *vsh_point[2:]])

        # Convert both points to RGB_OCV
        points = np.vstack([vsh_point, background_vsh])
        six_d_colors = self.convert(points, ColorSpaceType.VSH, ColorSpaceType.RGB_OCV)

        # Create TetraColor objects for foreground and background
        foreground = TetraColor(six_d_colors[0, :3], six_d_colors[0, 3:])
        background = TetraColor(six_d_colors[1, :3], six_d_colors[1, 3:])

        # Return the PlateColor
        return PlateColor(foreground, background)

    def __str__(self) -> str:
        """
        Generate a unique string representation of this color space for hashing.

        Returns:
            str: Hash string that uniquely identifies this color space's configuration
        """
        # Collect relevant properties that affect color gamut
        components = [
            # Observer properties
            f"observer_dim:{self.observer.dimension}",
            f"{str(self.observer)}",

            # Display Properties
            f"metameric_axis:{self.metameric_axis}",
            f"subset_leds:{self.subset_leds}",
        ]
        # Join all components with a separator
        return "|".join(components)
