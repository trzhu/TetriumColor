import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from enum import Enum
from PIL import Image, ImageDraw
from tqdm import tqdm

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
        self._gamut_lut = None
        self._gamut_cubemap = None
        self._lum_range = None
        self._sat_range = None
        self._cubemap_size = 64  # Default size for the cubemap

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

    def _generate_gamut_cubemap(self) -> Tuple[Dict, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Generate a cubemap representation of the gamut boundaries.

        Parameters:
            self._cubemap_size (int): Size of the cubemap images

        Returns:
            Tuple containing:
                - Dict of cubemap images
                - Tuple of luminance and saturation ranges
        """
        # Generate grid of UV coordinates
        all_us = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        all_vs = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        cube_u, cube_v = np.meshgrid(all_us, all_vs)
        flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

        # Get metameric direction matrix
        metamericDirMat = self._get_transform_chrom_to_metameric_dir()
        invMetamericDirMat = np.linalg.inv(metamericDirMat)

        # Process each face of the cube
        lut_dicts = []
        for i in tqdm(range(6)):
            # Convert UV to XYZ coordinates for this cube face
            xyz = Geometry.ConvertCubeUVToXYZ(i, cube_u, cube_v, 1).reshape(-1, 3)
            xyz = np.dot(invMetamericDirMat, xyz.T).T

            # Create hering coordinates with unit luminance
            lum_vector = np.ones(self._cubemap_size * self._cubemap_size)
            vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))

            # Convert to VSH space
            vshh = self._hering_to_vsh(vxyz)

            # Generate gamut LUT for this face
            face_dict = {}
            for j in range(len(flattened_u)):
                u, v = flattened_v[j], flattened_u[j]
                angle = tuple(vshh[j, 2:])
                # Find cusp point for this hue angle
                hue_cartesian = self._vsh_to_hering(np.array([[0, 1, *angle]]))
                max_sat_point = self._find_maximal_saturation(
                    (self.transform.hering_to_disp @ hue_cartesian.T).T[0]
                )
                max_sat_hering = np.linalg.inv(self.transform.hering_to_disp) @ max_sat_point
                max_sat_vsh = self._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
                lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]
                face_dict[(u, v)] = (lum_cusp, sat_cusp)

            lut_dicts.append(face_dict)

        # Compute overall ranges for normalization
        all_values = np.array([list(lut_dicts[i].values()) for i in range(6)]).reshape(-1, 2)
        lum_min, lum_max = np.min(all_values[:, 0]), np.max(all_values[:, 0])
        sat_min, sat_max = np.min(all_values[:, 1]), np.max(all_values[:, 1])

        # Generate cubemap images
        cubemap_images = {}
        for i in tqdm(range(6)):
            img = Image.new('RGB', (self._cubemap_size, self._cubemap_size))
            draw = ImageDraw.Draw(img)

            for j in range(len(flattened_u)):
                u, v = flattened_v[j], flattened_u[j]
                lum_cusp, sat_cusp = lut_dicts[i][(u, v)]
                normalized_lum = (lum_cusp - lum_min) / (lum_max - lum_min)
                normalized_sat = (sat_cusp - sat_min) / (sat_max - sat_min)
                rgb_color = (int(normalized_lum * 255), int(normalized_sat * 255), 0)
                draw.point((int(u * self._cubemap_size), int(v * self._cubemap_size)), fill=rgb_color)

            cubemap_images[i] = img

        return cubemap_images, ((lum_min, lum_max), (sat_min, sat_max))

    def _get_transform_chrom_to_metameric_dir(self) -> npt.NDArray:
        """
        Get the transformation matrix from chromatic coordinates to metameric direction.

        Returns:
            npt.NDArray: Transformation matrix
        """
        # "Q" direction
        metameric_axis = np.zeros(self.dim)
        metameric_axis[self.transform.metameric_axis] = 1

        direction = self.convert(metameric_axis, ColorSpaceType.CONE, ColorSpaceType.HERING)
        normalized_direction = direction / np.linalg.norm(direction)
        return Geometry.RotateToZAxis(normalized_direction[1:])

    def _interpolate_from_cubemap(self, angles: tuple) -> Tuple[float, float]:
        """
        Interpolate luminance and saturation values from the cubemap for given angles.

        Parameters:
            angles (npt.NDArray): Nx2 Array of angles for which to interpolate values

        Returns:
            Tuple[float, float]: List of interpolated luminance and saturation values
        """
        if self._gamut_cubemap is None:
            self.get_gamut_lut()  # Initialize the cubemap if not already done

        face_idx, u, v = self._angles_to_cube_uv(angles)

        lum_min, lum_max = self._lum_range
        sat_min, sat_max = self._sat_range

        img = self._gamut_cubemap[face_idx]
        # Convert UV to pixel coordinates
        x, y = u * img.width, v * img.height

        # Get integer pixel coordinates and fractional parts
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, img.width - 1), min(y0 + 1, img.height - 1)
        dx, dy = x - x0, y - y0

        # Get pixel values for the four surrounding pixels
        r00, g00, _ = img.getpixel((x0, y0))
        r10, g10, _ = img.getpixel((x1, y0))
        r01, g01, _ = img.getpixel((x0, y1))
        r11, g11, _ = img.getpixel((x1, y1))

        # Perform bilinear interpolation
        r = (1 - dx) * (1 - dy) * r00 + dx * (1 - dy) * r10 + (1 - dx) * dy * r01 + dx * dy * r11
        g = (1 - dx) * (1 - dy) * g00 + dx * (1 - dy) * g10 + (1 - dx) * dy * g01 + dx * dy * g11

        # Convert normalized values back to actual luminance and saturation
        lum = (r / 255.0) * (lum_max - lum_min) + lum_min
        sat = (g / 255.0) * (sat_max - sat_min) + sat_min

        return lum, sat

    def _angles_to_cube_uv(self, angles: tuple[float, float]):
        """
        Convert angles to cube face index and UV coordinates.

        Parameters:
            angles (npt.NDArray): Angles to convert

        Returns:
            Tuple[int, Tuple[float, float]]: Cube face index and UV coordinates
        """
        angles_with_ones = np.array([[1, 1, *angles]])
        x, y, z = self.convert(angles_with_ones, ColorSpaceType.VSH, ColorSpaceType.HERING)[0, 1:]
        face_id, u, v = Geometry.ConvertXYZToCubeUV(x, y, z)
        return int(face_id), float(u), float(v)

    def get_gamut_lut(self, force_recompute: bool = False):
        """
        Get the gamut lookup table, represented as a cubemap.

        Parameters:
            num_points (int): Number of points for cubemap resolution if recomputing
            force_recompute (bool): Whether to force recomputation

        Returns:
            Dict: Mapping from angle to (luminance_cusp, saturation_cusp)
        """
        if self._gamut_cubemap is None or force_recompute:
            self._gamut_cubemap, ranges = self._generate_gamut_cubemap()
            self._lum_range, self._sat_range = ranges

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

    def _get_metameric_axis_in_disp_space(self) -> npt.NDArray:
        """
        Get the metameric axis in display space.

        Returns:
            npt.NDArray: Normalized direction of the metameric axis
        """
        metameric_axis = np.zeros(self.transform.cone_to_disp.shape[0])
        metameric_axis[self.transform.metameric_axis] = 1
        direction = np.dot(self.transform.cone_to_disp, metameric_axis)
        normalized_direction = direction / np.linalg.norm(direction)
        return normalized_direction

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

    def remap_to_gamut(self, vshh: npt.NDArray) -> npt.NDArray:
        """
        Remap points to be within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Remapped points that are in gamut
        """
        # Ensure the cubemap is generated
        if self._gamut_cubemap is None:
            self.get_gamut_lut()

        # Get the maximum luminance
        max_L = (np.linalg.inv(self.transform.hering_to_disp) @
                 np.ones(self.transform.cone_to_disp.shape[0]))[0]

        # Copy the input to avoid modifying it
        remapped_vshh = vshh.copy()

        # Remap each point
        for i in range(len(remapped_vshh)):
            angle = tuple(remapped_vshh[i, 2:])

            # Get cusp values by interpolating from the cubemap
            lum_cusp, sat_cusp = self._interpolate_from_cubemap(angle)

            # Calculate the maximum saturation at the given luminance
            sat_max = self._solve_for_boundary(remapped_vshh[i, 0], max_L, lum_cusp, sat_cusp)

            # Clamp the saturation to the maximum
            remapped_vshh[i, 1] = min(sat_max, remapped_vshh[i, 1])

        return remapped_vshh

    def is_in_gamut(self, vshh: npt.NDArray) -> bool:
        """
        Check if points are within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Boolean array indicating if each point is in gamut
        """
        # Get the remapped points
        remapped = self.remap_to_gamut(vshh)

        # Check if any coordinates changed
        in_gamut = np.allclose(vshh, remapped, rtol=1e-05, atol=1e-08)

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

    def get_metameric_axis(self) -> npt.NDArray:
        """
        Get the metameric axis in VSH space.

        Returns:
            npt.NDArray: Normalized direction of the metameric axis in VSH space
        """
        # Get the metameric axis in display space
        metameric_axis_disp = self._get_metameric_axis_in_disp_space()

        # Convert to Hering space
        metameric_axis_hering = np.linalg.inv(self.transform.hering_to_disp) @ metameric_axis_disp

        # Convert to VSH space
        metameric_axis_vsh = self._hering_to_vsh(metameric_axis_hering[np.newaxis, :])[0]

        return metameric_axis_vsh
