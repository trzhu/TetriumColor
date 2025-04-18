from re import I
from colour.colorimetry import XYZ_ColourMatchingFunctions
import numpy as np
import numpy.typing as npt
from typing import List
from enum import Enum

from TetriumColor.Observer import Observer, GetHeringMatrix
from TetriumColor.Observer.Spectra import Spectra
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor, PlateColor
from TetriumColor.Observer.ColorSpaceTransform import (
    GetColorSpaceTransform, GetColorSpaceTransformTosRGB, GetColorSpaceTransformWODisplay, GetMaxBasisToDisplayTransform
)

from colour.models import RGB_COLOURSPACE_BT709
import TetriumColor.ColorMath.Geometry as Geometry
import TetriumColor.ColorMath.Conversion as Conversion


OKLAB_M1 = np.array([
    [0.8189330101, 0.0329845436, 0.0482003018],
    [0.3618667424, 0.9293118715, 0.2643662691],
    [-0.1288597137, 0.0361456387, 0.6338517070]
]).T

OKLAB_M2 = np.array([
    [0.210454, 0.793617, -0.004072],
    [1.977998, -2.428592, 0.450593],
    [0.025904, 0.782771, -0.808675]
])

M_XYZ_to_RGB = RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB


class ColorSpaceType(Enum):
    VSH = "vsh"  # Value-Saturation-Hue
    HERING = "hering"  # Hering opponent color space
    MAXBASIS = "maxbasis"  # Display space (RYGB)
    CONE = "cone"  # Cone responses (SMQL)
    RGB_OCV = "rgb_ocv"  # RGB/OCV 6D representation
    DISP = "disp"  # Display space (RGBO)
    SRGB = "srgb"  # sRGB display
    XYZ = "xyz"  # CIE XYZ color space
    CHROM = "chrom"  # Chromaticity space
    HERING_CHROM = "hering_chrom"  # Hering chromaticity space
    OKLAB = 'oklab'

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

        self.max_L = (np.linalg.inv(self.transform.hering_to_disp) @
                      np.ones(self.transform.cone_to_disp.shape[0]))[0]

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

        # Handle 3D only points separately -- only one directional
        if to_space == ColorSpaceType.SRGB:
            # Convert to cone space first, then to sRGB
            cone_points = self.convert(points, from_space, ColorSpaceType.XYZ)
            return (M_XYZ_to_RGB @ cone_points.T).T
        elif to_space == ColorSpaceType.OKLAB:
            if self.transform.dim != 3:
                raise ValueError("OKLAB color space is only defined for 3D color spaces")
            # Convert to cone space first, then to OKLAB
            xyz_points = self.convert(points, from_space, ColorSpaceType.XYZ)
            m1_points = OKLAB_M1 @ xyz_points.T
            m1_cubed = np.cbrt(m1_points)
            m2_points = OKLAB_M2 @ m1_cubed
            return m2_points.T

         # chromaticity based color transforms
        if from_space == ColorSpaceType.CHROM or from_space == ColorSpaceType.HERING_CHROM:
            raise ValueError("Cannot transform from chromaticity back to another color space")
        elif to_space == ColorSpaceType.CHROM:
            cone_pts = self.convert(points, from_space, ColorSpaceType.CONE)
            return (cone_pts.T / (np.sum(cone_pts.T, axis=0) + 1e-9))[1:].T  # auto drop first coordinate
        elif to_space == ColorSpaceType.HERING_CHROM:
            maxbasis_pts = self.convert(points, from_space, ColorSpaceType.MAXBASIS)
            return (GetHeringMatrix(self.transform.dim) @
                    (maxbasis_pts.T / (np.sum(maxbasis_pts.T, axis=0) + 1e-9)))[:, 1:].T

        # Handle the basic linear transforms
        if from_space == ColorSpaceType.VSH:
            return self.convert(self._vsh_to_hering(points), ColorSpaceType.HERING, to_space)
        elif from_space == ColorSpaceType.SRGB:
            # Convert from sRGB to cone space, then proceed with normal conversions
            cone_points = (np.linalg.inv(M_XYZ_to_RGB) @ points.T).T
            return self.convert(cone_points, ColorSpaceType.XYZ, to_space)
        elif from_space == ColorSpaceType.OKLAB:
            if self.transform.dim != 3:
                raise ValueError("OKLAB color space is only defined for 3D color spaces")
            # Convert to cone space first, then to OKLAB
            m2_points = np.linalg.inv(OKLAB_M2) @ points.T
            m2_cubed = np.power(m2_points, 3)
            xyz_points = np.linalg.inv(OKLAB_M1) @ m2_cubed
            return self.convert(xyz_points.T, ColorSpaceType.XYZ, to_space)
        elif from_space == ColorSpaceType.XYZ:
            if self.transform.dim != 3:
                raise ValueError("transforming from XYZ to another color space is only defined for 3D color spaces")
            cone_pts = np.linalg.inv(self.transform.cone_to_XYZ) @ points.T  # can't do this inverse if it's not 3D
            return self.convert(cone_pts.T, ColorSpaceType.CONE, to_space)
        elif from_space == ColorSpaceType.HERING:
            disp_points = self.transform.hering_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.MAXBASIS:
            disp_points = self.transform.maxbasis_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.CONE:
            disp_points = self.transform.cone_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.DISP:
            if to_space == ColorSpaceType.VSH:
                hering = (np.linalg.inv(self.transform.hering_to_disp) @ points.T).T
                return self._hering_to_vsh(hering)
            elif to_space == ColorSpaceType.HERING:
                return (np.linalg.inv(self.transform.hering_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.MAXBASIS:
                return (self.transform.cone_to_disp @ points.T).T
            elif to_space == ColorSpaceType.RGB_OCV:
                return Conversion.Map4DTo6D(points, self.transform)
        elif from_space == ColorSpaceType.RGB_OCV:
            display = Conversion.Map6DTo4D(points, self.transform)
            return self.convert(display, ColorSpaceType.DISP, to_space)

        # If we reach here, the transformation is not defined
        raise ValueError(f"Transformation from {from_space} to {to_space} not implemented")

    def convert_to_perceptual(self, points: npt.NDArray,
                              from_space: str | ColorSpaceType,
                              M_basis: str | ColorSpaceType,
                              denom_of_nonlin: float) -> npt.NDArray:
        """Convert points from some linear space to a perceptual space

        Args:
            points (npt.NDArray): input points in from_space, that will be converted to to_space
            from_space (str | ColorSpaceType): The space to convert from
            M_basis (str | ColorSpaceType): M_basis to use as the basis of the transform
            denom_of_nonlin (float): denominator for the non-linearity

        Returns:
            npt.NDArray: Converted points in "perceptual space" 
        """
        # Convert to the basis space
        points = self.convert(points, from_space, M_basis)
        # Apply the non-linearity
        return np.power(points, 1/denom_of_nonlin)

    def convert_to_linear(self, points: npt.NDArray,
                          to_space: str | ColorSpaceType,
                          M_basis: str | ColorSpaceType,
                          denom_of_nonlin: float) -> npt.NDArray:
        # revert the power
        points = np.power(points, denom_of_nonlin)
        # Convert to the basis space
        return self.convert(points, M_basis, to_space)

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

    def get_RYGB_to_RGBOCV(self):
        """
        Get the transformation matrix from RYGB to RGB/OCV

        Returns:
            npt.NDArray: Transformation matrix
        """
        return GetMaxBasisToDisplayTransform(self.transform)

    def get_RYGB_to_sRGB(self):
        """
        Get the transformation matrix from RYGB to sRGB

        Returns:
            npt.NDArray: Transformation matrix
        """
        max_to_cone = np.linalg.inv(self.transform.cone_to_disp) @ self.transform.maxbasis_to_disp
        max_to_sRGB = M_XYZ_to_RGB @ self.transform.cone_to_XYZ @ max_to_cone
        print(max_to_sRGB)
        return max_to_sRGB

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


if __name__ == "__main__":
    # Test if Oklab implementation is correct
    observer = Observer.trichromat()

    import matplotlib.pyplot as plt
    from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
    from colour import SpectralShape
    shape = SpectralShape(min(observer.wavelengths), max(observer.wavelengths),
                          int(observer.wavelengths[1] - observer.wavelengths[0]))
    xyz = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].copy().align(shape).values
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Plot the original XYZ color matching functions
    axes[0].plot(observer.wavelengths, xyz)
    axes[0].set_title("CIE 1931 XYZ Color Matching Functions")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Value")

    # Plot the transformed OKLAB M1 values
    axes[1].plot(observer.wavelengths, xyz @ OKLAB_M1.T)
    axes[1].set_title("Transformed OKLAB M1 Values")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Value")

    # Plot the transformed OKLAB M1 values
    axes[2].plot(observer.wavelengths, np.cbrt(xyz @ OKLAB_M1.T))
    axes[2].set_title("Transformed OKLAB M1 Values (Cube Root)")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("Value")

    axes[3].plot(observer.wavelengths, np.cbrt(xyz @ OKLAB_M1.T)@OKLAB_M2.T)
    axes[3].set_title("Transformed OKLAB M2 Values")
    axes[3].set_xlabel("Wavelength (nm)")
    axes[3].set_ylabel("Value")

    plt.tight_layout()
    plt.show()

    cs = ColorSpace(observer)

    white_pt = np.array([1, 1, 1])  # observer.get_whitepoint()
    # Convert white point to XYZ
    print("White point in Cone", white_pt)
    print("White point in XYZ", cs.convert(white_pt, from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.XYZ))
    print("White point in OKlab", cs.convert(white_pt, from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.OKLAB))

    print(np.round(cs.convert(np.array([0.950, 1.0, 1.089]),
          from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))
    print(np.round(cs.convert(np.array([1, 0, 0]), from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))
    print(np.round(cs.convert(np.array([0, 1, 0]), from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))
    print(np.round(cs.convert(np.array([0, 0, 1]), from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))

    # Store the results of the conversions
    oklab_results = [
        np.round(cs.convert(np.array([0.950, 1.0, 1.089]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3),
        np.round(cs.convert(np.array([1, 0, 0]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3),
        np.round(cs.convert(np.array([0, 1, 0]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3),
        np.round(cs.convert(np.array([0, 0, 1]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3)
    ]

    # Convert back to XYZ and check equivalence
    for original, oklab in zip(
        [np.array([0.950, 1.0, 1.089]),
         np.array([1, 0, 0]),
         np.array([0, 1, 0]),
         np.array([0, 0, 1])],
        oklab_results
    ):
        converted_back = np.round(cs.convert(oklab, from_space=ColorSpaceType.OKLAB, to_space=ColorSpaceType.XYZ), 3)
        print(f"Original: {original}, Converted Back: {converted_back}, Equivalent: {np.allclose(original, converted_back)}")
