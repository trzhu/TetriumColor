import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple, Union, Optional
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import pickle
import hashlib

from importlib import resources
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Utils.CustomTypes import TetraColor, PlateColor
import TetriumColor.ColorMath.Geometry as Geometry
from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation, FindMaximumIn1DimDirection
from TetriumColor.ColorMath import GamutMath


class ColorSampler:
    """
    A class for sampling colors in various ways from a color space.

    This class centralizes color sampling functionality and handles efficient
    gamut mapping by computing and caching the gamut boundary information.
    """

    def __init__(self, color_space: ColorSpace, cubemap_size: int = 64):
        """
        Initialize the ColorSampler with a ColorSpace.

        Parameters:
            color_space (ColorSpace): The color space to sample from
        """
        self.color_space = color_space
        self._gamut_cubemap = None
        self._lum_range = None
        self._sat_range = None
        self._cubemap_size = cubemap_size  # Default size for the cubemap
        self._max_L = color_space.max_L

        # Try to load cubemap from cache during initialization
        if not self._load_from_cache():
            print("Failed to load cubemap from cache, generating new cubemap")
            self.get_gamut_lut()

    def _get_cache_filename(self) -> str:
        """
        Generate a unique filename for caching the cubemap based on the color space.

        Returns:
            str: Cache filename
        """
        # Add cubemap size to the hash input
        hash_input = f"{str(self.color_space)}|cubemap_size:{self._cubemap_size}"

        # Create MD5 hash
        hash_obj = hashlib.md5(hash_input.encode())
        hash_str = hash_obj.hexdigest()

        return f"cubemap_{hash_str}.pkl"

    def _save_to_cache(self) -> None:
        """Save cubemap and range data to cache."""
        if self._gamut_cubemap is None:
            return

        # Save data
        cache_data = {
            'cubemap': self._gamut_cubemap,
            'lum_range': self._lum_range,
            'sat_range': self._sat_range
        }

        _cache_file = self._get_cache_filename()
        try:
            with resources.path("TetriumColor.Assets.Cache", _cache_file) as path:
                with open(path, "wb") as f:
                    pickle.dump(cache_data, f)
            print(f"Saved cubemap cache to {_cache_file}")
        except Exception as e:
            print(f"Failed to save cubemap cache: {e}")

    def _load_from_cache(self) -> bool:
        """
        Load cubemap and range data from cache.

        Returns:
            bool: True if data was successfully loaded, False otherwise
        """
        # Get cache filename
        _cache_file = self._get_cache_filename()
        with resources.path("TetriumColor.Assets.Cache", _cache_file) as path:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        cache_data = pickle.load(f)
                    self._gamut_cubemap = cache_data['cubemap']
                    self._lum_range = cache_data['lum_range']
                    self._sat_range = cache_data['sat_range']
                    return True
                except Exception as e:
                    print(f"Failed to load cubemap from cache: {e}")
                    return False
            else:
                return False

    def _generate_gamut_cubemap(self, cubemap_size: Optional[int] = None) -> Tuple[Dict, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Generate a cubemap representation of the gamut boundaries.

        Parameters:
            cubemap_size (int, optional): Size of the cubemap images

        Returns:
            Tuple containing:
                - Dict of cubemap images
                - Tuple of luminance and saturation ranges
        """
        if cubemap_size:
            self._cubemap_size = cubemap_size

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
        for i in tqdm(range(6), desc="Generating cubemap"):
            # Convert UV to XYZ coordinates for this cube face
            xyz = Geometry.ConvertCubeUVToXYZ(i, cube_u, cube_v, 1).reshape(-1, 3)
            xyz = np.dot(invMetamericDirMat, xyz.T).T

            # Create hering coordinates with unit luminance
            lum_vector = np.ones(self._cubemap_size * self._cubemap_size)
            vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))

            # Convert to VSH space
            vshh = self.color_space._hering_to_vsh(vxyz)

            # Generate gamut LUT for this face
            face_dict = {}
            for j in range(len(flattened_u)):
                u, v = flattened_v[j], flattened_u[j]
                angle = tuple(vshh[j, 2:])
                # Find cusp point for this hue angle
                hue_cartesian = self.color_space._vsh_to_hering(np.array([[0, 1, *angle]]))
                max_sat_point = self._find_maximal_saturation(
                    (self.color_space.transform.hering_to_disp @ hue_cartesian.T).T[0])
                max_sat_hering = np.linalg.inv(self.color_space.transform.hering_to_disp) @ max_sat_point
                max_sat_vsh = self.color_space._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
                lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]
                face_dict[(u, v)] = (lum_cusp, sat_cusp)

            lut_dicts.append(face_dict)

        # Compute overall ranges for normalization
        all_values = np.array([list(lut_dicts[i].values()) for i in range(6)]).reshape(-1, 2)
        lum_min, lum_max = np.min(all_values[:, 0]), np.max(all_values[:, 0])
        sat_min, sat_max = np.min(all_values[:, 1]), np.max(all_values[:, 1])

        # Generate cubemap images
        cubemap_images = {}
        for i in range(6):
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

    def get_gamut_lut(self, force_recompute: bool = False) -> None:
        """
        Get the gamut lookup table, represented as a cubemap.

        Tries to load from cache first, or computes and caches if not available.

        Parameters:
            force_recompute (bool): Whether to force recomputation
        """
        if force_recompute:
            # Skip cache if forced to recompute
            self._gamut_cubemap, ranges = self._generate_gamut_cubemap()
            self._lum_range, self._sat_range = ranges
            # Save to cache for future use
            self._save_to_cache()
        elif self._gamut_cubemap is None:
            # Try loading from cache first (already tried during init)
            # If not loaded, generate and save
            self._gamut_cubemap, ranges = self._generate_gamut_cubemap()
            self._lum_range, self._sat_range = ranges
            self._save_to_cache()

    def _get_transform_chrom_to_metameric_dir(self) -> npt.NDArray:
        """
        Get the transformation matrix from chromatic coordinates to metameric direction.

        Returns:
            npt.NDArray: Transformation matrix
        """
        normalized_direction = self.color_space.get_metameric_axis_in(ColorSpaceType.HERING)
        return Geometry.RotateToZAxis(normalized_direction[1:])

    def _angles_to_cube_uv(self, angles: tuple[float, float]):
        """
        Convert angles to cube face index and UV coordinates.

        Parameters:
            angles (tuple): Angles to convert

        Returns:
            Tuple[int, float, float]: Cube face index and UV coordinates
        """
        angles_with_ones = np.array([[0, 1, *angles]])
        x, y, z = self.color_space.convert(angles_with_ones, ColorSpaceType.VSH, ColorSpaceType.HERING)[0, 1:]
        face_id, u, v = Geometry.ConvertXYZToCubeUV(x, y, z)
        return int(face_id), float(u), float(v)

    def _interpolate_from_cubemap(self, angles: tuple) -> Tuple[float, float]:
        """
        Interpolate luminance and saturation values from the cubemap for given angles.

        Parameters:
            angles (tuple): Angles for which to interpolate values

        Returns:
            Tuple[float, float]: Interpolated luminance and saturation values
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

    def _find_maximal_saturation(self, hue_direction: npt.NDArray) -> npt.NDArray:
        """
        Find the point with maximal saturation in the given hue direction.

        Parameters:
            hue_direction (npt.NDArray): Hue direction vector

        Returns:
            npt.NDArray: Point with maximal saturation
        """
        from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation
        result = FindMaximalSaturation(hue_direction, np.eye(self.color_space.dim))
        if result is None:
            raise ValueError("Failed to find maximal saturation point")
        return result

    def compute_max_sat_at_luminance(self, luminance: float, angle: tuple) -> float:
        """
        Compute the maximum saturation at a given luminance for a specific angle directly.

        This method avoids using the cubemap and calculates the value precisely.

        Parameters:
            luminance (float): Luminance value
            angle (tuple): Hue angle

        Returns:
            float: Maximum saturation at the given luminance and angle
        """
        # Find the cusp point for this hue angle
        hue_cartesian = self.color_space._vsh_to_hering(np.array([[0, 1, *angle]]))
        max_sat_point = self._find_maximal_saturation(
            (self.color_space.transform.hering_to_disp @ hue_cartesian.T).T[0]
        )
        max_sat_hering = np.linalg.inv(self.color_space.transform.hering_to_disp) @ max_sat_point
        max_sat_vsh = self.color_space._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
        lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]

        # Calculate the maximum saturation at the given luminance
        return self._solve_for_boundary(luminance, self._max_L, lum_cusp, sat_cusp)

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
            return (L - max_L) / slope
        else:
            slope = lum_cusp / sat_cusp
            return L / slope

    def max_sat_at_luminance(self, luminance: float,
                             angles: Union[List[tuple], tuple]) -> Union[float, List[float]]:
        """
        Get the maximum saturation at a given luminance for specific angle(s).

        Uses the cubemap for interpolation if available, otherwise computes directly.

        Parameters:
            luminance (float): Luminance value
            angles (tuple or List[tuple]): Hue angle(s)

        Returns:
            float or List[float]: Maximum saturation at the given luminance for each angle
        """
        # Handle single angle vs list of angles
        is_single = isinstance(angles, tuple)
        angle_list = [angles] if is_single else angles

        # If cubemap is available, use interpolation (faster)
        if self._gamut_cubemap is not None:
            sat_maxes = []
            for angle in angle_list:
                lum_cusp, sat_cusp = self._interpolate_from_cubemap(angle)
                sat_max = self._solve_for_boundary(luminance, self._max_L, lum_cusp, sat_cusp)
                sat_maxes.append(sat_max)
        # Otherwise compute directly (more accurate)
        else:
            sat_maxes = [self.compute_max_sat_at_luminance(luminance, angle) for angle in angle_list]

        return sat_maxes[0] if is_single else sat_maxes

    def remap_to_gamut(self, vshh: npt.NDArray) -> npt.NDArray:
        """
        Remap points to be within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Remapped points that are in gamut
        """
        # Copy the input to avoid modifying it
        remapped_vshh = vshh.copy()

        # Remap each point
        for i in range(len(remapped_vshh)):
            angle = tuple(remapped_vshh[i, 2:])

            # Calculate the maximum saturation at the given luminance
            sat_max = self.max_sat_at_luminance(remapped_vshh[i, 0], angle)

            # Clamp the saturation to the maximum
            remapped_vshh[i, 1] = min(sat_max, remapped_vshh[i, 1])

        return remapped_vshh

    def is_in_gamut(self, vshh: npt.NDArray) -> Union[bool, npt.NDArray]:
        """
        Check if points are within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            bool or npt.NDArray: Boolean indicating if point(s) are in gamut
        """
        remapped = self.remap_to_gamut(vshh)

        # If single point
        if vshh.ndim == 1:
            return np.allclose(vshh, remapped, rtol=1e-05, atol=1e-08)

        # Check if any coordinates changed for multiple points
        return np.all(np.isclose(vshh, remapped, rtol=1e-05, atol=1e-08), axis=1)

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
        all_angles = Geometry.SampleAnglesEqually(num_points, self.color_space.dim-1)
        all_vshh = np.zeros((len(all_angles), self.color_space.dim))
        all_vshh[:, 0] = luminance
        all_vshh[:, 1] = saturation
        all_vshh[:, 2:] = all_angles
        return all_vshh

    def sample_equiluminant_plane(self, luminance: float, num_points: int = 100,
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

    # need to write an equivalent function for sampling the boundary of the OBS solid
    def sample_full_colors(self, num_points=10000) -> npt.NDArray:
        """Generate the Full Colors Boundary of the Object Color Solid

        Args:
            num_points (int, optional): number of points to generate the boundary of. Defaults to 10000.

        Returns:
            npt.NDArray: Array of full colors in Hering Space
        """
        # for every hue
        vshh: npt.NDArray = self.sample_hue_manifold(1, 0.5, num_points)
        all_disp_points = self.color_space.convert(vshh, ColorSpaceType.VSH, ColorSpaceType.DISP)

        # For every point, find the reflectance of maximum saturation
        generating_vecs = self.color_space.observer.get_normalized_sensor_matrix(wavelengths=np.arange(360, 831, 1)).T
        pts = []
        for pt in tqdm(all_disp_points):
            res = FindMaximalSaturation(pt, generating_vecs=generating_vecs)
            if res is not None:
                pts += [res]
        max_sat_cartesian_per_angle = np.array(pts)

        # return point of max saturation for every hue
        return self.color_space.convert(max_sat_cartesian_per_angle, ColorSpaceType.DISP, ColorSpaceType.HERING)

    @staticmethod
    def _concatenate_cubemap(faces):
        """
        Concatenate cubemap textures into a single cross-layout image with correct orientation.

        Parameters:
            basename (str): The base name of the input files, e.g., "texture". Files are assumed to follow the format "<basename>_i.png".
                        `i` corresponds to the index: 0 (+X), 1 (-X), 2 (+Y), 3 (-Y), 4 (+Z), 5 (-Z).
        """
        # Assume all faces are the same size
        face_width, face_height = faces[0].size

        # Create a blank image for the cross layout
        width = 4 * face_width
        height = 3 * face_height
        cubemap_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # +X (0)
        cubemap_image.paste(faces[0], (2 * face_width, face_height))
        # -X (1) flipped horizontally
        cubemap_image.paste(faces[1], (0, face_height))
        # +Y (2) flipped vertically
        cubemap_image.paste(faces[3], (face_width, 0))  # swap 2 and 3 because of the flipped orientation i think
        # -Y (3) flipped vertically
        cubemap_image.paste(faces[2], (face_width, 2 * face_height))
        # +Z (4)
        cubemap_image.paste(faces[4], (face_width, face_height))
        # -Z (5) flipped horizontally
        cubemap_image.paste(faces[5], (3 * face_width, face_height))

        # Save the concatenated image
        return cubemap_image

    def generate_cubemap(self, luminance: float, saturation: float,
                         display_color_space: ColorSpaceType = ColorSpaceType.SRGB) -> Image.Image:
        """Generate a cubemap within the gamut boundaries

        Args:
            luminance (float): luminance
            saturation (float): saturation
            display_color_space (ColorSpaceType, optional): color space that you want to transform to. Defaults to ColorSpaceType.SRGB.

        Returns:
            PIL.Image.Image: Image of the cubemap
        """

        # Generate grid of UV coordinates
        all_us = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        all_vs = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        cube_u, cube_v = np.meshgrid(all_us, all_vs)

        # Get metameric direction matrix
        metamericDirMat = self._get_transform_chrom_to_metameric_dir()
        invMetamericDirMat = np.linalg.inv(metamericDirMat)

        # Process each face of the cube
        cubemap_images = []
        for i in tqdm(range(6), desc="Generating cubemap"):
            # Convert UV to XYZ coordinates for this cube face
            xyz = Geometry.ConvertCubeUVToXYZ(i, cube_u, cube_v, 1).reshape(-1, 3)
            xyz = np.dot(invMetamericDirMat, xyz.T).T

            max_saturations = np.array(self._gamut_cubemap[i]).reshape(-1, 3)[:, 1]/255
            normalized_saturations = (
                max_saturations * (self._sat_range[1] - self._sat_range[0])) + self._sat_range[0]

            # Create hering coordinates with unit luminance
            lum_vector = np.ones(self._cubemap_size * self._cubemap_size) * luminance
            vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))

            # Convert to VSH space
            vshh = self.color_space.convert(vxyz, ColorSpaceType.HERING, ColorSpaceType.VSH)
            vshh[:, 1] = np.min(
                np.vstack((np.full(normalized_saturations.shape, saturation), normalized_saturations)), axis=0)
            remapped_points = self.remap_to_gamut(vshh)
            corresponding_colors = self.color_space.convert(remapped_points, ColorSpaceType.VSH, display_color_space)
            # Convert colors to 8-bit format and reshape for image saving
            corresponding_colors = np.clip(corresponding_colors, 0, 1) * 255
            corresponding_colors = corresponding_colors.astype(np.uint8)
            corresponding_colors = corresponding_colors.reshape(
                self._cubemap_size, self._cubemap_size, 3).transpose(1, 0, 2)

            # Create an image from the array
            cubemap_images += [Image.fromarray(corresponding_colors, 'RGB')]

        return self._concatenate_cubemap(cubemap_images)

    def to_tetra_color(self, vsh_points: npt.NDArray) -> List[TetraColor]:
        """
        Convert VSH points to TetraColor objects.

        Parameters:
            vsh_points (npt.NDArray): Points in VSH space

        Returns:
            List[TetraColor]: List of TetraColor objects
        """
        # Convert to RGB_OCV space
        six_d_color = self.color_space.convert(vsh_points, ColorSpaceType.VSH, ColorSpaceType.DISP_6P)

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
        six_d_colors = self.color_space.convert(points, ColorSpaceType.VSH, ColorSpaceType.DISP_6P)

        # Create TetraColor objects for foreground and background
        foreground = TetraColor(six_d_colors[0, :3], six_d_colors[0, 3:])
        background = TetraColor(six_d_colors[1, :3], six_d_colors[1, 3:])

        # Return the PlateColor
        return PlateColor(foreground, background)
