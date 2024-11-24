import numpy as np
from importlib import resources
import os

from typing import List
import numpy.typing as npt

from PIL import Image, ImageDraw
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor
import TetriumColor.ColorMath.HueToDisplay as HueToDisplay

# vectorized cubemap utilities


def ConvertXYZToCubeUV(x, y, z):
    # Compute absolute values
    absX = np.abs(x)
    absY = np.abs(y)
    absZ = np.abs(z)

    # Determine the positive and dominant axes
    isXPositive = x > 0
    isYPositive = y > 0
    isZPositive = z > 0

    # Initialize arrays for index, u, v
    index = np.zeros_like(x, dtype=int)
    maxAxis = np.zeros_like(x, dtype=float)
    uc = np.zeros_like(x, dtype=float)
    vc = np.zeros_like(x, dtype=float)

    # POSITIVE X
    mask = isXPositive & (absX >= absY) & (absX >= absZ)
    maxAxis[mask] = absX[mask]
    uc[mask] = -z[mask]
    vc[mask] = y[mask]
    index[mask] = 0

    # NEGATIVE X
    mask = ~isXPositive & (absX >= absY) & (absX >= absZ)
    maxAxis[mask] = absX[mask]
    uc[mask] = z[mask]
    vc[mask] = y[mask]
    index[mask] = 1

    # POSITIVE Y
    mask = isYPositive & (absY >= absX) & (absY >= absZ)
    maxAxis[mask] = absY[mask]
    uc[mask] = x[mask]
    vc[mask] = -z[mask]
    index[mask] = 2

    # NEGATIVE Y
    mask = ~isYPositive & (absY >= absX) & (absY >= absZ)
    maxAxis[mask] = absY[mask]
    uc[mask] = x[mask]
    vc[mask] = z[mask]
    index[mask] = 3

    # POSITIVE Z
    mask = isZPositive & (absZ >= absX) & (absZ >= absY)
    maxAxis[mask] = absZ[mask]
    uc[mask] = x[mask]
    vc[mask] = y[mask]
    index[mask] = 4

    # NEGATIVE Z
    mask = ~isZPositive & (absZ >= absX) & (absZ >= absY)
    maxAxis[mask] = absZ[mask]
    uc[mask] = -x[mask]
    vc[mask] = y[mask]
    index[mask] = 5

    # Convert range from -1 to 1 to 0 to 1
    u = 0.5 * (uc / maxAxis + 1.0)
    v = 0.5 * (vc / maxAxis + 1.0)

    return index, u, v


def ConvertCubeUVToXYZ(index, u, v, radius) -> npt.NDArray:
    """
    Convert cube UV coordinates back to XYZ with all points at a specified radius from the origin.
    """
    # Convert range 0 to 1 to -1 to 1
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    # Initialize x, y, z
    x = np.zeros_like(u)
    y = np.zeros_like(u)
    z = np.zeros_like(u)

    # POSITIVE X
    mask = index == 0
    x[mask], y[mask], z[mask] = 1.0, vc[mask], -uc[mask]

    # NEGATIVE X
    mask = index == 1
    x[mask], y[mask], z[mask] = -1.0, vc[mask], uc[mask]

    # POSITIVE Y
    mask = index == 2
    x[mask], y[mask], z[mask] = uc[mask], 1.0, -vc[mask]

    # NEGATIVE Y
    mask = index == 3
    x[mask], y[mask], z[mask] = uc[mask], -1.0, vc[mask]

    # POSITIVE Z
    mask = index == 4
    x[mask], y[mask], z[mask] = uc[mask], vc[mask], 1.0

    # NEGATIVE Z
    mask = index == 5
    x[mask], y[mask], z[mask] = -uc[mask], vc[mask], -1.0

    # Normalize to unit sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    x = (x / norm) * radius
    y = (y / norm) * radius
    z = (z / norm) * radius

    return np.array([x, y, z]).T


def __rotateToZAxis(vector: npt.NDArray) -> npt.NDArray:
    """
    Returns a rotation matrix that rotates the given vector to align with the Z-axis.

    Parameters:
        vector (array-like): The input vector to align with the Z-axis.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # Normalize the input vector
    v = np.array(vector, dtype=float)
    v = v / np.linalg.norm(v)

    # Z-axis unit vector
    z_axis = np.array([0, 0, 1], dtype=float)

    # Compute the axis of rotation (cross product)
    axis = np.cross(v, z_axis)
    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        # The vector is already aligned with the Z-axis
        return np.eye(3)

    axis = axis / axis_norm  # Normalize the axis

    # Compute the angle of rotation (dot product)
    angle = np.arccos(np.dot(v, z_axis))

    # Compute the skew-symmetric cross-product matrix for the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Use the Rodrigues' rotation formula
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def GetTransformChromToQDir(transform: ColorSpaceTransform):
    """
    Get the transformation matrix from chromaticity to the metameric direction.
    """
    shh = HueToDisplay.GetMetamericAxisInVSH(transform)[0, 1:]  # remove luminance
    return __rotateToZAxis(shh)


def GenerateCubeMapTextures(luminance: float, saturation: float, color_space_transform: ColorSpaceTransform,
                            image_size: int, filename_RGB: str, filename_OCV: str):
    """GenerateCubeMapTextures generates the cube map textures for a given luminance and saturation.

    Args:
        luminance (float): luminance value
        saturation (float): saturation value
        color_space_transform (ColorSpaceTransform): color space transform object
        filename_RGB (str): filename for the RGB cube map texture
        filename_OCV (str): filename for the OpenCV cube map texture
    Returns:
        Saves the generated textures to the specified filenames.
    """
    # Grid of UV coordinate that are the size of image_size
    all_us = (np.arange(image_size) + 0.5) / image_size
    all_vs = (np.arange(image_size) + 0.5) / image_size
    cube_u, cube_v = np.meshgrid(all_us, all_vs)
    flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

    # change the associated xyzs -> to a new direction, but the same color values
    qDirMat = GetTransformChromToQDir(color_space_transform)
    invQDirMat = np.linalg.inv(qDirMat)

    # Create the RGB/OCV GenerateCubeMapTextures
    for i in range(6):
        img_rgb = Image.new('RGB', (image_size, image_size))  # sample color per pixel to avoid empty spots
        img_ocv = Image.new('RGB', (image_size, image_size))

        draw_rgb = ImageDraw.Draw(img_rgb)
        draw_ocv = ImageDraw.Draw(img_ocv)

        # convert the xyz coordinates of the cube map back into the original hering space -- this defines the
        # cubemap directions exactly !
        xyz = ConvertCubeUVToXYZ(i, cube_u, cube_v, saturation).reshape(-1, 3)
        xyz = np.dot(invQDirMat, xyz.T).T
        lum_vector = luminance * np.ones(image_size * image_size)

        vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))
        vshh = HueToDisplay.ConvertHeringToVSH(vxyz)

        map_angle_sat = HueToDisplay.GenerateGamutLUT(vshh, color_space_transform)
        remapped_vshh = HueToDisplay.RemapGamutPoints(vshh, color_space_transform, map_angle_sat)
        corresponding_tetracolors = HueToDisplay.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)

        for j in range(len(flattened_u)):
            u, v = flattened_u[j], flattened_v[j]
            color: TetraColor = corresponding_tetracolors[j]
            rgb_color = (int(color.RGB[0] * 255), int(color.RGB[1] * 255), int(color.RGB[2] * 255))
            draw_rgb.point((u * image_size, v * image_size), fill=rgb_color)
            ocv_color = (int(color.OCV[0] * 255), int(color.OCV[1] * 255), int(color.OCV[2] * 255))
            draw_ocv.point((u * image_size, v * image_size), fill=ocv_color)
        # Save the images
        img_rgb.save(f'{filename_RGB}_{str(i)}.png')
        img_ocv.save(f'{filename_OCV}_{str(i)}.png')


def ConcatenateCubeMap(basename: str, output_filename: str):
    """
    Concatenate cubemap textures into a single cross-layout image with correct orientation.

    Parameters:
        basename (str): The base name of the input files, e.g., "texture". Files are assumed to follow the format "<basename>_i.png".
                       `i` corresponds to the index: 0 (+X), 1 (-X), 2 (+Y), 3 (-Y), 4 (+Z), 5 (-Z).
        output_filename (str): The output file name for the concatenated image.
    """
    # Load images for each face
    faces = []
    for i in range(6):
        filename = f"{basename}_{i}.png"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing cubemap texture: {filename}")
        faces.append(Image.open(filename))

    # Assume all faces are the same size
    face_width, face_height = faces[0].size

    # Create a blank image for the cross layout
    width = 4 * face_width
    height = 3 * face_height
    cubemap_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # +X (0)
    cubemap_image.paste(faces[0].rotate(90), (2 * face_width, face_height))
    # -X (1) flipped horizontally
    cubemap_image.paste(faces[1].rotate(90), (0, face_height))
    # +Y (2) flipped vertically
    cubemap_image.paste(faces[2].rotate(90), (face_width, 0))
    # -Y (3) flipped vertically
    cubemap_image.paste(faces[3].rotate(90), (face_width, 2 * face_height))
    # +Z (4)
    cubemap_image.paste(faces[4].rotate(90), (face_width, face_height))
    # -Z (5) flipped horizontally
    cubemap_image.paste(faces[5].rotate(90), (3 * face_width, face_height))

    # Save the concatenated image
    cubemap_image.save(output_filename)
    print(f"Saved concatenated cubemap to {output_filename}")
