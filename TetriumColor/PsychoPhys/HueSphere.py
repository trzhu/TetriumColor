import numpy as np
from typing import List
import numpy.typing as npt
import os

from scipy.spatial import ConvexHull

from PIL import Image, ImageDraw
from TetriumColor.ColorMath import Geometry
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor
import TetriumColor.ColorMath.HueToDisplay as HueToDisplay
from TetriumColor.ColorMath.Geometry import ConvertCubeUVToXYZ, ExportGeometryToObjFile, GenerateGeometryFromVertices


# FIXME: This function is garbage, please fix it!!!!
def LMStoSRGB(lms: np.ndarray) -> np.ndarray:
    """
    Converts LMS values to sRGB values.

    Args:
        lms (np.ndarray): Array of LMS values, shape (N, 3) or (3,).

    Returns:
        np.ndarray: Array of sRGB values, shape (N, 3) or (3,).
    """
    # TODO: Fix this, it sucks and isn't correct.
    # LMS to XYZ conversion matrix
    lms_to_xyz = np.array([
        [0.4002, 0.7075, -0.0808],
        [-0.2263, 1.1653, 0.0457],
        [0.0000, 0.0000, 0.9182]
    ])

    xyz_d65 = np.array([0.9505, 1.0000, 1.0890])
    lms_d65 = np.array([1.0, 1.0, 1.0])

    # Scaling factors to align [1, 1, 1]_LMS with xyz_d65
    scaling_factors = xyz_d65 / np.dot(lms_to_xyz, lms_d65)

    # Scale the matrix
    lms_to_xyz = lms_to_xyz * scaling_factors[:, np.newaxis]

    # XYZ to linear sRGB conversion matrix
    xyz_to_srgb = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [0.0557, -0.2040,  1.0570]
    ])

    # Apply LMS to XYZ conversion
    xyz = np.dot(lms, lms_to_xyz.T)

    # Apply XYZ to linear sRGB conversion
    srgb_linear = np.dot(xyz, xyz_to_srgb.T)

    # Perform gamma correction to get sRGB
    def gamma_correction(channel):
        return np.where(channel <= 0.0031308,
                        12.92 * channel,
                        1.055 * np.power(channel, 1 / 2.4) - 0.055)

    # print(srgb_linear)
    # srgb = gamma_correction(np.clip(srgb_linear, 0, 1))

    # Clamp values to [0, 1] range
    srgb = np.clip(srgb_linear, 0, 1)

    return srgb


def GetHueSphereGeometryWithLineTexture(num_points: int, luminance: float, saturation: float, color_space_transform: ColorSpaceTransform,
                                        rgb_texture_filename: str, ocv_texture_filename: str, obj_filename: str) -> None:
    """GetHueSphereGeometryWithLineTexture generates a hue sphere geometry with fibonacci sampling and line texture

    Args:
        num_points (int): number of points to sample for the sphere
        luminance (float): luminance value
        saturation (float): saturation value
        color_space_transform (ColorSpaceTransform): color space transform object
        rgb_texture_filename (str): filename for the RGB texture
        ocv_texture_filename (str): filename for the OCV texture
        obj_filename (str): filename for the OBJ file
    """
    # ------ GEOMETRY CREATION
    vshh = HueToDisplay.SampleHueManifold(luminance, saturation, 4, num_points)
    # hering_pts = HueToDisplay.ConvertVSHToHering(vshh)

    remapped_vshh = HueToDisplay.RemapGamutPoints(vshh, color_space_transform,
                                                  HueToDisplay.GenerateGamutLUT(vshh, color_space_transform))
    chromaticity_space = HueToDisplay.ConvertVSHToHering(remapped_vshh)[:, 1:]
    tetra_colors: List[TetraColor] = HueToDisplay.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)

    vertices, triangles, normals, indices = GenerateGeometryFromVertices(chromaticity_space)
    tetra_colors = [tetra_colors[i] for i in indices]

    uv_coords = np.array([[i / len(vertices), i / len(vertices)] for i in range(len(vertices))])
    ExportGeometryToObjFile(vertices, triangles, normals, uv_coords, obj_filename)

    # ------- TEXTURE CREATION
    # Split color tuples into RGB and OCV components
    rgb_colors, ocv_colors = [], []
    for color in tetra_colors:
        rgb_colors.append(color.RGB)
        ocv_colors.append(color.OCV)
    rgb_colors = np.array(rgb_colors)
    ocv_colors = np.array(ocv_colors)

    # Save RGB texture as an image
    rgb_texture = (rgb_colors * 255).astype(np.uint8)
    rgb_texture_image = Image.fromarray(rgb_texture.reshape(1, -1, 3))
    rgb_texture_image.save(rgb_texture_filename)

    # Save OCV texture as an image
    ocv_texture = (ocv_colors * 255).astype(np.uint8)
    ocv_texture_image = Image.fromarray(ocv_texture.reshape(1, -1, 3))
    ocv_texture_image.save(ocv_texture_filename)


def GetHueSphereGeometryWithCubeMapTexture(luminance: float, saturation: float, color_space_transform: ColorSpaceTransform,
                                           image_size: int, filename_RGB: str, filename_OCV: str, filename_OBJ: str) -> None:
    """GetHueSphereGeometryWithCubeMapTexture generates a hue sphere geometry with cube map textures.

    Args:
        luminance (float): luminance value 
        saturation (float): saturation value
        color_space_transform (ColorSpaceTransform): color space transform object
        image_size (int): size of the image
        filename_RGB (str): filename for the RGB cube map texture
        filename_OCV (str): filename for the OCV cube map texture
        filename_OBJ (str): filename for the OBJ file
    """
    # ----- TEXTURE CREATION
    # Grid of UV coordinate that are the size of image_size
    all_us = (np.arange(image_size) + 0.5) / image_size
    all_vs = (np.arange(image_size) + 0.5) / image_size
    cube_u, cube_v = np.meshgrid(all_us, all_vs)
    flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

    # change the associated xyzs -> to a new direction, but the same color values
    qDirMat = HueToDisplay.GetTransformChromToQDir(color_space_transform)
    invQDirMat = np.linalg.inv(qDirMat)

    # Create the RGB/OCV GenerateCubeMapTextures
    img_rgb = Image.new('RGB', (6 * image_size, image_size))  # sample color per pixel to avoid empty spots
    img_ocv = Image.new('RGB', (6 * image_size, image_size))

    draw_rgb = ImageDraw.Draw(img_rgb)
    draw_ocv = ImageDraw.Draw(img_ocv)

    vertices = np.zeros((6, image_size * image_size, 3))
    uv_coords = np.zeros((6, image_size * image_size, 2))
    for i, cube_idx in enumerate([1, 4, 0, 5, 2, 3]):  # range(6):
        # convert the xyz coordinates of the cube map back into the original hering space -- this defines the
        # cubemap directions exactly !
        xyz = ConvertCubeUVToXYZ(cube_idx, cube_u, cube_v, saturation).reshape(-1, 3)
        xyz = np.dot(invQDirMat, xyz.T).T
        vertices[cube_idx] = xyz
        lum_vector = luminance * np.ones(image_size * image_size)

        vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))
        vshh = HueToDisplay.ConvertHeringToVSH(vxyz)

        map_angle_sat = HueToDisplay.GenerateGamutLUT(vshh, color_space_transform)
        remapped_vshh = HueToDisplay.RemapGamutPoints(vshh, color_space_transform, map_angle_sat)
        corresponding_tetracolors = HueToDisplay.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)
        for j in range(len(flattened_u)):
            u, v = flattened_u[j], flattened_v[j]
            color: TetraColor = corresponding_tetracolors[j]
            # swap to row major order by flipping them
            uv_coords[cube_idx, j] = ((i + v) / 6, u)
            rgb_color = (int(color.RGB[0] * 255), int(color.RGB[1] * 255), int(color.RGB[2] * 255))
            draw_rgb.point(((i * image_size) + v * image_size, u * image_size), fill=rgb_color)
            ocv_color = (int(color.OCV[0] * 255), int(color.OCV[1] * 255), int(color.OCV[2] * 255))
            draw_ocv.point(((i * image_size) + v * image_size,  u * image_size), fill=ocv_color)

    # Save the images
    img_rgb.save(filename_RGB)
    img_ocv.save(filename_OCV)

    # ----- GEOMETRY CREATION
    vertices, triangles, normals, indices = GenerateGeometryFromVertices(vertices.reshape(-1, 3))
    uv_coords = uv_coords.reshape(-1, 2)[indices]
    ExportGeometryToObjFile(vertices, triangles, normals, uv_coords, filename_OBJ)


def GenerateCubeMapTextures(luminance: float, saturation: float, color_space_transform: ColorSpaceTransform,
                            image_size: int, filename_RGB: str, filename_OCV: str, filename_sRGB: str):
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
    qDirMat = HueToDisplay.GetTransformChromToQDir(color_space_transform)
    invQDirMat = np.linalg.inv(qDirMat)

    # Create the RGB/OCV GenerateCubeMapTextures
    for i in range(6):
        img_rgb = Image.new('RGB', (image_size, image_size))  # sample color per pixel to avoid empty spots
        img_ocv = Image.new('RGB', (image_size, image_size))
        img_srgb = Image.new('RGB', (image_size, image_size))

        draw_rgb = ImageDraw.Draw(img_rgb)
        draw_ocv = ImageDraw.Draw(img_ocv)
        draw_srgb = ImageDraw.Draw(img_srgb)

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
        hering_to_cone = np.linalg.inv(color_space_transform.cone_to_disp)@color_space_transform.hering_to_disp
        lms = (hering_to_cone@HueToDisplay.ConvertVSHToHering(remapped_vshh).T).T[:, [3, 1, 0]]
        corresponding_srgbs = LMStoSRGB(lms)

        for j in range(len(flattened_u)):
            u, v = flattened_v[j], flattened_u[j]  # swap axis for PIL
            color: TetraColor = corresponding_tetracolors[j]
            rgb_color = (int(color.RGB[0] * 255), int(color.RGB[1] * 255), int(color.RGB[2] * 255))
            draw_rgb.point((u * image_size, v * image_size), fill=rgb_color)
            ocv_color = (int(color.OCV[0] * 255), int(color.OCV[1] * 255), int(color.OCV[2] * 255))
            draw_ocv.point((u * image_size, v * image_size), fill=ocv_color)

            srgb = corresponding_srgbs[j]
            srgb_color = (int(srgb[0] * 255), int(srgb[1] * 255), int(srgb[2] * 255))
            draw_srgb.point((u * image_size, v * image_size), fill=srgb_color)

        # Save the images
        img_rgb.save(f'{filename_RGB}_{str(i)}.png')
        img_ocv.save(f'{filename_OCV}_{str(i)}.png')
        img_srgb.save(f'{filename_sRGB}_{str(i)}.png')


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
    cubemap_image.save(output_filename)
    print(f"Saved concatenated cubemap to {output_filename}")


def CreateCircleGrid(grid: npt.NDArray, padding: int, radius: int, output_base: str):
    """
    Creates two images from grids of colors, where each grid cell is represented by a circle.

    Args:
        grid (np.ndarray): First grid of colors with shape (s, s, 2, dim).
        padding (int): Padding around the grid in pixels.
        radius (int): Radius of each circle in pixels.
        output_base (str): Base filename for the output images, consisting of RGB OCV pairs.
    """
    def __createPairImages(metamer_idx):
        s = grid.shape[0]  # Grid size (s x s)

        # Image size calculation
        cell_size = 2 * radius  # Each cell is defined by the diameter of the circle
        grid_size = s * cell_size + (s + 1) * padding
        image_size = (grid_size, grid_size)

        # Create blank image
        img_RGB = Image.new("RGB", image_size, "black")
        img_OCV = Image.new("RGB", image_size, "black")

        draw_RGB = ImageDraw.Draw(img_RGB)
        draw_OCV = ImageDraw.Draw(img_OCV)

        # Loop over the grid
        for i in range(s):
            for j in range(s):
                # Compute circle center
                cx = padding + j * (cell_size + padding) + radius
                cy = padding + i * (cell_size + padding) + radius

                # Convert color to tuple and scale (assuming colors are normalized [0, 1])
                color = tuple((grid[i, j, metamer_idx, :3] * 255).astype(int))  # Use the first color in the pair
                draw_RGB.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline="black")

                color = tuple((grid[i, j, metamer_idx, 3:] * 255).astype(int))  # Use the first color in the pair
                draw_OCV.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline="black")

            # Save image
        img_RGB.save(output_base + f'_{metamer_idx}_RGB_.png')
        img_OCV.save(output_base + f'_{metamer_idx}_OCV_.png')

    __createPairImages(0)
    __createPairImages(1)
