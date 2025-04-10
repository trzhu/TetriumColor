import math
from PIL import Image
import numpy as np
from typing import Callable, List
from numpy.random import normal
import numpy.typing as npt
import os

from PIL import Image
from TetriumColor.ColorMath import Geometry
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor
import TetriumColor.ColorMath.GamutMath as GamutMath
from TetriumColor.ColorMath.Geometry import ConvertCubeUVToXYZ, ConvertPolarToCartesian, ExportGeometryToObjFile, GenerateGeometryFromVertices
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator


def GetSphereGeometry(luminance: float, saturation: float, num_points: int, filename: str, color_space_transform: ColorSpaceTransform):
    """
    GetSphereGeometry generates a sphere geometry with fibonacci sampling.

    Args:
        num_points (int): number of points to sample for the sphere
        filename (str): filename for the OBJ file
    """
    # Generate sphere vertices
    vshh = GamutMath.SampleHueManifold(luminance, saturation, 4, num_points)
    hering = GamutMath.ConvertVSHToHering(vshh)
    vertices, triangles, normals, _ = Geometry.GenerateGeometryFromVertices(hering[:, 1:])
    uv_coords = Geometry.CartesianToUV(vertices)
    print(len(uv_coords))
    remapped_vshh = GamutMath.RemapGamutPoints(vshh, color_space_transform,
                                               GamutMath.GenerateGamutLUT(vshh, color_space_transform))
    tetra_colors: List[TetraColor] = GamutMath.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)

    rgb_colors = np.array([color.RGB for color in tetra_colors])
    # Export geometry to OBJ file
    ExportGeometryToObjFile(vertices, triangles, normals, uv_coords, rgb_colors, filename)

    return vertices, triangles, normals, uv_coords, rgb_colors


def CreatePaddedGrid(image_files, grid_size=None, padding=10, bg_color=(0, 0, 0)):
    """
    Create a padded grid of images from a list of image files.

    Args:
        image_files (list of str): List of image file paths.
        grid_size (tuple, optional): Tuple (rows, cols) specifying grid dimensions.
                                     If None, grid is square.
        padding (int, optional): Padding between images in pixels. Defaults to 10.
        bg_color (tuple, optional): Background color for the grid (R, G, B). Defaults to white.

    Returns:
        Image: The grid as a Pillow Image object.
    """
    # Load all images
    images = [Image.open(file) for file in image_files]

    # Ensure all images are the same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    resized_images = [img.resize((max_width, max_height)) for img in images]

    # Determine grid size if not provided
    num_images = len(images)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
    else:
        rows, cols = grid_size

    # Calculate final grid dimensions
    grid_width = cols * max_width + (cols - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding

    # Create a blank canvas for the grid
    grid_image = Image.new("RGB", (grid_width, grid_height), bg_color)

    # Paste images onto the grid
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        grid_image.paste(img, (x, y))

    return grid_image


def CreatePseudoIsochromaticGrid(grid, output_dir: str, output_base: str, seed: int = 42, noise_generator: BackgroundNoiseGenerator | None = None):
    subdirname = f"./{output_dir}/sub_images"
    os.makedirs(subdirname, exist_ok=True)
    plate: IshiharaPlateGenerator = IshiharaPlateGenerator(seed=seed)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            metamer1 = TetraColor(grid[i, j, 0, :3], grid[i, j, 0, 3:])
            metamer2 = TetraColor(grid[i, j, 1, :3], grid[i, j, 1, 3:])
            plate_color = PlateColor(metamer1, metamer2)
            noise_generator_fn = noise_generator.GenerateNoiseFunction(plate_color) if noise_generator else None
            plate.GeneratePlate(seed, -1, plate_color, noise_generator_fn)
            plate.ExportPlate(os.path.join(subdirname, f"{output_base}_{i}_{j}_RGB.png"),
                              os.path.join(subdirname, f"{output_base}_{i}_{j}_OCV.png"))

    img_rgb = CreatePaddedGrid([os.path.join(subdirname, f"{output_base}_{i}_{j}_RGB.png") for i in range(grid.shape[0])
                                for j in range(grid.shape[1])])
    img_rgb = img_rgb.resize((1024, 1024), Image.Resampling.BOX)
    img_rgb.save(f"./{output_dir}/{output_base}_RGB.png")

    img_ocv = CreatePaddedGrid([os.path.join(subdirname, f"{output_base}_{i}_{j}_OCV.png") for i in range(grid.shape[0])
                                for j in range(grid.shape[1])])
    img_ocv = img_ocv.resize((1024, 1024), Image.Resampling.BOX)
    img_ocv.save(f"./{output_dir}/{output_base}_OCV.png")


def CreatePseudoIsochromaticImages(colors, output_dir: str, output_base: str, names: List[str], seed=42, noise_generator: List[BackgroundNoiseGenerator | None] | None = None, sub_image_dir: str = "sub_images"):
    """Generate images in an output dir

    Args:
        colors (_type_): _description_
        output_dir (str): _description_
        output_base (str): _description_
        seed (int, optional): _description_. Defaults to 42.
    """
    subdirname = f"./{output_dir}/{sub_image_dir}"
    os.makedirs(subdirname, exist_ok=True)
    plate: IshiharaPlateGenerator = IshiharaPlateGenerator(seed=seed)
    chars = "ZABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(len(colors)):
        metamer1 = TetraColor(colors[i, 0, :3], colors[i, 0, 3:])
        metamer2 = TetraColor(colors[i, 1, :3], colors[i, 1, 3:])
        plate_color = PlateColor(metamer1, metamer2)
        plate.GeneratePlate(seed, -1, plate_color)
        plate.DrawCorner(chars[i])
        plate.ExportPlate(os.path.join(subdirname, f"{output_base}_{names[i]}_RGB.png"),
                          os.path.join(subdirname, f"{output_base}_{names[i]}_OCV.png"))
