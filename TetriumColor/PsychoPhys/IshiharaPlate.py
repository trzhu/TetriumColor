import numpy as np
import packcircles
import importlib.resources as resources
from importlib.resources import as_file

from typing import Callable, List, Tuple, Optional
import numpy.typing as npt

from PIL import Image, ImageDraw
from TetriumColor.Utils.CustomTypes import PlateColor, TetraColor
from PIL import ImageFont
from pathlib import Path


# Available hidden numbers
_SECRETS = list(range(10, 100))


def _standardize_color(color: TetraColor) -> np.ndarray:
    """
    Ensure a TetraColor is a float in [0, 1].

    :param color: TetraColor to standardize
    :return: Standardized color as numpy array
    """
    if np.issubdtype(color.RGB.dtype, np.integer):
        rgb = color.RGB.astype(float) / 255.0
    else:
        rgb = color.RGB

    if np.issubdtype(color.OCV.dtype, np.integer):
        ocv = color.OCV.astype(float) / 255.0
    else:
        ocv = color.OCV

    return np.concatenate([rgb, ocv])


def _generate_geometry(dot_sizes: List[int], image_size: int, seed: int) -> List[List[float]]:
    """
    Generate the geometry for the Ishihara plate.

    :param dot_sizes: List of dot sizes to use
    :param image_size: Size of the output image
    :param seed: Random seed for reproducibility
    :return: List of circle definitions [x, y, r]
    """
    np.random.seed(seed)

    # Create packed_circles, a list of (x, y, r) tuples
    radii = dot_sizes * 2000
    np.random.shuffle(radii)
    packed_circles = packcircles.pack(radii)

    # Generate output_circles
    center = image_size // 2
    output_circles = []

    for (x, y, radius) in packed_circles:
        if np.sqrt((x - center) ** 2 + (y - center) ** 2) < center * 0.95:
            r = radius - np.random.randint(2, 5)
            output_circles.append([x, y, r])

    return output_circles


def _compute_inside_outside(
    circles: List[List[float]],
    secret_img: np.ndarray,
    image_size: int,
    num_samples: int,
    noise: float,
    gradient: bool
) -> Tuple[List[float], List[float]]:
    """
    Compute which circles are inside vs outside the secret shape.

    :param circles: List of circle definitions [x, y, r]
    :param secret_img: Secret image as numpy array
    :param image_size: Size of the image
    :param num_samples: Number of samples to take for gradient plates
    :param noise: Amount of noise to add
    :param gradient: Whether to use gradient sampling
    :return: Tuple of (inside_props, outside_props)
    """
    # Inside corresponds to numbers; outside corresponds to background
    outside = np.int32(np.sum(secret_img == 255, -1) == 4)
    inside = None

    if gradient:
        inside = np.int32((secret_img[:, :, 3] == 255)) - outside

    inside_props = []
    outside_props = []
    n = np.random.rand(len(circles))

    for i, [x, y, r] in enumerate(circles):
        x, y = int(round(x)), int(round(y))

        if gradient:
            assert inside is not None
            inside_count, outside_count = 0, 0

            for _ in range(num_samples):
                while True:
                    dx = np.random.uniform(-r, r)
                    dy = np.random.uniform(-r, r)
                    if (dx**2 + dy**2) <= r**2:
                        break

                x_grid = int(np.clip(np.round(x + dx), 0, image_size - 1))
                y_grid = int(np.clip(np.round(y + dy), 0, image_size - 1))
                if inside[y_grid, x_grid]:
                    inside_count += 1
                elif outside[y_grid, x_grid]:
                    outside_count += 1

            in_p = np.clip(inside_count / num_samples * (1 - (n[i] * noise / 100)), 0, 1)
            out_p = np.clip(outside_count / num_samples * (1 - (n[i] * noise / 100)), 0, 1)
        else:
            # Non-gradient sampling -- only sample center of circles
            x = int(np.clip(x, 0, image_size - 1))
            y = int(np.clip(y, 0, image_size - 1))
            is_outside = 1 if outside[y, x] else 0
            is_inside = 1 - is_outside
            in_p, out_p = is_inside, is_outside

        inside_props.append(in_p)
        outside_props.append(out_p)

    return inside_props, outside_props


def _draw_plate(
    circles: List[List[float]],
    inside_props: List[float],
    outside_props: List[float],
    inside_color: np.ndarray,
    outside_color: np.ndarray,
    channel_draws: List[ImageDraw.ImageDraw],
    lum_noise: float,
    noise_generator: Optional[Callable[[], npt.NDArray]] = None
) -> None:
    """
    Draw the plate with the computed circle positions and colors.

    :param circles: List of circle definitions [x, y, r]
    :param inside_props: List of inside proportions for each circle
    :param outside_props: List of outside proportions for each circle
    :param inside_color: Color for shape elements
    :param outside_color: Color for background elements
    :param channel_draws: ImageDraw objects for each channel
    :param lum_noise: Luminance noise amount
    :param noise_generator: Optional custom noise generator
    """
    for i, [x, y, r] in enumerate(circles):
        in_p, out_p = inside_props[i], outside_props[i]

        if noise_generator:
            new_color = np.clip(noise_generator(), 0, 1)
            if in_p:
                new_color = new_color[0]
            else:
                new_color = new_color[1]
        else:
            circle_color = in_p * inside_color + out_p * outside_color
            # Noise applied to the six channel, scale the entire vector
            lum_noise_val = np.random.normal(0, lum_noise)
            # Only apply to vector that are on
            new_color = np.clip(circle_color + (lum_noise_val * (circle_color > 0)), 0, 1)

        # Draw the ellipse
        bounding_box = [x-r, y-r, x+r, y+r]
        ellipse_color = (new_color * 255).astype(int)
        channel_draws[0].ellipse(bounding_box, fill=tuple(ellipse_color[:3]), width=0)
        channel_draws[1].ellipse(bounding_box, fill=tuple(ellipse_color[3:]), width=0)


def generate_ishihara_plate(
    plate_color: PlateColor,
    secret: int = _SECRETS[0],
    num_samples: int = 100,
    dot_sizes: List[int] = [16, 22, 28],
    image_size: int = 1024,
    seed: int = 0,
    lum_noise: float = 0,
    noise: float = 0,
    gradient: bool = False,
    noise_generator: Optional[Callable[[], npt.NDArray]] = None,
    corner_label: Optional[str] = None,
    corner_color: npt.ArrayLike = np.array([255/2, 255/2, 255/2, 255/2, 0, 0]).astype(int),
    background_color: TetraColor = TetraColor(RGB=np.array([0, 0, 0]), OCV=np.array([0, 0, 0]))
) -> Tuple[Image.Image, Image.Image]:
    """
    Generate an Ishihara Plate with specified properties.

    Parameters:
    -----------
    plate_color : PlateColor
        A PlateColor object with shape and background colors (RGB/OCV tuples).
    secret : int
        Specifies which secret file to use from the secrets directory.
    num_samples : int
        Number of samples to take for gradient plates.
    dot_sizes : List[int]
        List of dot sizes to use in the plate.
    image_size : int
        Size of the output image.
    seed : int
        RNG seed for plate generation.
    lum_noise : float
        Amount of luminance noise to add.
    noise : float
        Amount of noise to add to gradient plates.
    gradient : bool
        Whether to generate a gradient plate.
    noise_generator : Callable[[], npt.NDArray]
        Custom noise generator function.
    corner_label : str
        Optional label text to draw in the corner of the plate.
    corner_color : npt.ArrayLike
        Color for the corner label.

    Returns:
    --------
    Tuple[Image.Image, Image.Image]
        A tuple of (RGB_image, OCV_image).
    """
    # Validate inputs
    if secret not in _SECRETS:
        raise ValueError(f"Invalid Hidden Number {secret}")

    if not gradient:
        num_samples = 1
        if noise != 0:
            raise ValueError("None-zero noise is not supported for non-gradient plates -- it doesn't make sense!")

    # Standardize colors
    inside_color = _standardize_color(plate_color.shape)
    outside_color = _standardize_color(plate_color.background)

    # Load secret image
    with resources.path("TetriumColor.Assets.HiddenImages", f"{str(secret)}.png") as data_path:
        secret_img = Image.open(data_path)
    secret_img = secret_img.resize([image_size, image_size])
    secret_img = np.asarray(secret_img)

    # Generate geometry
    circles = _generate_geometry(dot_sizes, image_size, seed)

    # Calculate inside/outside proportions
    inside_props, outside_props = _compute_inside_outside(
        circles, secret_img, image_size, num_samples, noise, gradient
    )

    backgrounds = [tuple((background_color.RGB * 255).astype(np.uint8)),
                   tuple((background_color.OCV * 255).astype(np.uint8))]
    # Create images
    channels: List[Image.Image] = [Image.new(mode='RGB', size=(
        image_size, image_size), color=backgrounds[i]) for i in range(2)]
    channel_draws = [ImageDraw.Draw(ch) for ch in channels]

    # Draw plate
    _draw_plate(
        circles, inside_props, outside_props, inside_color, outside_color,
        channel_draws, lum_noise, noise_generator
    )

    # Draw corner label if provided
    if corner_label:
        font = ImageFont.load_default(size=150)
        corner_color_array = np.array(corner_color)
        if np.issubdtype(corner_color_array.dtype, np.floating):
            corner_color_array = (corner_color_array * 255).astype(int)

        channel_draws[0].text((10, 10), corner_label, fill=tuple(corner_color_array[:3]), font=font)
        channel_draws[1].text((10, 10), corner_label, fill=tuple(corner_color_array[3:]), font=font)

    if len(channels) != 2:
        raise ValueError("Expected exactly two channels, but got {len(channels)}")
    return channels[0], channels[1]


def export_plate(rgb_img: Image.Image, ocv_img: Image.Image, filename_rgb: str, filename_ocv: str):
    """
    Export the generated plate images to files.

    :param rgb_img: RGB channel image
    :param ocv_img: OCV channel image
    :param filename_rgb: Filename for the RGB image
    :param filename_ocv: Filename for the OCV image
    """
    rgb_img.save(filename_rgb)
    ocv_img.save(filename_ocv)


def GenerateHiddenImages(output_dir: str):
    """
    Generate a series of images from 1-99 that resemble the style of Assets/HiddenImages/27.png.
    Each image will have a transparent background, a white circle, and a black number centered.

    :param output_dir: Directory to save the generated images.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for number in range(10, 100):
        # Create a transparent image
        image_size = 1024
        img = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw a white circle
        circle_radius = image_size // 2
        circle_bbox = [
            (image_size // 2 - circle_radius, image_size // 2 - circle_radius),
            (image_size // 2 + circle_radius, image_size // 2 + circle_radius),
        ]
        draw.ellipse(circle_bbox, fill=(255, 255, 255, 255))

        # Draw the black number centered
        font_size = 700
        resource = resources.files('TetriumColor.Assets.Fonts') / 'Rubik-Medium.ttf'
        with as_file(resource) as font_path:
            font = ImageFont.truetype(str(font_path), size=font_size)
        text = str(number)
        draw.text((image_size/2, image_size/2), text, font=font, anchor="mm", fill=(0, 0, 0, 255))

        # Save the image
        img.save(output_path / f"{number}.png")


if __name__ == "__main__":
    GenerateHiddenImages("TetriumColor/Assets/HiddenImages")
