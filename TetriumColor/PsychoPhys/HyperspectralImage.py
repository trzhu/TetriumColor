import numpy as np
from PIL import Image

from spectral import open_image

from TetriumColor import ColorSpaceTransform
from TetriumColor.Observer import Observer
from TetriumColor.ColorMath.Conversion import Map4DTo6D


def GenerateHyperspectralImage(hyperspectral_filename: str, observer: Observer, color_space_transform: ColorSpaceTransform,
                               filename_RGB: str, filename_OCV: str):

    hyperspectral_wavelengths = np.arange(400, 701, 10)
    hyperspectral_image = open_image(hyperspectral_filename)

    data = np.array(hyperspectral_image.load().tolist())

    image_size_h = data.shape[0]
    image_size_w = data.shape[1]

    sensor_mat = observer.get_normalized_sensor_matrix(hyperspectral_wavelengths)

    # Convert hyperspectral images to display_space
    cone_space = sensor_mat@(data.reshape(-1, data.shape[2])).T

    # Convert to RGBOCV
    rgbocv = Map4DTo6D(cone_space.T@color_space_transform.cone_to_disp.T, color_space_transform)

    rgb = rgbocv[:, :3].reshape((image_size_h, image_size_w, 3))
    ocv = rgbocv[:, 3:].reshape((image_size_h, image_size_w, 3))

    Image.fromarray((rgb * 255).astype(np.uint8)).save(filename_RGB)
    Image.fromarray((ocv * 255).astype(np.uint8)).save(filename_OCV)
