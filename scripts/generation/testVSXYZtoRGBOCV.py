import argparse
import numpy as np

from typing import List

from PIL import Image, ImageDraw
import TetriumColor.ColorMath.GamutMath as GamutMath
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor
from TetriumColor.Observer import GetCustomObserver, Spectra
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.Utils.ParserOptions import AddObserverArgs
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import VSXYZToRGBOCV
from TetriumColor.ColorMath.Geometry import ConvertCubeUVToXYZ

parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, od=0.5, m_cone_peak=args.m_cone_peak, q_cone_peak=args.q_cone_peak,
                             l_cone_peak=args.l_cone_peak, template=args.template, macular=args.macula, lens=args.lens)
# primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
primaries: List[Spectra] = LoadPrimaries("./measurements/2025-01-16/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
    observer, primaries, metameric_axis=2, scaling_factor=10000)

image_size = 64
all_us = (np.arange(image_size) + 0.5) / image_size
all_vs = (np.arange(image_size) + 0.5) / image_size
cube_u, cube_v = np.meshgrid(all_us, all_vs)
flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

# change the associated xyzs -> to a new direction, but the same color values
metamericDirMat = GamutMath.GetTransformChromToMetamericDir(color_space_transform)
invMetamericDirMat = np.linalg.inv(metamericDirMat)

luminance = 0.7
saturation = 0.3

xyz = ConvertCubeUVToXYZ(4, cube_u, cube_v, saturation).reshape(-1, 3)
xyz = np.dot(invMetamericDirMat, xyz.T).T

corresponding_tetracolors: List[TetraColor] = []
for i in range(len(xyz)):
    corresponding_tetracolors.append(VSXYZToRGBOCV(luminance, saturation, xyz[i], color_space_transform))


img_rgb = Image.new('RGB', (image_size, image_size))  # sample color per pixel to avoid empty spots
img_ocv = Image.new('RGB', (image_size, image_size))

draw_rgb = ImageDraw.Draw(img_rgb)
draw_ocv = ImageDraw.Draw(img_ocv)

for j in range(len(flattened_u)):
    u, v = flattened_v[j], flattened_u[j]  # swap axis for PIL
    color = corresponding_tetracolors[j]
    rgb_color = (int(color.RGB[0] * 255), int(color.RGB[1] * 255), int(color.RGB[2] * 255))
    draw_rgb.point((u * image_size, v * image_size), fill=rgb_color)
    ocv_color = (int(color.OCV[0] * 255), int(color.OCV[1] * 255), int(color.OCV[2] * 255))
    draw_ocv.point((u * image_size, v * image_size), fill=ocv_color)

# Save the images
img_rgb.save(f'RGB.png')
img_ocv.save(f'OCV.png')
