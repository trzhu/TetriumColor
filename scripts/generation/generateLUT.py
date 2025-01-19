import pdb
import argparse
import numpy as np
import os

import numpy.typing as npt
from typing import List

from PIL import Image, ImageDraw
import TetriumColor.ColorMath.GamutMath as GamutMath
from TetriumColor.PsychoPhys import HueSphere
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor
from TetriumColor.Observer import GetCustomObserver, Spectra, GetMaxBasisToDisplayTransform
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.Utils.ParserOptions import AddObserverArgs
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import VSXYZToRYGB
from TetriumColor.ColorMath.Geometry import ConvertCubeUVToXYZ

parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, od=0.5, m_cone_peak=args.m_cone_peak, q_cone_peak=args.q_cone_peak,
                             l_cone_peak=args.l_cone_peak, template=args.template, macular=args.macula, lens=args.lens)
# primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2025-01-16/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
    observer, primaries, metameric_axis=2, scaling_factor=10000)

image_size = 256

dirname = "LUT_cube_map_outputs"
os.makedirs(f"./{dirname}", exist_ok=True)
outputs = HueSphere.GenerateLUTCubeMap(
    color_space_transform, image_size, f"./{dirname}/cube")
HueSphere.ConcatenateCubeMap(f'./{dirname}/cube', f'./{dirname}/cubemap.png')

print(outputs)
# all_us = (np.arange(image_size) + 0.5) / image_size
# all_vs = (np.arange(image_size) + 0.5) / image_size
# cube_u, cube_v = np.meshgrid(all_us, all_vs)
# flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

# # change the associated xyzs -> to a new direction, but the same color values
# metamericDirMat = GamutMath.GetTransformChromToMetamericDir(color_space_transform)
# invMetamericDirMat = np.linalg.inv(metamericDirMat)

# luminance = 0.7
# saturation = 0.3
# cube_idx = 4

# xyz = ConvertCubeUVToXYZ(cube_idx, cube_u, cube_v, saturation).reshape(-1, 3)
# xyz = np.dot(invMetamericDirMat, xyz.T).T

# lum_vector = np.ones(image_size * image_size) * luminance
# vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))
# vshh = GamutMath.ConvertHeringToVSH(vxyz)

# pdb.set_trace()
# all_sats = []
# all_lums = []
# for i in range(len(vshh)):
#     angle = tuple(vshh[i, 2:])
#     lum_cusp, sat_cusp = map_angle_sat[angle]
#     all_lums.append(lum_cusp)
#     all_sats.append(sat_cusp)

# img_sat = Image.new('RGB', (image_size, image_size))  # sample color per pixel to avoid empty spots

# all_sats = np.array(all_sats)
# normalized_sats = (all_sats - np.min(all_sats)) / (np.max(all_sats) - np.min(all_sats))

# draw_sat = ImageDraw.Draw(img_sat)
# for j in range(len(flattened_u)):
#     u, v = flattened_v[j], flattened_u[j]
#     draw_sat.point((u * image_size, v * image_size), fill=int(normalized_sats[j] * 255, normalized_))

# img_sat.save("saturation_map.png")
