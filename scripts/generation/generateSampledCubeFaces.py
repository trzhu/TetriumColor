import os
import numpy as np
import argparse

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.PsychoPhys.HueSphere import CreateCircleGrid, CreatePseudoIsochromaticGrid
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.Utils.ParserOptions import AddObserverArgs

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

for metameric_axis in range(4):
    color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
        observer, primaries, metameric_axis=metameric_axis, scaling_factor=10000)

    dirname = f"grid_outputs/q_cone_{args.q_cone_peak}_ma_{metameric_axis}"
    os.makedirs(dirname, exist_ok=True)

    grid_size = 5
    disp_points, _ = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, grid_size, color_space_transform)
    CreateCircleGrid(disp_points, padding=20, radius=100, output_base="./outputs/5x5")
    # CreatePseudoIsochromaticGrid(
    #     disp_points, f"./{dirname}/", f"{metameric_axis}_{grid_size}x{grid_size}_grid")
