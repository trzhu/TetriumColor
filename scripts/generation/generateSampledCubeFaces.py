import os
import numpy as np

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.PsychoPhys.HueSphere import CreateCircleGrid, CreatePseudoIsochromaticGrid
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransforms
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomTetraObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomTetraObserver(wavelengths, od=0.5, m_cone_peak=533,
                                  l_cone_peak=559, template="neitz", macular=0.5, lens=1)
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransforms(
    [observer], [primaries], scaling_factor=10000)[0][0]


grid_size = 5
disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, grid_size, color_space_transform)
CreateCircleGrid(disp_points, padding=10, radius=20, output_base="./outputs/5x5")
CreatePseudoIsochromaticGrid(
    disp_points, f"./outputs/", f"Neitz_{grid_size}x{grid_size}_grid")
