import os
import numpy as np

from typing import List

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransforms
from TetriumColor.Observer.Observer import gaussian
from TetriumColor.PsychoPhys.HueSphere import GenerateCubeMapTextures, ConcatenateCubeMap
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomTetraObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomTetraObserver(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransforms(
    [observer], [primaries], scaling_factor=10000)[0][0]
GenerateCubeMapTextures(0.7, 0.3, color_space_transform, 128, './outputs/RGB_cube_map',
                        './outputs/OCV_cube_map')
ConcatenateCubeMap('./outputs/RGB_cube_map', './outputs/cubemap_RGB.png')
ConcatenateCubeMap('./outputs/OCV_cube_map', './outputs/cubemap_OCV.png')
