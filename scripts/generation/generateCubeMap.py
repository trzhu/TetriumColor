import os
import numpy as np

from typing import List

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransforms
from TetriumColor.PsychoPhys.HueSphere import GenerateCubeMapTextures, ConcatenateCubeMap
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomTetraObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 4)
observer = GetCustomTetraObserver(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=560, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/12-3/12-3-primaries-tetrium")[:4]

color_space_transform: ColorSpaceTransform = GetColorSpaceTransforms(
    [observer], [primaries], scaling_factor=10000)[0][0]

GenerateCubeMapTextures(0.7, 0.3, color_space_transform, 128, './outputs/RGB_cube_map',
                        './outputs/OCV_cube_map')
ConcatenateCubeMap('./outputs/RGB_cube_map', './outputs/cubemap_RGB.png')
ConcatenateCubeMap('./outputs/OCV_cube_map', './outputs/cubemap_OCV.png')
