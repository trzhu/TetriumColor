import os
import numpy as np
from typing import List

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.PsychoPhys.HyperspectralImage import GenerateHyperspectralImage
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

# Load Observer and Measured Primaries
wavelengths = np.arange(400, 701, 10)
observer = GetCustomObserver(wavelengths, dimension=4, od=0.5, m_cone_peak=530,
                             q_cone_peak=547, l_cone_peak=559, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(observer, primaries, scaling_factor=10000)

img_num = 418
hyperspectral_filename = f"/Users/jessicalee/Projects/generalized-colorimetry/code/TetriumColor/data/ARAD_1K_ENVI/ARAD_1K_{img_num:04d}.hdr"
hyperspectral_outputs = "./hyperspectral_outputs"
GenerateHyperspectralImage(hyperspectral_filename, observer, color_space_transform,
                           os.path.join(hyperspectral_outputs, f"ARAD_1K_{img_num:04d}"))
