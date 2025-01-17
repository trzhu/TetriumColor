import pdb
import os
import numpy as np

import matplotlib.pyplot as plt

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid, SampleDualCircle
from TetriumColor.PsychoPhys.HueSphere import CreateColorCircleSidebySide
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, od=0.5, dimension=4, m_cone_peak=533,
                             l_cone_peak=559, template="neitz", macular=0.5, lens=1)
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
    observer, primaries, scaling_factor=10000)

grid_size = 5
CreateColorCircleSidebySide((912, 1140), "./circle/tmp", 0.7, 0.3, 0.15, 100, color_space_transform)
