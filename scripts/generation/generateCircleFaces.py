import os
import numpy as np
import argparse

from typing import List

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.Observer.Observer import gaussian
from TetriumColor.PsychoPhys.HueSphere import GenerateEachFaceOfGrid, GenerateCubeMapTextures, ConcatenateCubeMap
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver, Spectra
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.Utils.ParserOptions import AddObserverArgs

# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
parser.add_argument('--scrambleProb', type=float, default=0, help='Probability of scrambling the color')
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                             args.l_cone_peak, args.macula, args.lens, args.template)

primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

for metameric_axis in range(2, 3):
    color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
        observer, primaries, metameric_axis=metameric_axis, scaling_factor=10000)

    GenerateEachFaceOfGrid(1.3, 0.3, color_space_transform, 5, f'./outputs/cubemap_faces_5x5')
