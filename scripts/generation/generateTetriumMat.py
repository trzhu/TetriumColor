
from typing import List
import numpy as np

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransforms
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomTetraObserver, Spectra, GetMaxBasisToDisplayTransform
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

wavelengths = np.arange(380, 781, 1)
observer = GetCustomTetraObserver(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransforms(
    [observer], [primaries], scaling_factor=10000)[0][0]

rygb_to_rgb, rygb_to_ocv = GetMaxBasisToDisplayTransform(color_space_transform)
print(rygb_to_rgb)
print(rygb_to_ocv)
