
from typing import List
import numpy as np

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransforms
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver, Spectra, GetMaxBasisToDisplayTransform
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransforms(
    [observer], [primaries], scaling_factor=10000)[0][0]


def print_glm_format(matrix: np.ndarray):
    print("glm::mat4x4{")
    for row in matrix:
        # print(f"{{{row[0]}, {row[1]}, {row[2]}, {0.0}}},")
        print(f"{{{row[0]}, {row[1]}, {row[2]}, {row[3]}}},")
    print("},")


print("color_space_transform.cone_to_disp")
print_glm_format(color_space_transform.cone_to_disp.T)
print("color_space_transform.maxbasis_to_disp")
print_glm_format(color_space_transform.maxbasis_to_disp.T)
print("color_space_transform.hering_to_disp")
print_glm_format(color_space_transform.hering_to_disp.T)
