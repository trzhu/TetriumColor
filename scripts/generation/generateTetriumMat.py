
import pdb
from typing import List
import numpy as np

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransforms, GetColorSpaceTransformTosRGB
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver, Spectra, GetMaxBasisToDisplayTransform, GetRGBOCVToConeBasis
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries

wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransforms(
    [observer], [primaries], scaling_factor=10000)[0][0]

rygb_to_rgb, rygb_to_ocv = GetMaxBasisToDisplayTransform(color_space_transform)


# rygb_to_rgb = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
# rygb_to_ocv = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])


def print_glm_format(matrix: np.ndarray):
    print("glm::mat4x4{")
    for row in matrix:
        print(f"{{{row[0]}, {row[1]}, {row[2]}, {0.0}}},")
    print("},")


print_glm_format(rygb_to_rgb)

print_glm_format(rygb_to_ocv)

np.set_printoptions(precision=8)

print(rygb_to_rgb)
print(rygb_to_ocv)

np.save("rygb_to_rgb.npy", rygb_to_rgb)
np.save("rygb_to_ocv.npy", rygb_to_ocv)


trichromat = GetCustomObserver(wavelengths=np.arange(360, 831, 1), od=0.5, dimension=3)
cst_for_trichromat = GetColorSpaceTransformTosRGB(trichromat)

rgbo_to_sml = GetRGBOCVToConeBasis(color_space_transform)[[0, 1, 3]]

rygb_to_sml = (np.linalg.inv(color_space_transform.cone_to_disp) @
               np.flip(color_space_transform.maxbasis_to_disp, axis=1))[[0, 1, 3]]

rgbo_to_sRGB = cst_for_trichromat.cone_to_disp@rgbo_to_sml
rygb_to_sRGB = cst_for_trichromat.cone_to_disp@rygb_to_sml

print(repr(rgbo_to_sRGB))
print(repr(rygb_to_sRGB))

print_glm_format(rygb_to_sRGB.T)

pdb.set_trace()
