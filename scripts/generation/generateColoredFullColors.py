import pdb
import numpy as np
import matplotlib.pyplot as plt

from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransformWODisplay
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import Observer, GetCustomObserver, Spectra, GetMaxBasisToDisplayTransform, GetRGBOCVToConeBasis
from TetriumColor.PsychoPhys.HueSphere import SampleSaturatedSubManifold

wavelengths = np.arange(360, 831, 1)
observer = Observer.tetrachromat(wavelengths)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransformWODisplay(observer, metameric_axis=2)
lum_sat_pairs = [(0.5, 0.2), (0.8, 0.3), (1.2, 0.3), (1.5, 0.2)]
cones = SampleSaturatedSubManifold(lum_sat_pairs, color_space_transform)
cone_to_hering = np.linalg.inv(color_space_transform.hering_to_disp)@color_space_transform.cone_to_disp
hering = cones@np.linalg.inv(color_space_transform.hering_to_cone).T
sRGBs = np.clip(cones@color_space_transform.cone_to_sRGB.T, 0, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hering[:, 1], hering[:, 2], hering[:, 3], c=sRGBs, marker='o')
plt.show()
