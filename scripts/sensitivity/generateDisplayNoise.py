"""Fix a single observer, and peturb the measurement of the primaries to be as close to the real leds as possible
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomTetraObserver, GetColorSpaceTransforms, Spectra, Observer
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticGrid


# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer: Observer = GetCustomTetraObserver(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=560, template="neitz")
primaries: List[Spectra] = LoadPrimaries("../../measurements/12-3/12-3-primaries-tetrium")[:4]

# Perturb Primaries to Find Better Models of the Display Primaries -- we get a list of color space transforms as a result, with one observer
perturbation = 0.01
gaussian_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
color_space_transforms: List[ColorSpaceTransform] = GetColorSpaceTransforms(
    [observer], [primaries, gaussian_primaries], scaling_factor=1000)[0]


fig = plt.figure()
ax = fig.add_subplot(111)
# Display the original primaries
for primary in primaries:
    primary.plot(ax=ax, color=primary.to_rgb())

for primary in gaussian_primaries:
    primary.plot(ax=ax, color=primary.to_rgb())
plt.show()

# Display the center points of the metamer grids for each of the observers -- see how close we are
center_points = []
for color_space_transform in color_space_transforms:
    disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, color_space_transform)
    center_points += [disp_points[2][2]]

closest_square = int(np.ceil(np.sqrt(len(center_points))))
display_grid_points = np.array(center_points).reshape(closest_square, 1, 2, 6)
CreatePseudoIsochromaticGrid(
    display_grid_points, f"./outputs/",  f"smoothed_grid")

# Pretty print the numpy array of center points
np.set_printoptions(precision=3, suppress=True)
print(center_points[0])
print(center_points[1])
