"""Fix a single observer, and peturb the measurement of the primaries to be as close to the real leds as possible
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetAllObservers, GetColorSpaceTransforms, Spectra, Observer
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries, PerturbPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticGrid


# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observers: List[Observer] = GetAllObservers()
primaries: List[Spectra] = LoadPrimaries("../../measurements/12-3/12-3-primaries-tetrium")[:4]
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
perturbed_primaries: List[List[Spectra]] = PerturbPrimaries(primaries)

fig = plt.figure()
ax = fig.add_subplot(111)
for primary in primaries:
    primary.plot(ax=ax, color='k')
for primary in gaussian_smooth_primaries:
    primary.plot(ax=ax, color='r')

for perturbed_primary in perturbed_primaries[2:]:
    for primary in perturbed_primary:
        primary.plot(ax=ax, color='b')
plt.show()

# # Perturb Observer with Fixed Primaries to see if the issues come from the observer side
# color_space_transforms: List[List[ColorSpaceTransform]] = GetColorSpaceTransforms(
#     observers, [primaries, gaussian_smooth_primaries] + perturbed_primaries, scaling_factor=1000)

# with open('./outputs/all_observers_color_space_transforms.pkl', 'wb') as f:
#     pickle.dump(color_space_transforms, f)

# Load the color space transforms from the pickle file
with open('./outputs/all_observers_color_space_transforms.pkl', 'rb') as f:
    color_space_transforms = pickle.load(f)

primary_types = ['Measured', 'Gaussian', 'Perterbed_Up', 'Perterbed_Down']  # 'Perterbed_Left', 'Perterbed_Right'
# Display the center points of the metamer grids for each of the observers -- see how close we are

for i, primary_type in enumerate(primary_types):
    center_points = []
    for observer_type in range(len(observers)):
        color_space_transform = color_space_transforms[observer_type][i]
        disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, color_space_transform)
        center_points += [disp_points[2][2]]

    closest_square = int(np.ceil(np.sqrt(len(center_points))))
    display_grid_points = np.array(center_points).reshape(closest_square, closest_square, 2, 6)
    CreatePseudoIsochromaticGrid(
        display_grid_points, f"./outputs", f"all_observers_{primary_type}_grid")
