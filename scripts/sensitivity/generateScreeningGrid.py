"""
Generate a screening grid for the general experiment
"""
import pdb
import numpy as np
import argparse

from PIL import Image
from typing import List
import numpy.typing as npt

from TetriumColor.Observer.DisplayObserverSensitivity import GetCustomTetraObserver
from TetriumColor.Observer import *
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticImages, CreatePaddedGrid
from TetriumColor.TetraColorPicker import NoiseGenerator
from TetriumColor.TetraPlate import GetControlTest, GetConeIdentifyingTest, GetObserverIdentifyingTest

# load parameters from the script
parser = argparse.ArgumentParser(description='Generate Screening Grid')
parser.add_argument('--factor', type=float, required=False, default=1, help='Scaling factor for noise generation')
parser.add_argument('--lum_noise', type=float, required=False, default=0.035, help='Luminance noise level')
parser.add_argument('--seed', type=int, required=False, default=12, help='Random seed for reproducibility')
parser.add_argument('--grid_indices', type=int, nargs=2, required=True,
                    help='Indices that we will take from the grid to choose the color')
parser.add_argument('--grid_size', type=int, required=False, default=5, help='Grid size as two integers')
parser.add_argument('--luminance', type=float, required=False, default=0.7, help='Luminance level')
parser.add_argument('--saturation', type=float, required=False, default=0.3, help='Saturation level')
args = parser.parse_args()

factor = args.factor
lum_noise = args.lum_noise
seed = args.seed
grid_indices = args.grid_indices
grid_size = args.grid_size
luminance = args.luminance
saturation = args.saturation
cube_idx = 4

# load primaries
wavelengths = np.arange(380, 781, 1)
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)


outputs: List[tuple[npt.NDArray, NoiseGenerator | None]] = []
names: List[str] = []

# Control Test
avg_observer = GetCustomTetraObserver(
    wavelengths=wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559)
outputs.append(GetControlTest(
    avg_observer, gaussian_smooth_primaries, 2, luminance, saturation, grid_indices, grid_size, cube_idx))
names.append("control")

# Cone Identifying Test -- we want to identify peaks S cone, M cone, and variants of all the L cones
cone_identifying_observers = []
cone_names = [419, 530, 559]
q_peaks = [547, 552, 555]
for peak in q_peaks:
    cone_identifying_observers.append(GetCustomTetraObserver(
        wavelengths=wavelengths, od=0.5, m_cone_peak=530, q_cone_peak=peak, l_cone_peak=559))

for observer, peak in zip(cone_identifying_observers, q_peaks):
    # axis over Q cone
    outputs.append(GetConeIdentifyingTest(
        observer, gaussian_smooth_primaries, 2, luminance, saturation, grid_indices, factor, grid_size, cube_idx))
    names.append(f"cone_identifying_{peak}")

noise_divisions = [1, 2, 4]
for axis, peak, noise in zip([0, 1, 3], cone_names, noise_divisions):
    outputs.append(GetConeIdentifyingTest(
        cone_identifying_observers[0], gaussian_smooth_primaries, axis, luminance, saturation, grid_indices, noise, grid_size, cube_idx))
    names.append(f"cone_identifying_{peak}_{noise}")

# Observer Identifying Test
observer_identifying_observers, peaks = GetPeakPrevalentObservers()
for observer, peak in zip(observer_identifying_observers, peaks):
    # axis over Q cone
    outputs.append(GetObserverIdentifyingTest(
        observer, gaussian_smooth_primaries, 2, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
    names.append(f"observer_identifying_{peak[0]}_{peak[1]}")

# Generate the Output Grid
dirname = "./screening_outputs/"
sub_image_dir = f"sub_images_{lum_noise}"
root = f"./{dirname}/{sub_image_dir}"
output_base = "screening"
rgb_image_files = [os.path.join(root, f"{output_base}_{base_name}_RGB.png") for base_name in names]
ocv_image_files = [os.path.join(root, f"{output_base}_{base_name}_OCV.png") for base_name in names]

center_pts, noise_generators = zip(*[(output[0], output[1]) for output in outputs])
center_pts = np.array(center_pts)
CreatePseudoIsochromaticImages(center_pts, f"./{dirname}/", output_base, names,
                               noise_generator=noise_generators, sub_image_dir=sub_image_dir, seed=seed)

img_rgb = CreatePaddedGrid(rgb_image_files, grid_size=(3, 4))
img_rgb = img_rgb.resize((1365, 1024), Image.Resampling.BOX)
img_rgb.save(f"./{dirname}/grid_lum_{factor}_{lum_noise}_RGB.png")

img_ocv = CreatePaddedGrid(ocv_image_files, grid_size=(3, 4))
img_ocv = img_ocv.resize((1365, 1024), Image.Resampling.BOX)
img_ocv.save(f"./{dirname}/grid_lum_{factor}_{lum_noise}_OCV.png")
