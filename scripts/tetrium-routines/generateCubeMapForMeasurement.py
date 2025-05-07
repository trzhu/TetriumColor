import numpy as np
import argparse

from typing import List

from numpy.lib import imag

from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
from TetriumColor.Observer import Observer, Spectra
from TetriumColor.Measurement import load_primaries_from_csv, compare_dataset_to_primaries, get_spectras_from_rgbo_list, plot_measured_vs_predicted
from TetriumColor.Utils.ParserOptions import AddObserverArgs
# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
parser.add_argument('--scrambleProb', type=float, default=0, help='Probability of scrambling the color')
args = parser.parse_args()


# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)
primaries: List[Spectra] = load_primaries_from_csv("../../measurements/2025-05-06/primaries")
metameric_axis = 2


cs_4d = ColorSpace(observer, cst_display_type='led',
                   display_primaries=primaries, metameric_axis=metameric_axis)
color_sampler = ColorSampler(cs_4d, cubemap_size=3)

images = color_sampler.generate_cubemap(1.0, 0.4, ColorSpaceType.SRGB)[4:]
for i, image in enumerate(images):
    image.save(f"cubemap_{i}.png")

colors = color_sampler.output_cubemap_values(1.0, 0.4, ColorSpaceType.DISP)[4:]
colors = [item for sublist in colors for item in sublist]
colors_8bit = (np.array(colors) * 255).astype(np.uint8)
print("{" + ",\n ".join("{" + ", ".join(map(str, inner_list)) + "}" for inner_list in colors_8bit) + "}")

measurements_dir = "../../measurements/2025-05-06/cubemap_patches"
measured_spectras = get_spectras_from_rgbo_list(measurements_dir, colors_8bit.tolist())
results = compare_dataset_to_primaries(measurements_dir, colors_8bit.tolist(), primaries)
