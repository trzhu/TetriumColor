import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse

from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
from TetriumColor.Observer import Observer
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Utils.ParserOptions import AddObserverArgs
from TetriumColor.Utils.ImageUtils import CreatePaddedGrid
# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
parser.add_argument('--scrambleProb', type=float, default=0, help='Probability of scrambling the color')
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)

# plt.plot(observer.wavelengths, observer.sensor_matrix.T)
# plt.show()
primaries = load_primaries_from_csv("../../measurements/2025-05-11/primaries/")  # RGBO

print(primaries)

metameric_axis = 2
grid_size = 5
cs_4d = ColorSpace(observer, cst_display_type='led',
                   display_primaries=primaries, metameric_axis=metameric_axis,
                   luminance_per_channel=[np.sqrt(1/3)] * 4, chromas_per_channel=[np.sqrt(2/3)] * 4)

color_sampler = ColorSampler(cs_4d, cubemap_size=grid_size)

images = color_sampler.get_metameric_grid_plates(1, 0.5, 4)

foreground_images = [x for x, _ in images]
background_images = [y for _, y in images]

bg_color = cs_4d.get_background(1)
foreground = CreatePaddedGrid(foreground_images, padding=0, bg_color=tuple(
    (bg_color.RGB * 255).astype(np.uint8).tolist()))
background = CreatePaddedGrid(background_images, padding=0, bg_color=tuple(
    (bg_color.OCV * 255).astype(np.uint8).tolist()))

foreground.save("./assets/plates_RGB.png")
background.save("./assets/plates_OCV.png")
