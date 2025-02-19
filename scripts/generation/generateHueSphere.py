import pdb
import matplotlib.pyplot as plt
import argparse
import numpy as np
import matplotlib.pyplot as plt

import TetriumColor.ColorMath.GamutMath as GamutMath
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetCustomObserver
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransformWODisplay
from TetriumColor.Utils.ParserOptions import AddObserverArgs

parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = GetCustomObserver(wavelengths, od=0.5, m_cone_peak=args.m_cone_peak, q_cone_peak=args.q_cone_peak,
                             l_cone_peak=args.l_cone_peak, template=args.template, macular=args.macula, lens=args.lens)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransformWODisplay(observer)

boundary_pts_in_hering = GamutMath.GenerateMaximalHueSpherePoints(10000, color_space_transform, observer)

hering_to_cone = np.linalg.inv(color_space_transform.cone_to_disp)@color_space_transform.hering_to_disp
cones = boundary_pts_in_hering@hering_to_cone.T
sRGBs = np.clip(cones@color_space_transform.cone_to_sRGB.T, 0, 1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, z coordinates
x = boundary_pts_in_hering[:, 1]
y = boundary_pts_in_hering[:, 2]
z = boundary_pts_in_hering[:, 3]

# Plot each point with its corresponding sRGB color
ax.scatter(x, y, z, color=sRGBs)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

pdb.set_trace()
