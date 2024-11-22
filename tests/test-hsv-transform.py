from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np

from TetriumColor.ColorMath.HueToDisplay import GenerateGamutLUT, RemapGamutPoints, SampleHueManifold, ConvertVSHToHering
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Utils.IO import LoadColorSpaceTransform
from TetriumColor.Visualization.GamutViz import Display2DPlane, DisplayParallepiped


def getTransformDirs(display_primaries: str):
    transforms_base_path: str = './TetriumColor/Assets/ColorSpaceTransforms'
    peaks: List[tuple] = [(530, 559), (530, 555), (533, 559), (533, 555)]
    transformDirs: List[str] = [os.path.join(
        transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
    return transformDirs


def test3DCase():
    dirs = getTransformDirs('RGB')
    color_space_transform: ColorSpaceTransform = LoadColorSpaceTransform(dirs[0])
    vshh = SampleHueManifold(1.7/2, 0.6, 3, 250)  # 0.6 is the max with all contained
    map_angle_sat = GenerateGamutLUT(vshh, color_space_transform)

    remapped_points = RemapGamutPoints(vshh, color_space_transform, map_angle_sat)

    all_hering_points = ConvertVSHToHering(remapped_points)
    cartesian_remapped_points = (color_space_transform.hering_to_disp @ all_hering_points.T).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.scatter(all_hering_points[:, 0], all_hering_points[:, 1], all_hering_points[:, 2], color='b', s=100)
    ax.set_xlim([-0.1, 1.7])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    DisplayParallepiped(ax, np.eye(3))
    # Display2DPlane(ax, np.array([1, 1, 1]), all_cartesian_pts[idx])
    # points = np.array(list(map_angle_sat.values()))
    ax.scatter(cartesian_remapped_points[:, 0], cartesian_remapped_points[:, 1],
               cartesian_remapped_points[:, 2], color='b', s=100)
    plt.show()


def test4DCase():
    dirs = getTransformDirs('RGBO')
    color_space_transform: ColorSpaceTransform = LoadColorSpaceTransform(dirs[0])
    vshh = SampleHueManifold(1.7/2, 0.4, 4, 1000)
    map_angle_sat = GenerateGamutLUT(vshh, color_space_transform)
    remapped_points = RemapGamutPoints(vshh, color_space_transform, map_angle_sat)
    print(np.min(remapped_points[:, 1]), np.max(remapped_points[:, 1]))
    all_hering_points = ConvertVSHToHering(remapped_points)
    cartesian_remapped_points = (color_space_transform.hering_to_disp @ all_hering_points.T).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.scatter(all_hering_points[:, 1], all_hering_points[:, 2], all_hering_points[:, 3], color='b', s=100)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    # DisplayParallepiped(ax, np.eye(3))
    # # Display2DPlane(ax, np.array([1, 1, 1]), all_cartesian_pts[idx])
    # # points = np.array(list(map_angle_sat.values()))
    # ax.scatter(cartesian_remapped_points[:, 0], cartesian_remapped_points[:, 1],
    #            cartesian_remapped_points[:, 2], color='b', s=100)
    # plt.show()


if __name__ == "__main__":
    # test3DCase()
    test4DCase()
