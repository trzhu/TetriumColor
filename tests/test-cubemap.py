import os
import numpy as np

from typing import List

from TetriumColor.PsychoPhys.HueSphere import GenerateCubeMapTextures, ConcatenateCubeMap
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Utils.IO import LoadColorSpaceTransform


def getTransformDirs(display_primaries: str):
    transforms_base_path: str = './TetriumColor/Assets/ColorSpaceTransforms'
    peaks: List[tuple] = [(530, 559), (530, 555), (533, 559), (533, 555)]
    transformDirs: List[str] = [os.path.join(
        transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
    return transformDirs


dirs = getTransformDirs('RGBO')
color_space_transform: ColorSpaceTransform = LoadColorSpaceTransform(dirs[0])

GenerateCubeMapTextures(0.7, 0.3, color_space_transform, 128, './tmp/test_RGB_cube_map',
                        './tmp/test_OCV_cube_map', './tmp/test_sRGB_cube_map')
ConcatenateCubeMap('./tmp/test_RGB_cube_map', './tmp/test_RGB_cube_map.png')
ConcatenateCubeMap('./tmp/test_OCV_cube_map', './tmp/test_OCV_cube_map.png')
# ConcatenateCubeMap('./tmp/test_sRGB_cube_map', './tmp/test_sRGB_cube_map.png')
