import os

from TetriumColor.PsychoPhys.HueSphere import GetHueSphereGeometryWithLineTexture, GetHueSphereGeometryWithCubeMapTexture
from TetriumColor.Utils.IO import LoadColorSpaceTransform


def getTransformDirs(display_primaries: str):
    transforms_base_path: str = './TetriumColor/Assets/ColorSpaceTransforms'
    peaks = [(530, 559), (530, 555), (533, 559), (533, 555)]
    transformDirs = [os.path.join(
        transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
    return transformDirs


dirs = getTransformDirs('RGBO')
color_space_transform = LoadColorSpaceTransform(dirs[0])
GetHueSphereGeometryWithLineTexture(1200, 0.6, 0.25, color_space_transform,
                                    './tmp/geometry/textures/RGB.png',
                                    './tmp/geometry/textures/OCV.png',
                                    './tmp/geometry/fibonacci_sampled.obj')

GetHueSphereGeometryWithCubeMapTexture(0.6, 0.3, color_space_transform, 64,
                                       './tmp/geometry/textures/cube_map_RGB.png',
                                       './tmp/geometry/textures/cube_map_OCV.png',
                                       './tmp/geometry/cubemap_sampled.obj')
