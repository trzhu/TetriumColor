import os

from TetriumColor.PsychoPhys.HueSphere import GetSphereGeometry, GetFibonacciSampledHueTexture
from TetriumColor.Utils.IO import LoadColorSpaceTransform


def getTransformDirs(display_primaries: str):
    transforms_base_path: str = './TetriumColor/Assets/ColorSpaceTransforms'
    peaks = [(530, 559), (530, 555), (533, 559), (533, 555)]
    transformDirs = [os.path.join(
        transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
    return transformDirs


dirs = getTransformDirs('RGBO')
color_space_transform = LoadColorSpaceTransform(dirs[0])
luminance: float = 0.7
saturation: float = 0.3
num_points: int = 15000

GetSphereGeometry(luminance, saturation, num_points, './tmp/geometry/sphere.obj')
GetFibonacciSampledHueTexture(num_points, luminance, saturation, color_space_transform,
                              './tmp/geometry/textures/RGB.png', './tmp/geometry/textures/OCV.png')
