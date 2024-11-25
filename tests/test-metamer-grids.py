import os

from typing import List

from TetriumColor.ColorMath.HueToDisplay import GetMaximalMetamerPointsOnGrid
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Utils.IO import LoadColorSpaceTransform
from TetriumColor.PsychoPhys.HueSphere import CreateCircleGrid


def getTransformDirs(display_primaries: str):
    transforms_base_path: str = './TetriumColor/Assets/ColorSpaceTransforms'
    peaks: List[tuple] = [(530, 559), (530, 555), (533, 559), (533, 555)]
    transformDirs: List[str] = [os.path.join(
        transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
    return transformDirs


dirs = getTransformDirs('RGBO')
color_space_transform: ColorSpaceTransform = LoadColorSpaceTransform(dirs[0])

disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 9, color_space_transform)
CreateCircleGrid(disp_points, padding=10, radius=20, output_base="./tmp/grid")
