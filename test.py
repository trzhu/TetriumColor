from typing import List
import os
from TetriumColor.TetraPlate import PseudoIsochromaticPlateGenerator
from TetriumColor.TetraColorPicker import ScreeningTestColorGenerator

# Create a TetraPlate object

# Pregenerate Neitz Common Genes w/10 Metamers Each
num_tests:int= 10
transforms_base_path:str = './TetriumColor/Assets/ColorSpaceTransforms'
pregen_base_path:str= './TetriumColor/Assets/PreGeneratedMetamers'
display_primaries:str = 'RGBO'
peaks:List[tuple] = [(530, 559), (530, 555), (533, 559), (533, 555)]
transformDirs: List[str] = [os.path.join(transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
saveDirs: List[str] = [os.path.join(pregen_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}.pkl') for m_peak, l_peak in peaks]

tetraPlate: PseudoIsochromaticPlateGenerator = PseudoIsochromaticPlateGenerator(transformDirs, saveDirs, num_tests)

os.makedirs('./tmp', exist_ok=True)
for i in range(num_tests):
    tetraPlate.GetPlate(1, f'./tmp/test_RGB_{i}.png', f'./tmp/test_OCV_{i}.png', 27)