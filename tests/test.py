from typing import List
import os
import time
from TetriumColor.TetraPlate import PseudoIsochromaticPlateGenerator
from TetriumColor.TetraColorPicker import ColorGenerator, ScreeningTestColorGenerator, TargetedTestColorGenerator, InDepthTestColorGenerator
from TetriumColor.Utils.CustomTypes import ColorTestResult

# Create a TetraPlate object

# Pregenerate Neitz Common Genes w/10 Metamers Each
num_tests: int = 10
transforms_base_path: str = './TetriumColor/Assets/ColorSpaceTransforms'
pregen_base_path: str = './TetriumColor/Assets/PreGeneratedMetamers'
display_primaries: str = 'RGBO'
peaks: List[tuple] = [(530, 559), (530, 555), (533, 559), (533, 555)]
transformDirs: List[str] = [os.path.join(
    transforms_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}') for m_peak, l_peak in peaks]
saveDirs: List[str] = [os.path.join(
    pregen_base_path, f'Neitz_{m_peak}_{l_peak}-{display_primaries}.pkl') for m_peak, l_peak in peaks]

# Create Color Generators to Test
screening_test_color_generator: ScreeningTestColorGenerator = ScreeningTestColorGenerator(
    num_tests, transformDirs, saveDirs)

# targeted_test_color_generator: TargetedTestColorGenerator = TargetedTestColorGenerator(
#     num_tests, transformDirs[0], 0.7, 5)

# indepth_test_color_generator: InDepthTestColorGenerator = InDepthTestColorGenerator(
#     transformDirs[0], 0.7, 0.3, 25, 5)

all_color_generators: List[ColorGenerator] = [
    screening_test_color_generator]
# targeted_test_color_generator]

all_color_generator_names: List[str] = ["Screening", "Targeted"]

# Test PseudoIsochromaticPlateGenerator with each of the color generators
for color_generator_name, color_generator in zip(all_color_generator_names, all_color_generators):
    start_time = time.time()
    tetraPlate: PseudoIsochromaticPlateGenerator = PseudoIsochromaticPlateGenerator(
        color_generator, num_tests)
    elapsed_time = time.time() - start_time
    print(f"Initialization -- Elapsed Time: {elapsed_time:.3f} seconds")

    os.makedirs('./tmp', exist_ok=True)
    for i in range(num_tests):
        start_time = time.time()
        tetraPlate.GetPlate(ColorTestResult(1), f'./tmp/{color_generator_name}_RGB_{i}.png',
                            f'./tmp/{color_generator_name}_OCV_{i}.png', 27)
        elapsed_time = time.time() - start_time
        print(f"Plate {i} -- Elapsed Time: {elapsed_time:.3f} seconds")
