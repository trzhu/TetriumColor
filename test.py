import os
from TetriumColor.TetraPlate import PseudoIsochromaticPlateGenerator

# Create a TetraPlate object
num_tests = 3
tetraPlate = PseudoIsochromaticPlateGenerator('./TetriumColor/Assets/ColorSpaceTransforms/StandardTetrachromat-RGBO', num_tests)
os.makedirs('./tmp', exist_ok=True)
for i in range(num_tests):
    tetraPlate.GetPlate(1, f'./tmp/test_RGB_{i}.png', f'./tmp/test_OCV_{i}.png', 27)
