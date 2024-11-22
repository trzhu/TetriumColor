from TetriumColor.Utils.CustomTypes import ColorTestResult

from TetriumColor.TetraColorPicker import ColorGenerator
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlate


class PseudoIsochromaticPlateGenerator:

    def __init__(self, color_generator: ColorGenerator, num_tests: int, seed: int = 42):
        """
        Initializes the PseudoIsochromaticPlateGenerator with the given number of tests and seed

        Args:
            numTests (int): The number of tests to generate in total (setting the color generator arguments)
            seed (int): The seed for the plate pattern generation.
        """
        self.seed: int = seed
        self.color_generator: ColorGenerator = color_generator
        self.current_plate: IshiharaPlate = IshiharaPlate(seed=self.seed)

    # must be called before GetPlate
    def NewPlate(self, filename_RGB: str, filename_OCV: str, hidden_number: int):
        """
        Generates a new plate with the given hidden number, and colored by the ScreenTestColorGenerator

        Args:
            hidden_number (int): The hidden number to save to the plate
        """
        plate_color = self.color_generator.NewColor()
        self.current_plate.GeneratePlate(self.seed, hidden_number, plate_color)
        self.current_plate.ExportPlate(filename_RGB, filename_OCV)

    def GetPlate(self, previous_result: ColorTestResult, filename_RGB: str, filename_OCV: str, hidden_number: int):
        """
        Generates a new plate and saves it to a file with the given hidden number, and colored by the ScreenTestColorGenerator

        Args:
            previousResult (ColorTestResult): The result of the previous test (did they get it right or not)
            filenameRGB (str): The filename to save the plate in RGB LEDs
            filenameOCV (str): The filename to save the plate in OCV LEDs
            hiddenNumber (int): The hidden number to save to the plate
        """
        plate_color = self.color_generator.GetColor(previous_result)
        self.current_plate.GeneratePlate(self.seed, hidden_number, plate_color)
        self.current_plate.ExportPlate(filename_RGB, filename_OCV)
