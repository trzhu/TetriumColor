from TetriumColor.Utils.CustomTypes import ColorTestResult

from TetriumColor.TetraColorPicker import ScreeningTestColorGenerator
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlate


class PseudoIsochromaticPlateGenerator:

    def __init__(self, transform_dirs:str, pregenerated_filenames:str, num_tests:int, seed:int=42):
        """
        Initializes the PseudoIsochromaticPlateGenerator with the given number of tests and seed
        
        Args:
            numTests (int): The number of tests to generate in total (setting the color generator arguments)
            seed (int): The seed for the plate pattern generation.
        """
        self.seed = seed
        self.color_generator = ScreeningTestColorGenerator(num_tests, transform_dirs, pregenerated_filenames)
        self.current_plate : IshiharaPlate = None


    def NewPlate(self, filename_RGB: str, filename_OCV: str, hidden_number:int): # must be called before GetPlate
        """
        Generates a new plate with the given hidden number, and colored by the ScreenTestColorGenerator

        Args:
            hiddenNumber (int): The hidden number to save to the plate
        """
        newColor = self.color_generator.NewColor()
        self.current_plate = IshiharaPlate(newColor, hidden_number)
        self.current_plate.GeneratePlate(self.seed, hidden_number, newColor)
        self.current_plate.ExportPlate(filename_RGB, filename_OCV)
        

    def GetPlate(self, previousResult: ColorTestResult, filenameRGB: str, filenameOCV: str, hiddenNumber:int):
        """
        Generates a new plate and saves it to a file with the given hidden number, and colored by the ScreenTestColorGenerator

        Args:
            previousResult (ColorTestResult): The result of the previous test (did they get it right or not)
            filenameRGB (str): The filename to save the plate in RGB LEDs
            filenameOCV (str): The filename to save the plate in OCV LEDs
            hiddenNumber (int): The hidden number to save to the plate
        """
        
        if self.current_plate is None:
            self.NewPlate(filenameRGB, filenameOCV, hiddenNumber)
            return
        
        color = self.color_generator.GetColor(previousResult)
        self.current_plate.GeneratePlate(self.seed, hiddenNumber, color)
        self.current_plate.ExportPlate(filenameRGB, filenameOCV) # should block until it is done writing