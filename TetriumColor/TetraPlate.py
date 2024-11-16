from TetriumColor.Utils.CustomTypes import ColorTestResult

from TetriumColor.TetraColorPicker import ScreeningTestColorGenerator
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlate


class PseudoIsochromaticPlateGenerator:

    def __init__(self, transformDir:str, numTests:int, seed:int=42):
        """
        Initializes the PseudoIsochromaticPlateGenerator with the given number of tests and seed
        
        Args:
            numTests (int): The number of tests to generate in total (setting the color generator arguments)
            seed (int): The seed for the plate pattern generation.
        """
        self.seed = seed
        self.colorGenerator = ScreeningTestColorGenerator(numTests, transformDir)
        self.currentPlate : IshiharaPlate = None


    def NewPlate(self, filenameRGB: str, filenameOCV: str, hiddenNumber:int): # must be called before GetPlate
        """
        Generates a new plate with the given hidden number, and colored by the ScreenTestColorGenerator

        Args:
            hiddenNumber (int): The hidden number to save to the plate
        """
        newColor = self.colorGenerator.NewColor()
        self.currentPlate = IshiharaPlate(newColor, hiddenNumber)
        self.currentPlate.generate_plate(self.seed, hiddenNumber, newColor)
        self.currentPlate.export_plate(filenameRGB, filenameOCV)
        

    def GetPlate(self, previousResult: ColorTestResult, filenameRGB: str, filenameOCV: str, hiddenNumber:int):
        """
        Generates a new plate and saves it to a file with the given hidden number, and colored by the ScreenTestColorGenerator

        Args:
            previousResult (ColorTestResult): The result of the previous test (did they get it right or not)
            filenameRGB (str): The filename to save the plate in RGB LEDs
            filenameOCV (str): The filename to save the plate in OCV LEDs
            hiddenNumber (int): The hidden number to save to the plate
        """
        
        if self.currentPlate is None:
            self.NewPlate(filenameRGB, filenameOCV, hiddenNumber)
            return
        
        color = self.colorGenerator.GetColor(previousResult)
        self.currentPlate.generate_plate(self.seed, hiddenNumber, color)
        self.currentPlate.export_plate(filenameRGB, filenameOCV) # should block until it is done writing