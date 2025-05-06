import numpy.typing as npt
from typing import List

from TetriumColor.Utils.CustomTypes import ColorTestResult
from TetriumColor.Observer import *
from TetriumColor.TetraColorPicker import ColorGenerator, ConeLuminanceNoiseGenerator
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.TetraColorPicker import BackgroundNoiseGenerator, LuminanceNoiseGenerator, NoiseGenerator
from TetriumColor.Observer.DisplayObserverSensitivity import GetAllObservers, GetColorSpaceTransformsOverObservers
from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid, GetMaxMetamerOverGridSample
import pickle


# Control Test
def GetControlTest(observer: Observer, primaries: List[Spectra], metameric_axis: int,
                   luminance: float, saturation: float, lum_noise: float, grid_indices: tuple,
                   grid_size: int = 5, cube_idx: int = 4) -> tuple[npt.NDArray, NoiseGenerator | None]:

    avg_obs_cst = GetColorSpaceTransform(observer, primaries, metameric_axis=metameric_axis)
    disp_points, cone_diff = GetMaximalMetamerPointsOnGrid(luminance, saturation, cube_idx, grid_size, avg_obs_cst)
    disp_point = disp_points[
        grid_indices[0]][grid_indices[1]]
    # cone_diff = cone_diff[grid_indices[0]][grid_indices[1]]
    # set them equal to each other
    disp_point[1] = disp_point[0]
    noise_generator = LuminanceNoiseGenerator(avg_obs_cst, lum_noise)
    return disp_point, 0, noise_generator


def GetConeIdentifyingTest(observer: Observer, primaries: List[Spectra], metameric_axis: int,
                           luminance: float, saturation: float, grid_indices: tuple, noise_division: float = 1,
                           grid_size: int = 5, cube_idx: int = 4) -> tuple[npt.NDArray, NoiseGenerator]:

    cst = GetColorSpaceTransform(observer, primaries, metameric_axis=metameric_axis)
    points_disp = GetMaxMetamerOverGridSample(luminance, saturation, cube_idx, grid_size, cst)

    all_observers = GetAllObservers()
    color_space_transforms = GetColorSpaceTransformsOverObservers(
        all_observers, primaries, metameric_axis=metameric_axis)

    noise_generator = BackgroundNoiseGenerator(color_space_transforms, noise_division)

    return points_disp, noise_generator


def GetObserverIdentifyingTest(observer: Observer, primaries: List[Spectra], metameric_axis: int,
                               luminance: float, lum_noise: float, saturation: float, grid_indices: tuple,
                               grid_size: int = 5, cube_idx: int = 4) -> tuple[npt.NDArray, npt.NDArray, NoiseGenerator]:

    cst = GetColorSpaceTransform(observer, primaries, metameric_axis=metameric_axis)
    points_disps, cone_diffs = GetMaximalMetamerPointsOnGrid(luminance, saturation, cube_idx, grid_size, cst)
    points_disp = points_disps[grid_indices[0]][grid_indices[1]]
    cone_diff = cone_diffs[grid_indices[0]][grid_indices[1]]
    # noise_generator = LuminanceNoiseGenerator(cst, lum_noise)
    noise_generator = ConeLuminanceNoiseGenerator(cst, lum_noise)
    return points_disp, cone_diff, noise_generator


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
        self.current_plate: IshiharaPlateGenerator = IshiharaPlateGenerator(seed=self.seed)

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
