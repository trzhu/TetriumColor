from abc import ABC, abstractmethod
from typing import List
from TetriumColor.Utils.CustomTypes import *
import numpy as np
import pickle
import os

from TetriumColor.PsychoPhys.Quest import Quest
from TetriumColor.Utils.IO import loadColorSpaceTransform
from TetriumColor.Utils.MathHelper import sampleFibonacciSphere
from TetriumColor.ColorMath.Metamers import getKMetamers


# TODO: Implement the following classes
class BackgroundNoiseGenerator: 
    def __init__(self):
        pass

    def NoiseGenerator(self) -> PlateColor:
        pass


class ColorGenerator(ABC):
    def __init__(self, transform_dir: str):
        """
        Initializes the ColorGenerator with the given directory to load the ColorSpaceTransform
        Args:
        transformDir (str): The directory to load the ColorSpaceTransform from
        """
        self.color_space_transform : ColorSpaceTransform = loadColorSpaceTransform(transform_dir)
    
    @staticmethod
    def _GenerateQuestObject(tGuess, tGuessSd) -> Quest:
        # TODO: For now don't mess with quest, just sample at an interval to get idea of parameters
        return Quest(tGuess,tGuessSd,0.8,beta=3.5,delta=0.01,gamma=0.05,grain=0.01,range=None)

    @abstractmethod
    def NewColor(self) -> List[PlateColor]:
        pass
    
    @abstractmethod
    def GetColor(self, previous_result: ColorTestResult) -> List[PlateColor]:
        pass


class TestColorGenerator(ColorGenerator):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        # super().__init__() # don't want to initialize a directory for testing

    def NewColor(self) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([0, 255, 0], dtype=int), np.array([255, 0, 0], dtype=int)), background=TetraColor(np.array([255, 255, 0], dtype=int), np.array([0, 255, 255], dtype=int)))

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([255, 0, 0]), np.array([0, 255, 0])), background=TetraColor(np.array([255, 255, 0]), np.array([0, 255, 255])))


class ScreeningTestColorGenerator(ColorGenerator):
    def __init__(self, num_tests: int, transform_dirs: str, pre_generated_filenames:str=None):

        self.num_tests = num_tests
        self.metamer_list : List[PlateColor] = []
        for transform_dir, pre_generated_filename in zip(transform_dirs, pre_generated_filenames):
            colorSpaceTransform = loadColorSpaceTransform(transform_dir)

            if pre_generated_filename is not None and os.path.exists(pre_generated_filename):
                with open(pre_generated_filename, 'rb') as pickle_file:
                    self.metamer_list += pickle.load(pickle_file)
            else:
                metamer_list = getKMetamers(colorSpaceTransform, self.num_tests)
                if pre_generated_filename is not None:
                    with open(pre_generated_filename, 'wb') as pickle_file:
                        pickle.dump(metamer_list, pickle_file)
                self.metamer_list += metamer_list
        self.currentIdx = 0

    def NewColor(self) -> PlateColor:
        plateColor = self.metamer_list[self.currentIdx]
        self.currentIdx += 1
        return plateColor

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        # TODO: incorporate feedback but for now not necessary
        if self.currentIdx >= self.num_tests:
            return
        return self.NewColor()
   

class TargetedTestColorGenerator(ColorGenerator):

    def __init__(self, transform_dir: str, num_trials: int = 10):
        super().__init__(transform_dir)
        self.current_idx = 0
        # self.metameric_axis = self._computeMetamericAxis()
        maxSaturation = 0.4 # need to find correct number
        self.sampled_points = [self.metameric_axis * maxSaturation * i / num_trials for i in range(num_trials)]
        self.sampled_points.reverse()

    def NewColor(self) -> PlateColor:
        # colors = self.metamer_list[self.currentIdx]
        self.current_idx += 1
        # return PlateColor(shape=metamer[0], background=metamer[1])

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        pass


class InDepthTestColorGenerator(ColorGenerator):
    
    def __init__(self, transform_dir: str, num_directions: int = 25):
        super().__init__(transform_dir)
        self.num_directions = num_directions
        self.hue_directions = sampleFibonacciSphere(self.num_directions)

    def NewColor(self) -> PlateColor:
        pass
    
    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        pass