from abc import ABC, abstractmethod
from typing import List
from TetriumColor.Utils.CustomTypes import *
import numpy as np

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
    def __init__(self, transformDir: str):
        """
        Initializes the ColorGenerator with the given directory to load the ColorSpaceTransform
        Args:
        transformDir (str): The directory to load the ColorSpaceTransform from
        """
        self.colorSpaceTransform : ColorSpaceTransform = loadColorSpaceTransform(transformDir)
    

    @staticmethod
    def _GenerateQuestObject(tGuess, tGuessSd) -> Quest:
        # TODO: For now don't mess with quest, just sample at an interval to get idea of parameters
        return Quest(tGuess,tGuessSd,0.8,beta=3.5,delta=0.01,gamma=0.05,grain=0.01,range=None)

    @abstractmethod
    def NewColor(self) -> List[PlateColor]:
        pass
    
    @abstractmethod
    def GetColor(self, previousResult: ColorTestResult) -> List[PlateColor]:
        pass


class TestColorGenerator(ColorGenerator):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        # super().__init__() # don't want to initialize a directory for testing

    def NewColor(self) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([0, 255, 0], dtype=int), np.array([255, 0, 0], dtype=int)), background=TetraColor(np.array([255, 255, 0], dtype=int), np.array([0, 255, 255], dtype=int)))

    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([255, 0, 0]), np.array([0, 255, 0])), background=TetraColor(np.array([255, 255, 0]), np.array([0, 255, 255])))


class ScreeningTestColorGenerator(ColorGenerator):
    def __init__(self, num_tests: int, transformDir: str):
        super().__init__(transformDir)

        self.num_tests = num_tests
        self.metamer_list: List[PlateColor] = getKMetamers(self.colorSpaceTransform, self.num_tests)
        print(f"Metamers: {self.metamer_list}")
        self.currentIdx = 0

    def NewColor(self) -> PlateColor:
        plateColor = self.metamer_list[self.currentIdx]
        self.currentIdx += 1
        return plateColor

    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        # TODO: incorporate feedback but for now not necessary
        if self.currentIdx >= self.num_tests:
            return
        return self.NewColor()
   

class TargetedTestColorGenerator(ColorGenerator):

    def __init__(self, transformDir: str, numTrials: int = 10):
        super().__init__(transformDir)
        self.metamericAxis = self._computeMetamericAxis()
        maxSaturation = 0.4 # need to find correct number
        self.SampledPoints = [self.metamericAxis * maxSaturation * i / numTrials for i in range(numTrials)]
        self.SampledPoints.reverse()

    def NewColor(self) -> PlateColor:
        # colors = self.metamer_list[self.currentIdx]
        self.currentIdx += 1
        # return PlateColor(shape=metamer[0], background=metamer[1])

    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        pass


class InDepthTestColorGenerator(ColorGenerator):
    
    def __init__(self, transformDir: str, num_directions: int = 25):
        super().__init__(transformDir)

        self.num_directions = num_directions
        self.hue_directions = sampleFibonacciSphere(self.num_directions)


    def NewColor(self) -> PlateColor:
        pass
    
    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        pass