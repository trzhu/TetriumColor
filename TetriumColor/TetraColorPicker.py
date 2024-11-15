from abc import ABC, abstractmethod
from typing import List
from TetriumColor.Utils.CustomTypes import *
import numpy as np

from TetriumColor.PsychoPhys.Quest import Quest
from TetriumColor.Utils.MathHelper import sampleFibonacciSphere


# TODO: Implement the following classes
class BackgroundNoiseGenerator: 
    def __init__(self):
        pass

    def NoiseGenerator(self) -> PlateColor:
        pass


class ColorGenerator(ABC):
    def __init__(self):
        pass
    
    @staticmethod
    def _GenerateQuestObject(tGuess, tGuessSd) -> Quest:
        return Quest(tGuess,tGuessSd,0.8,beta=3.5,delta=0.01,gamma=0.05,grain=0.01,range=None)

    @abstractmethod
    def NewColor(self) -> List[PlateColor]:
        pass
    
    @abstractmethod
    def GetColor(self, previousResult: ColorTestResult) -> List[PlateColor]:
        pass


class ScreeningTestColorGenerator(ColorGenerator):

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        super().__init__()

    def NewColor(self) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([0, 255, 0], dtype=int), np.array([255, 0, 0], dtype=int)), background=TetraColor(np.array([255, 255, 0], dtype=int), np.array([0, 255, 255], dtype=int)))

    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([255, 0, 0]), np.array([0, 255, 0])), background=TetraColor(np.array([255, 255, 0]), np.array([0, 255, 255])))


class TargetedTestColorGenerator(ColorGenerator):

    def __init__(self):
        super().__init__()
        self.quest_objs = [ColorGenerator._GenerateQuestObject(0.5, 0.1) for i in range(2)]

    def NewColor(self) -> PlateColor:
        pass

    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        pass


class InDepthTestColorGenerator(ColorGenerator):
    
    def __init__(self, num_directions=25):
        super().__init__()
        self.num_directions = num_directions
        self.quest_objs = [ColorGenerator._GenerateQuestObject(0.5, 0.1) for i in range(self.num_directions)]

        self.hue_directions = sampleFibonacciSphere(self.num_directions)


    def NewColor(self) -> PlateColor:
        pass
    
    def GetColor(self, previousResult: ColorTestResult) -> PlateColor:
        pass