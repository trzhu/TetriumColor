import numpy as np

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform


def loadColorSpaceTransform(directory:str) -> ColorSpaceTransform:
    """
    Load a Color Space Transform from a file
    Args:
        filename (str): The filename of the Color Space Transform
    """
    ConeToDisp = np.load(directory + "/ConeToDisp.npy")
    MaxBasisToDisp = np.load(directory + "/MaxBasisToDisp.npy")
    HeringToDisp = np.load(directory + "/HeringToDisp.npy")
    Axis = np.load(directory + "/Axis.npy")[0]
    DisplayBasis = np.load(directory + "/DisplayBasis.npy").tolist()
    return ColorSpaceTransform(ConeToDisp, MaxBasisToDisp, HeringToDisp, Axis, DisplayBasis)