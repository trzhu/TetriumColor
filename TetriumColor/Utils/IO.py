import numpy as np

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform


def LoadColorSpaceTransform(directory: str) -> ColorSpaceTransform:
    """
    Load a Color Space Transform from a file
    Args:
        filename (str): The filename of the Color Space Transform
    """
    cone_to_disp = np.load(directory + "/ConeToDisp.npy")
    maxbasis_to_disp = np.load(directory + "/MaxBasisToDisp.npy")
    hering_to_disp = np.load(directory + "/HeringToDisp.npy")
    axis = np.load(directory + "/Axis.npy")[0]
    display_basis = np.load(directory + "/DisplayBasis.npy").tolist()
    white_weights = np.load(directory + "/WhiteWeights.npy")
    # white_weights = np.ones(white_weights.shape)
    dim = cone_to_disp.shape[0]
    return ColorSpaceTransform(dim, cone_to_disp, maxbasis_to_disp, hering_to_disp, axis, display_basis, white_weights)
