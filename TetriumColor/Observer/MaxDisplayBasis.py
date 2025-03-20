import numpy as np
import numpy.typing as npt

from itertools import combinations
from enum import Enum
from functools import reduce
from typing import List
from collections.abc import Callable

from .Observer import Observer, GetHeringMatrix


class ChromaticityDiagramType(Enum):
    XY = 0
    ConeBasis = 1
    HeringMaxBasisDisplay = 2


class MaxDisplayGamut:

    @staticmethod
    def TransformToDisplayChromaticity(matrix, T, idxs=None) -> npt.NDArray:
        """
        Transform Coordinates (dim x n_rows) into Display Chromaticity Coordinates (divide by Luminance)
        """
        return (T@(matrix / np.sum(matrix, axis=0)))[idxs]

    def __init__(self, observer: Observer, chromaticity_diagram_type: ChromaticityDiagramType = ChromaticityDiagramType.ConeBasis,
                 transform_mat: npt.NDArray | None = None, projection_idxs: List[int] | None = None, verbose: bool = False) -> None:
        # TODO: Create the rest of them
        """Create a MaxDisplayGamut object that computes the max display gamut for a given observer

        Args:
            observer (Observer): Observer object
            chromaticity_diagram_type (ChromaticityDiagramType, optional): type to perform the chromaticity analysis. Defaults to ChromaticityDiagramType.ConeBasis.
            transformMatrix (npt.NDArray | None, optional): transform matrix before projection. Defaults to None.
            projection_idxs (List[int] | None, optional): which indices to take after projection. Defaults to None.
            verbose (bool, optional): have the routine explain itself. Defaults to False.
        """
        self.verbose: bool = verbose
        self.observer: Observer = observer
        self.wavelengths: npt.NDArray = observer.wavelengths
        self.matrix: npt.NDArray = observer.get_normalized_sensor_matrix()
        self.dimension: int = observer.dimension
        idxs: List[int] = list(
            range(1, self.dimension)) if projection_idxs is None else projection_idxs
        chromaticity_transform: Callable[[npt.NDArray], npt.NDArray] = self.GetChromaticityConvertFn(
            chromaticity_diagram_type, transform_mat, idxs=idxs)
        self.chromaticity_mat: npt.NDArray = chromaticity_transform(self.matrix)
        self.max_primaries_full = self.ComputeMaxPrimariesInFull()
        self.max_primaries_chrom = self.ComputeMaxPrimariesInChrom()

    def GetChromaticityConvertFn(self, chrom_diag_type: ChromaticityDiagramType,
                                 transformMatrix: npt.NDArray | None = None, idxs: List[int] | None = None):
        T = np.eye(self.observer.dimension)
        match chrom_diag_type:
            case ChromaticityDiagramType.XY:
                if self.observer.dimension != 3:
                    raise ValueError(
                        "Chromaticity Diagram Type XY is only supported for 3 dimensions")
                raise NotImplementedError(
                    "Chromaticity Diagram Type XY is not implemented")

            case ChromaticityDiagramType.ConeBasis:
                return lambda pts: MaxDisplayGamut.TransformToDisplayChromaticity(pts, T, idxs=idxs)

            case ChromaticityDiagramType.HeringMaxBasisDisplay:
                if transformMatrix is not None:
                    dim = self.observer.dimension

                    def conv_chromaticity(pts):
                        vecs = (transformMatrix@pts)
                        T = GetHeringMatrix(dim)
                        return MaxDisplayGamut.TransformToDisplayChromaticity(vecs, T, idxs=idxs)
                    return conv_chromaticity
                else:
                    raise ValueError(
                        "Transform Matrix is not set with ChromaticityDiagramType.HeringMaxBasis")

    def ComputeSimplexVolume(self, wavelengths: tuple[float | int]) -> float:
        """Routine to compute the volume of a simplex in the chromaticity space

        Args:
            wavelengths (tuple[float  |  int]): wavelengths in which to select the monochromatic primaries

        Returns:
            float: the volume of the simplex
        """
        # pick index wavelengths from list of wavelengths
        idxs = np.searchsorted(self.wavelengths, wavelengths)
        mat = np.ones((self.dimension, self.dimension))
        # list of basis vectors per index in cone space
        submat = self.chromaticity_mat[:, idxs]
        mat[:, 1:] = submat.T
        fact = reduce(lambda x, y: x*y, range(1, self.dimension + 1))
        vol = np.abs(np.linalg.det(mat))/fact
        return vol

    def ComputeMaxPrimariesInFull(self, wavelengths: npt.NDArray | None = None) -> tuple[float | int]:
        """Routine to compute the Visually-Efficient Primaries given an observer

        Returns:
            tuple[float | int]: the maximal primaries in full space
        """

        def ComputeParalleletopeVolume(wavelengths):
            # set of n wavelengths, and we want to pick the monochromatic wavelength in each direction
            # pick index wavelengths from list of wavelengths
            idxs = np.searchsorted(self.wavelengths, wavelengths)
            # list of basis vectors per index in cone space
            submat = self.matrix[:, idxs]
            # volume of resulting parallelepiped -> according to chatGPT, it is divided by a fixed 1/n! factor based on dimensionality, so doesn't matter if we divide or not.
            vol = np.linalg.det(submat)
            return vol

        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        data = list(combinations(wavelengths, self.dimension))
        result = list(map(ComputeParalleletopeVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        return max_primaries

    def ComputeMaxPrimariesInChrom(self, wavelengths: npt.NDArray | None = None) -> tuple[float | int]:
        """Compute the maximal primaries in chromaticity space. 

        Args:
            wavelengths (npt.NDArray | None, optional): all possible wavelengths to do the 
            computation over. Defaults to None, which means that the object's wavelengths are used.

        Returns:
            tuple[float | int]: the maximal wavelengths in chromaticity space
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        data = list(combinations(wavelengths, self.dimension))
        result = list(map(self.ComputeSimplexVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        return max_primaries
