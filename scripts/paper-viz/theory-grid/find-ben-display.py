

import numpy as np

from TetriumColor.Observer import *


wavelengths = np.arange(380, 700, 5)
observers = [GetCustomObserver(wavelengths, dimension=3), GetCustomObserver(wavelengths, dimension=4)]
ben_display_peaks = np.array([410, 455, 475, 500, 530, 570, 615, 625, 660])

for observer in observers:
    print(observer)
    max_basis = MaxBasis(observer)
    max_disp_basis = MaxDisplayGamut(observer, ChromaticityDiagramType.HeringMaxBasisDisplay,
                                     transform_mat=max_basis.GetConeToMaxBasisTransform(), verbose=True)
    print("Output of MaxDisplayGamut under MaxBasis")
    print(max_disp_basis.max_primaries_chrom)
    print(max_disp_basis.max_primaries_full)

    print("From Ben's Peaks")
    print(max_disp_basis.ComputeMaxPrimariesInChrom(ben_display_peaks))
    print(max_disp_basis.ComputeMaxPrimariesInFull(ben_display_peaks))
