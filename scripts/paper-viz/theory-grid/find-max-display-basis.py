import numpy as np

from TetriumColor.Observer import *


wavelengths = np.arange(380, 700, 5)
observers = [GetCustomObserver(wavelengths, dimension=3), GetCustomObserver(wavelengths, dimension=4)]

for observer in observers:
    print(observer)
    max_disp_basis = MaxDisplayGamut(observer, ChromaticityDiagramType.ConeBasis, verbose=True)
    print("Output of MaxDisplayGamut under ConeBasis")
    print(max_disp_basis.max_primaries_chrom)
    print(max_disp_basis.max_primaries_full)

    max_basis = MaxBasis(observer)
    max_disp_basis = MaxDisplayGamut(observer, ChromaticityDiagramType.HeringMaxBasisDisplay,
                                     transform_mat=max_basis.GetConeToMaxBasisTransform(), verbose=True)
    print("Output of MaxDisplayGamut under MaxBasis")
    print(max_disp_basis.max_primaries_chrom)
    print(max_disp_basis.max_primaries_full)
