import argparse
import numpy as np

from TetriumColor.Observer import Observer, MaxBasisFactory
from TetriumColor.Observer.Spectra import Illuminant
from TetriumColor.Utils.ParserOptions import *
from TetriumColor import ColorSpace, ColorSpaceType


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=5, help='Step size for wavelengths')
    parser.add_argument('--primary_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the ideal chromatic display')
    parser.add_argument('--viz_efficient_wavelengths', nargs='+', type=float, default=[445, 535, 590, 635],
                        help='Wavelengths for the visually efficient display')
    parser.add_argument('--display_type', choices=['ideal', 'viz-efficient',
                        'ours'], default='ours', help='Type of display to visualize')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(380, 781, 5)

    observer = Observer.custom_observer(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)

    cst = ColorSpace(observer)

    max_basis = MaxBasisFactory.get_object(observer, denom=1, verbose=False)

    print("Maxbasis cutpoints: ", max_basis.cutpoints)
    max_basis_243 = MaxBasisFactory.get_object(observer, denom=2.43, verbose=False)
    print("Maxbasis cutpoints 2.43: ", max_basis_243.cutpoints)
    max_basis_3 = MaxBasisFactory.get_object(observer, denom=3, verbose=False)
    print("Maxbasis cutpoints 3: ", max_basis_3.cutpoints)
    refs, _, _, _ = max_basis_3.GetDiscreteRepresentation()
    print([r.to_hex(illuminant=Illuminant.get("D65")) for r in refs[1: args.dimension + 1]])
    cones = observer.observe_spectras(refs[1: args.dimension + 1])
    srgbs = cst.convert(cones, ColorSpaceType.CONE, ColorSpaceType.SRGB)
    print("sRGB: ", srgbs)

    # Apply gamma correction
    gamma = 2.2
    srgb_with_gamma = np.clip(cones, 0, 1) ** (gamma)
    print("sRGB with gamma correction applied: ", srgb_with_gamma)


main()
