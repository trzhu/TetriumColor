import argparse
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomTetraObserver
from TetriumColor import DisplayBasisType
import TetriumColor.Visualization as viz


def main():
    parser = argparse.ArgumentParser(description='Visualize Custom Tetra Observer.')
    parser.add_argument('--display_basis', type=lambda choice: DisplayBasisType[choice], choices=list(DisplayBasisType))
    parser.add_argument('--od', type=float, required=False, default=0.5, help='Optical Density')
    parser.add_argument('--s_cone_peak', type=int, required=False, default=419, help='S cone peak')
    parser.add_argument('--m_cone_peak', type=int, required=False, default=530, help='M cone peak')
    parser.add_argument('--l_cone_peak', type=int, required=False, default=559, help='L cone peak')
    parser.add_argument('--q_cone_peak', type=int, required=False, default=547, help='Q cone peak')
    parser.add_argument('--macula', type=float, required=False, default=1.0, help='Macula Pigment Density')
    parser.add_argument('--lens', type=float, required=False, default=1.0, help='Lens Density')
    parser.add_argument('--template', type=str, required=False, default='neitz',
                        help='Template for the cone fundamentals used')
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose output")
    args = parser.parse_args()

    wavelengths = np.arange(380, 781, 10)
    observer = GetCustomTetraObserver(wavelengths, args.od, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                      args.l_cone_peak, args.macula, args.lens, args.template)

    ps.init()
    viz.RenderOBS("tetra-custom-observer", observer, args.display_basis)
    viz.RenderMaxBasis("tetra-max-basis", observer, display_basis=args.display_basis)
    ps.show()


if __name__ == "__main__":
    main()
