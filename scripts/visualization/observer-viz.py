import argparse
import numpy as np
import tetrapolyscope as ps
import time

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

    # Parser Args for Animation / Video Saving
    parser.add_argument("--output_filename", type=str, required=False,
                        help="Output file name. If none is specified, the video will not be saved.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")
    parser.add_argument("--total_frames", type=int, default=200, help="Total number of frames to render")

    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 781, 10)
    observer = GetCustomTetraObserver(wavelengths, args.od, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                      args.l_cone_peak, args.macula, args.lens, args.template)

    # Polyscope Animation Inits
    ps.init()
    position = [0, 0, 0]
    velocity = [0, 0, 0]  # [-0.02, 0, 0]
    rotation_speed = 3
    rotation_axis = [0, 1, 0]

    # Create Geometry & Register with Polyscope, and define the animation
    viz.RenderOBS("tetra-custom-observer", observer, args.display_basis)
    ps.get_surface_mesh("tetra-custom-observer").set_transparency(0.5)

    viz.AnimationUtils.AddObject("tetra-custom-observer", "surface_mesh",
                                 position, velocity, rotation_axis, rotation_speed)

    viz.RenderMaxBasis("tetra-max-basis", observer, display_basis=args.display_basis)
    viz.AnimationUtils.AddObject("tetra-max-basis", "surface_mesh", position, velocity, rotation_axis, rotation_speed)

    # Output Video to Screen or Save to File (based on options)
    if args.output:
        fd = viz.OpenVideo(args.output)
        viz.RenderVideo(fd, args.total_frames, args.fps)
        viz.CloseVideo(fd)
    else:
        delta_time: float = 1 / args.fps
        while True:
            viz.AnimationUtils.UpdateObjects(delta_time)
            ps.frame_tick()


if __name__ == "__main__":
    main()
