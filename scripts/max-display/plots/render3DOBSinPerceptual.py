import argparse
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import Observer
import TetriumColor.Visualization as viz
from TetriumColor import ColorSpace, ColorSpaceType
from TetriumColor.Utils.ParserOptions import *


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=10, help='Step size for wavelengths')
    parser.add_argument('--ideal', action='store_true', default=False, help='Use ideal primaries')
    parser.add_argument('--primary_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the display')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(380, 781, 10)

    observer = Observer.custom_observer(observer_wavelengths, args.od, 3, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    viz.RenderOBS("observer", observer, PolyscopeDisplayType.HERING_MAXBASIS_PERCEPTUAL_3)
    ps.get_surface_mesh("observer").set_transparency(0.8)

    ps.set_automatically_compute_scene_extents(False)

    if args.output_filename:
        fd = viz.OpenVideo(args.output_filename)
        viz.RenderVideo(fd, args.total_frames, args.fps)
        viz.CloseVideo(fd)
    else:
        delta_time: float = 1 / args.fps

        def callback():
            pass
            # viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
