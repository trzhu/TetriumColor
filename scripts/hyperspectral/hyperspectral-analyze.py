import argparse
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomObserver, ObserverFactory
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.PsychoPhys.HyperspectralImage import ProjectHyperSpectral
from TetriumColor.Utils.ParserOptions import *


def main():

    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--hyperspectral_filename', type=str, required=True,
                        help='Filename for hyperspectral image')
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(400, 701, 10)

    observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)

    points = ProjectHyperSpectral(args.hyperspectral_filename, observer).reshape(-1, args.dimension)
    transformed_pts = viz.ConvertPointsToBasis(points, observer, args.display_basis)[::100]
    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    # Create Geometry & Register with Polyscope, and define the animation
    viz.RenderOBS("observer", observer, args.display_basis)
    ps.get_surface_mesh("observer").set_transparency(0.5)

    viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    viz.RenderPointCloud("points", transformed_pts, np.ones((transformed_pts.shape[0], 3)) * 0.5, radius=0.001)

    # Need to call this after registering structures
    ps.set_automatically_compute_scene_extents(False)
    # Output Video to Screen or Save to File (based on options)
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
