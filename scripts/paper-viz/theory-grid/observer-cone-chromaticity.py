import argparse
from re import I
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomObserver, ObserverFactory
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 781, 10)

    observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    projection_idxs = list(range(1, observer.dimension))
    # Create Geometry & Register with Polyscope, and define the animation

    points = viz.ConvertPointsToChromaticity(observer.normalized_sensor_matrix.T, observer, projection_idxs)
    basis_points = viz.ConvertPointsToChromaticity(np.eye(args.dimension), observer, projection_idxs)
    points = points[~np.all(points == 0, axis=1)]
    if args.dimension < 4:
        points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))
        basis_points_3d = np.hstack((basis_points, np.zeros((basis_points.shape[0], 1))))
        viz.Render3DLine("spectral_locus", points_3d, np.array([0.25, 0, 1]) * 0.5, 1)
        viz.Render2DMesh("gamut", points, np.array([0.25, 0, 1]) * 0.5)
        ps.get_surface_mesh("gamut").set_transparency(0.4)
    else:
        points_3d = points
        basis_points_3d = basis_points
        viz.Render3DLine("spectral_locus", points_3d, np.array([0.25, 0, 1]) * 0.5, 1)
        viz.Render3DMesh("gamut", points_3d, rgbs=np.tile(np.array([0.25, 0, 1]) * 0.5, (points_3d.shape[0], 1)))
        ps.get_surface_mesh("gamut").set_transparency(0.4)
        viz.AnimationUtils.AddObject("spectral_locus", "curve_network", args.position,
                                     args.velocity, args.rotation_axis, args.rotation_speed)
        viz.AnimationUtils.AddObject("gamut", "surface_mesh",
                                     args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # viz.RenderPointCloud("basis", basis_points_3d, np.eye(3))
    # viz.Render2DMesh("basis_mesh", basis_points, np.eye(3))
    # ps.get_surface_mesh("basis_mesh").set_transparency(0.5)

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
            viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
