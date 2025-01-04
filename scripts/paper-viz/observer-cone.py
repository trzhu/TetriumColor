import argparse
import numpy as np
import tetrapolyscope as ps
import time

from TetriumColor.Observer import GetCustomObserver, ObserverFactory
from TetriumColor.Observer.Observer import GetHeringMatrix
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import AddObserverArgs, AddVideoOutputArgs


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
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
    position = [0, 0, 0]
    velocity = [0, 0, 0]  # [-0.02, 0, 0]
    rotation_speed = 10
    rotation_axis = [0, 1, 0]

    # Create Geometry & Register with Polyscope, and define the animation

    viz.RenderPointCloud("center-zero", np.array([np.zeros(3)]))

    points = viz.ConvertPointsToBasis(observer.normalized_sensor_matrix.T, observer, args.display_basis)
    basis_points = viz.ConvertPointsToBasis(np.eye(args.dimension), observer, DisplayBasisType.ConeHering)

    viz.Render3DCone("observer-cone", points, np.zeros(3), 1, 1)
    viz.RenderBasisArrows("basis", basis_points * 0.3)

    viz.AnimationUtils.AddObject("observer-cone", "surface_mesh",
                                 position, velocity, rotation_axis, rotation_speed)
    viz.AnimationUtils.AddObject("observer-cone_arrows", "surface_mesh",
                                 position, velocity, rotation_axis, rotation_speed)
    viz.AnimationUtils.AddObject("observer-cone_curve", "curve_network",
                                 position, velocity, rotation_axis, rotation_speed)

    viz.AnimationUtils.AddObject("basis", "surface_mesh",
                                 position, velocity, rotation_axis, rotation_speed)

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
