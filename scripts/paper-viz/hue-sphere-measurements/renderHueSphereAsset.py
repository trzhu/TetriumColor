import argparse
import numpy as np
import tetrapolyscope as ps
import tifffile

import numpy.typing as npt
from typing import List

from TetriumColor.Observer import GetCustomObserver, ObserverFactory, GetParalleletopeBasis, convert_refs_to_spectras, Spectra, GetHeringMatrix
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform, getDisplayedSpectraFromWeights
from TetriumColor.Utils.CustomTypes import DisplayBasisType, ColorSpaceTransform
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.PsychoPhys.HueSphere import GenerateEachFaceOfGrid, GenerateHueSpherePoints
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries, LoadMetamers


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=5, help='Step size for wavelengths')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(360, 831, 1)

    observer = GetCustomObserver(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)
    factor = 0.1/5.25
    viz.ps.set_background_color((factor, factor, factor, 1))

    viz.RenderOBS("observer", observer, args.display_basis, num_samples=100000)
    ps.get_surface_mesh("observer").set_transparency(0.5)

    viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    viz.RenderMaxBasis("max-basis", observer, args.display_basis)
    viz.AnimationUtils.AddObject("max-basis", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    ps.set_automatically_compute_scene_extents(False)

    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=50., aspect=1.)
    params = ps.CameraParameters(intrinsics, ps.get_view_camera_parameters().get_extrinsics())
    ps.set_view_camera_parameters(params)
    # Output Video to Screen or Save to File (based on options)
    if args.output_filename:
        fd = viz.OpenVideo(args.output_filename)
        total_frames = int(360 / args.rotation_speed * args.fps)
        args.total_frames = total_frames
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
