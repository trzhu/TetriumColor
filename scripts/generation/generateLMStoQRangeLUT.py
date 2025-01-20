from multiprocessing import process
import pdb
import argparse
from re import I
import numpy as np
import tqdm
import time
import numpy.typing as npt
from typing import List

from PIL import Image, ImageDraw
import TetriumColor.ColorMath.GamutMath as GamutMath
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, TetraColor
from TetriumColor.Observer import GetCustomObserver, Spectra, GetMaxBasisToDisplayTransform
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform
from TetriumColor.Utils.ParserOptions import AddObserverArgs
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import VSXYZToRYGB
from TetriumColor.ColorMath.Geometry import ConvertCubeUVToXYZ
from joblib import Parallel, delayed


parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observer = GetCustomObserver(wavelengths, od=0.5, m_cone_peak=args.m_cone_peak, q_cone_peak=args.q_cone_peak,
                             l_cone_peak=args.l_cone_peak, template=args.template, macular=args.macula, lens=args.lens)
# primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2025-01-16/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
    observer, primaries, metameric_axis=2, scaling_factor=10000)


def generate_3d_hypercube(depth: int = 4):
    grid = np.linspace(0, 1, 2**depth, dtype=np.float64)  # 8-bit values
    rybg = np.array(np.meshgrid(grid, grid, grid), copy=False).T.reshape(-1, 3)
    return rybg


def process_lms(lms):
    lmsq = np.append(lms, lms[2])  # dummy value, just need to set the direction
    q_range = GamutMath.GenerateLUTLMStoQ(lmsq, color_space_transform)
    if q_range is None:
        return None
    return np.hstack((lms, q_range))


# use this matrix to convert from RYGB to LMSQ
M_RYGB_to_SMQL = np.linalg.inv(
    color_space_transform.cone_to_disp)@np.flip(color_space_transform.maxbasis_to_disp, axis=1)

# # Runs in 7.559729814529419 for 2**4
start_time = time.time()
lms_hypercube = generate_3d_hypercube(depth=6)
results = Parallel(n_jobs=-1)(delayed(process_lms)(x) for x in lms_hypercube)
print("Elapsed time: ", time.time() - start_time, " for ", len(results), " results.")


# Get the Boundary of this Hypercube
# def generate_4d_hypercube():
#     grid = np.linspace(0, 1, 2**depth, dtype=np.float64)  # 8-bit values
#     rybg = np.array(np.meshgrid(grid, grid, grid, grid), copy=False).T.reshape(-1, 4)
#     return rybg

# # Step 1: Generate the RYGB Hypercube

# rgbo_to_smql = np.linalg.inv(color_space_transform.cone_to_disp)
# print("generating hypercube")
# rgbo_hypercube = generate_4d_hypercube()
# print("generating LUT")
# lut = []
# lmsq_hypercube = rgbo_hypercube@rgbo_to_smql.T

# def process_lmsq(lmsq):
#     q_range = GamutMath.GenerateLUTLMStoQ(lmsq, color_space_transform)
#     if q_range is None:
#         return None
#     return np.hstack((lmsq[[0, 1, 3]], q_range))

# start_time = time.time()

# # Runs in 7.559729814529419 for 2**4
# results = Parallel(n_jobs=-1)(delayed(process_lmsq)(x) for x in lmsq_hypercube)

# # results = []
# # for x in lmsq_hypercube:
# #     results.append(process_lmsq(x))

# print("Elapsed time: ", time.time() - start_time, " for ", len(results), " results.")


# Filter out None values from results
results = [result for result in results if result is not None]
lut = np.array(results, dtype=np.float32)
# Save in Vulkan-compatible format (e.g., binary or structured format)
lut.tofile("lut.bin")
print("LUT saved as 'lut.bin'.")
