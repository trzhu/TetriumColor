"""
Metamers.py
This module contains functions to calculate and finetune metamers 
for a given Display->Cone Basis on a given axis in cone space.
"""

from typing import List, Dict, Union
import numpy.typing as npt
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor

############################################################################################################
# Varun's Functions
def BucketPoints(points: npt.NDArray , axis:int=2, prec:float=0.005, exponent:int=8) -> Dict:
    # disjointed buckets
    buckets = defaultdict(list)
    N, d = points.shape

    # prec = 0.005
    # prec = 0.0005
    # prec = 0.0000005
    # 8 is large enough for prec = 0.005:
    # 8 > log_2 (1 / 0.005)
    # 22 > log_2(1/0.0000005)
    # 14 > log_2(1/0.00005)
    # weights = (2 ** (8 * np.arange(0, d)))
    weights = (2 ** (exponent * np.arange(0, d)))
    # weights = (2 ** (22 * np.arange(0, d)))
    weights[axis] = 0

    values = points // prec
    keys = values @ weights
    for i, (key, point) in enumerate(zip(keys, values)):
        buckets[key].append((point / 2, i))  # can replace 2 with 0.01 // prec

    return {k: v for k, v in buckets.items() if len(v) > 1}


def SortBuckets(buckets:defaultdict, axis:int=2) -> List:
    dist_buckets = []

    for metamers in buckets.values():
        if len(metamers) <= 1:
            continue

        axis_values = [metamer[0][axis] for metamer in metamers]

        min_val = min(axis_values)
        max_val = max(axis_values)

        distance = max_val - min_val

        min_index = axis_values.index(min_val)
        max_index = axis_values.index(max_val)
        best_indices = (metamers[min_index][1], metamers[max_index][1])

        dist_buckets.append((distance, best_indices))

    return sorted(dist_buckets, reverse=True)

def GetMetamerBuckets(points: npt.ArrayLike, axis:int=2, prec:float =0.005, exponent:int=8) -> List:
    sorted_points = []
    buckets = SortBuckets(BucketPoints(points, axis=axis, prec=prec, exponent=exponent), axis=axis)
    for dst, (i, j) in buckets:
        sorted_points.append((dst, (tuple(points[i]), tuple(points[j]))))
    sorted_points.sort(reverse=True)
    return sorted_points

############################################################################################################

def GetSampledHyperCube(step_size: float, dimension: int, outer_range: List[List[float]]=[[0, 1], [0, 1], [0, 1], [0, 1]]) -> npt.ArrayLike:
    """
    Get a hypercube sample of the space
    """
    g = np.meshgrid(*[np.arange(outer_range[i][0], outer_range[i][1] + 0.0000001, step_size) for i in range(dimension)])
    return np.array(list(zip(*(x.flat for x in g))))


def _getRefinedHypercube( metamer_led_weights: npt.ArrayLike, previous_step:float) -> npt.ArrayLike:
    """
    Get a hypercube around a points and a range.
    Args:
        metamer_led_weights (npt.ArrayLike): The metamer weights that we want to finetune around
        previous_step (float): The step size for the hypercube, to set the width of the hypercube
    """
    outer_range = [ [metamer_led_weights[i] - previous_step, metamer_led_weights[i] + previous_step ]for i in range(4)]
    outer_range = [[round(max(0, min(1, outer_range[i][0])) * 255) / 255, 
            round(max(0, min(1, outer_range[i][1])) * 255) / 255] 
                for i in range(4)]
    return GetSampledHyperCube(0.004, 4, outer_range)


def _getTopKMetamers(M_WeightsInCone: npt.ArrayLike, hypercube: npt.ArrayLike, metameric_axis:int, K:int, prec:float =0.005, exponent:int=8)->npt.ArrayLike:
    """
    Get the Top K Metamers (in Display Space) for a Given Transform
    Args:
        M_weightToCone (npt.ArrayLike): Transform from display weights to Cone Space
        hypercube (float): Point Cloud in the Display Space That We're Using to Sample
        metameric_axis (int): The axis that we are bucketizing against in Cone Space (default is 2 for SMQL for Q)
        prec (float): The precision of the bucketing (default is 0.005)
        exponent (int): The exponent for the bucketing (default is 8)
    """
    all_lms_intensities = (M_WeightsInCone@hypercube.T).T # multiply all possible led combinations with the intensities
    buckets = GetMetamerBuckets(all_lms_intensities, axis=metameric_axis, prec=prec, exponent=exponent)
    random_indices = np.random.randint(0, len(buckets) //10, K)
    metamers = np.zeros((K, 2, 4))
    for i, idx in enumerate(random_indices):
        metamers[i][0]= buckets[idx][1][0]
        metamers[i][1] = buckets[idx][1][1]
    metamers = metamers.reshape((K * 2, 4))
    metamers = (np.linalg.inv(M_WeightsInCone)@metamers.T).T
    return metamers.reshape(K, 2, 4)   # return in display space weight coordinates


def _refineMetamer(weights_1:npt.ArrayLike, weights_2:npt.ArrayLike, M_weightToCone:npt.ArrayLike, metameric_axis:int=2, hypercube_sample:float=0.01)-> npt.ArrayLike:
    """
    Refine the estimate of the metamers by sampling a smaller hypercube around each of the pair of metamers
    Args:
        weights_1 (npt.ArrayLike): The weights of the first metamer
        weights_2 (npt.ArrayLike): The weights of the second metamer
        M_weightToCone (npt.ArrayLike): The matrix that converts the led weights to the cone space
        metameric_axis (int): The axis that we are looking for the metamers
        hypercube_sample (float): The step size for the hypercube
    """
    
    hypercube1 = _getRefinedHypercube(weights_1, hypercube_sample * 2)
    hypercube2 = _getRefinedHypercube(weights_2, hypercube_sample * 2)
    hypercube = np.vstack([hypercube1, hypercube2])
    return _getTopKMetamers(M_weightToCone, hypercube, K=1, metameric_axis=metameric_axis, prec=0.0005, exponent=11)


def _refineMetamers(metamers: npt.ArrayLike, M_weightToCone: npt.ArrayLike, metameric_axis:int=2, hypercube_sample:float=0.01)-> npt.ArrayLike:
    """
    Refine the estimate of the metamers by sampling a smaller hypercube around each of the pair of metamers
    Args:
        metamers (npt.ArrayLike): The metamers to refine
        M_weightToCone (npt.ArrayLike): The matrix that converts the led weights to the cone space
        metameric_axis (int): The axis that we are looking for the metamers
        hypercube_sample (float): The step size for the hypercube
    """
    refined_metamers = []
    print("Refining Metamers")
    for met in tqdm(metamers):
        refined = _refineMetamer(met[0], met[1], M_weightToCone, metameric_axis=metameric_axis, hypercube_sample=hypercube_sample)
        refined_metamers.append(refined)
    return np.concatenate(refined_metamers)


def ConvertToPlateColors(colors: npt.ArrayLike, transform: ColorSpaceTransform) -> List[PlateColor]:
    """
    Nx4 Array, transform into PlateColor
    """
    mat = np.zeros((colors.shape[0], 6))
    for i, mappedIdx in enumerate(transform.display_basis):
        mat[:, mappedIdx] = colors[:, i]
    
    metamers = mat.reshape((colors.shape[0]//2, 2, 6))
    plateColors : List[PlateColor] = []
    for i in range(metamers.shape[0]):
        plateColors += [PlateColor(TetraColor(metamers[i][0][:3], metamers[i][0][3:]), TetraColor(metamers[i][1][:3], metamers[i][1][3:]))]
    return plateColors

# TODO: Modify method later in order to return a point close to a given direction in the cone space
# Or get top K that are furthest apart. For now return a random top K from top 100
def GetKMetamers(transform: ColorSpaceTransform, K:int, hypercubeSample:float = 0.05) -> List[PlateColor]:
    """
    Get the metamers for the given matrix and axis, and return the weights of the primaries for the metamers
    Args:
        M_weightToCone (npt.ArrayLike): The matrix that converts the led weights to the cone space
        metameric_axis (int): The axis that we are looking for the metamers
        hypercube_sample (float): The step size for the hypercube (default is 0.05)
    """
    invMat = np.linalg.inv(transform.cone_to_disp)
    hypercube = GetSampledHyperCube(hypercubeSample, 4) # takes no time at 0.05. 
    metamers_first_pass = _getTopKMetamers(invMat, hypercube, metameric_axis=transform.metameric_axis, K=K)
    final_metamers = _refineMetamers(metamers_first_pass, invMat, metameric_axis=transform.metameric_axis, hypercube_sample=hypercubeSample)
    return ConvertToPlateColors(final_metamers.reshape(-1, 4), transform)