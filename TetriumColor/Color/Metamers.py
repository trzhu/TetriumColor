"""
Metamers.py
This module contains functions to calculate and finetune metamers 
for a given Display->Cone Basis on a given axis in cone space.
"""

from typing import List, Dict, Union
import numpy.typing as npt
from collections import defaultdict

import numpy as np

############################################################################################################
# Varun's Functions
def bucket_points(points: npt.NDArray , axis:int=2, prec:float=0.005, exponent:int=8) -> Dict:
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


def sort_buckets(buckets, axis=2) -> List:
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

def get_metamer_buckets(points, axis=2, prec=0.005, exponent=8) -> List:
    sorted_points = []

    buckets = sort_buckets(bucket_points(points, axis=axis, prec=prec, exponent=exponent), axis=axis)
    for dst, (i, j) in buckets:
        sorted_points.append((dst, (tuple(points[i]), tuple(points[j]))))

    sorted_points.sort(reverse=True)
    return sorted_points

############################################################################################################

def getSampledHyperCube(step_size: float, dimension: int, outer_range: List[List[float]]=[[0, 1], [0, 1], [0, 1], [0, 1]]) -> npt.ArrayLike:
    """
    Get a hypercube sample of the space
    """
    g = np.meshgrid(*[np.arange(outer_range[i][0], outer_range[i][1] + 0.0000001, step_size) for i in range(dimension)])
    return np.array(list(zip(*(x.flat for x in g))))


def _get_refined_hypercube( metamer_led_weights: npt.ArrayLike, previous_step:float) -> npt.ArrayLike:
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
    return getSampledHyperCube(0.004, 4, outer_range)


def _get_top_metamer(M_weightToCone: npt.ArrayLike, hypercube: npt.ArrayLike, metameric_axis:int=2, prec:float =0.005, exponent:int=8)-> Union[npt.ArrayLike, npt.ArrayLike]:
    """
    Get the Top Metamer for a Given Transform
    Args:
        M_weightToCone (npt.ArrayLike): Transform from display weights to Cone Space
        hypercube (float): Point Cloud in the Display Space That We're Using to Sample
        metameric_axis (int): The axis that we are bucketizing against in Cone Space (default is 2 for SMQL for Q)
        prec (float): The precision of the bucketing (default is 0.005)
        exponent (int): The exponent for the bucketing (default is 8)
    """
    all_lms_intensities = (M_weightToCone@hypercube.T).T # multiply all possible led combinations with the intensities
    buckets = get_metamer_buckets(all_lms_intensities, axis=metameric_axis, prec=prec, exponent=exponent)
    random_index = 0
    dst, (metamer_1, metamer_2) = buckets[random_index]
    return metamer_1, metamer_2


def _refine_metamers(weights_1:npt.ArrayLike, weights_2:npt.ArrayLike, M_weightToCone:npt.ArrayLike, metameric_axis:int=2, hypercube_sample:float=0.01)-> Union[npt.ArrayLike, npt.ArrayLike]:
    """
    Refine the estimate of the metamers by sampling a smaller hypercube around the metamer
    Args:
        weights_1 (npt.ArrayLike): The weights of the first metamer
        weights_2 (npt.ArrayLike): The weights of the second metamer
        M_weightToCone (npt.ArrayLike): The matrix that converts the led weights to the cone space
        metameric_axis (int): The axis that we are looking for the metamers
        hypercube_sample (float): The step size for the hypercube
    """
    
    hypercube1 = _get_refined_hypercube(weights_1, hypercube_sample * 2)
    hypercube2 = _get_refined_hypercube(weights_2, hypercube_sample * 2)
    hypercube = np.vstack([hypercube1, hypercube2])
    return _get_top_metamer(M_weightToCone, hypercube, metameric_axis=metameric_axis, prec=0.0005, exponent=11)


# TODO: Modify method later in order to return a point close to a given direction in the cone space
def get_metamers(M_weightToCone: npt.ArrayLike, metameric_axis:int=2, hypercube_sample:float = 0.01) -> Union[npt.ArrayLike, npt.ArrayLike]:
    """
    Get the metamers for the given matrix and axis, and return the weights of the primaries for the metamers
    Args:
        M_weightToCone (npt.ArrayLike): The matrix that converts the led weights to the cone space
        metameric_axis (int): The axis that we are looking for the metamers
        hypercube_sample (float): The step size for the hypercube (default is 0.01)
    """
    invMat = np.linalg.inv(M_weightToCone)
    # TODO: @Tian this should be cached so we don't have to spend time computing this. Idk how u want your filesystem organized, so I'm leaving it to you.
    hypercube = getSampledHyperCube(hypercube_sample, 4) # takes 40 seconds at 0.01 step, fine if we only run it once
    met1, met2 = _get_top_metamer(M_weightToCone, hypercube, metameric_axis=metameric_axis)
    met1, met2 = _refine_metamers(invMat@ met1, invMat@ met2, M_weightToCone, metameric_axis=metameric_axis, hypercube_sample=hypercube_sample)
    return invMat@met1, invMat@met2