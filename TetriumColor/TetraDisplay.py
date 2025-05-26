import matplotlib.pyplot as plt
import numpy as np
import math


from TetriumColor.Observer import *
from TetriumColor import *
from tqdm import tqdm

AVG_FWHM = 22.4


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_pareto_front(volumes: np.ndarray, efficiencies: np.ndarray):
    """
    Efficiently identifies the Pareto front for maximizing both volume and efficiency.

    Returns:
        pareto_indices: Indices of the Pareto-optimal points
        pareto_volumes: Volumes of the Pareto-optimal points
        pareto_efficiencies: Efficiencies of the Pareto-optimal points
    """
    data = np.stack([volumes, efficiencies], axis=1)

    # Sort by volume descending, then efficiency descending
    sorted_idx = np.lexsort((-efficiencies, -volumes))
    sorted_data = data[sorted_idx]

    # Scan for Pareto-optimal points
    pareto_idx = []
    max_eff = -np.inf
    for i, (v, e) in tqdm(enumerate(sorted_data)):
        if e > max_eff:
            pareto_idx.append(sorted_idx[i])
            max_eff = e

    pareto_idx = np.array(pareto_idx)
    return pareto_idx


def compute_efficiency(color_space: ColorSpace, primary_candidates: npt.NDArray, spds: List[Spectra]):
    # compute total power needed to reach luminance
    spd_powers = np.array([np.trapz(spd.data) for spd in spds])
    weights = []
    for p in primary_candidates:
        p = p.T
        try:
            w = np.linalg.solve(p, np.ones(color_space.dim))
            weights.append(w)
        except Exception as e:
            # Handle singular matrix case
            weights.append(np.ones(color_space.dim) * -1)
    efficacies = 1.0 / np.array([np.dot(w, spd_powers[idx])
                                 for w, idx in zip(weights, idxs)])  # power needed to reach luminance
    weights = np.array(weights)
    efficacies[np.any(np.array(weights) < 0, axis=1)] = 0
    return efficacies


def compute_max_chromatic_vol(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray):
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)
    volumes[volumes < 0] = 0
    return volumes


def compute_perceptual_volume(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray):
    """
    Computes the perceptual volume of a set of primary candidates in a given color space.

    Args:
        color_space (ColorSpace): The color space object.
        chrom_basis (ColorSpaceType): The chromaticity basis to project into.
        primary_candidates (npt.NDArray): The primary candidates.

    Returns:
        tuple: A tuple containing the index of the best candidate, the maximum volume, and the corresponding efficacy.
    """
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)
    volumes[volumes < 0] = 0
    return volumes


def compute_max_pareto_vol_efficiency(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray,
                                      idxs: npt.NDArray, spds: List[Spectra], paretoPlot: bool | str = False):
    efficacies = compute_efficiency(color_space, primary_candidates, spds)
    volumes = compute_max_chromatic_vol(color_space, chrom_basis, primary_candidates)

    # Normalize volumes and efficacies to [0, 1]
    v_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    e_norm = (efficacies - efficacies.min()) / (efficacies.max() - efficacies.min())
    distances = np.sqrt((1 - v_norm) ** 2 + (1 - e_norm) ** 2)
    best_idx = np.argmin(distances)

    pareto_idxs = get_pareto_front(v_norm, e_norm)

    if paretoPlot:
        plt.figure(figsize=(10, 6))
        plt.scatter(v_norm, e_norm, c=distances, cmap='viridis', label='Candidates')
        plt.plot(v_norm[pareto_idxs], e_norm[pareto_idxs], color='red', label='Pareto Front')
        plt.colorbar(label='Scores')
        plt.xlabel('Volume')
        plt.ylabel('Efficacy')
        plt.title('Pareto Front: Volume vs Efficacy')
        plt.grid(True)
        plt.legend()
        if isinstance(paretoPlot, str):
            plt.savefig(paretoPlot)
        else:
            plt.show()
        plt.close()

    return best_idx, volumes[best_idx], efficacies[best_idx]


def compute_max_parallelotope(primary_candidates: npt.NDArray):
    volumes = np.array([np.linalg.det(p) for p in primary_candidates])
    best_idx = np.argmax(volumes)
    return best_idx, volumes[best_idx]
