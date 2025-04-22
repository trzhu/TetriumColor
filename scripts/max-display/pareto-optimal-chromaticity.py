

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from TetriumColor.Observer import *
from TetriumColor import *
import math
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


def compute_max_pareto_vol_efficiency(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray,
                                      idxs: npt.NDArray, spds: npt.NDArray, paretoPlot: bool | str = False):
    # compute total power needed to reach luminance
    spd_powers = np.array([np.trapz(spd) for spd in spds])
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
    # efficacies = np.clip(efficacies, 0, None)  # clip to avoid negative efficaciess
    weights = np.array(weights)
    efficacies[np.any(np.array(weights) < 0, axis=1)] = 0

    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)
    volumes[volumes < 0] = 0

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

    return best_idx, volumes[best_idx], efficacies[best_idx]


def compute_max_parallelotope(primary_candidates: npt.NDArray):
    volumes = np.array([np.linalg.det(p) for p in primary_candidates])
    best_idx = np.argmax(volumes)
    return best_idx, volumes[best_idx]


wavelengths = np.arange(400, 701, 5)
observer_wavelengths = np.arange(380, 781, 1)
observers = [
    Observer.custom_observer(observer_wavelengths, dimension=3),  # standard LMS observer
    #  Observer.custom_observer(observer_wavelengths, dimension=3, l_cone_peak=551),  # ben-like observer
    #  Observer.custom_observer(observer_wavelengths, dimension=3, l_cone_peak=547),  # Cda29's kid
    # most likely functional tetrachromatic observer
    Observer.custom_observer(observer_wavelengths, dimension=4, template='govardovskii'),
    #  Observer.custom_observer(observer_wavelengths, q_cone_peak=551, dimension=4),  # ben-like tetrachromatic observer
    #  Observer.custom_observer(observer_wavelengths, q_cone_peak=555, dimension=4)
]  # ser180ala like observer

# set of primaries - monochromatic, gaussian, or discrete

fwhm = AVG_FWHM
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
peak_wavelengths = np.arange(400, 701, 5)  # Peaks every 10nm from 380nm to 720nm

gaussian_primaries = [gaussian(wavelengths, peak, sigma) for peak in peak_wavelengths]
monochromatic_lights = [np.eye(1, len(wavelengths), np.abs(wavelengths - p).argmin()).flatten()
                        for p in peak_wavelengths]

primary_sets = [monochromatic_lights, gaussian_primaries]

# set of bases to project into from chromaticity
bases = [ColorSpaceType.HERING_CHROM, ColorSpaceType.CHROM]

save_dir = "./results/"
os.makedirs(save_dir, exist_ok=True)
results_dict = {}

# For all possible combinations of observers, primary sets, and bases
for observer in observers:
    corresponding_primaries = list(combinations(peak_wavelengths, observer.dimension))
    for basis in bases:
        for pset_idx, spds in enumerate(primary_sets):
            cs = ColorSpace(observer)
            observed_primaries = np.array(
                [observer.observe(Spectra(wavelengths=wavelengths, data=primary)) for primary in spds])
            peak_combinations = np.array(list(combinations(peak_wavelengths, observer.dimension)))

            sets_of_observed = np.array(list(combinations(observed_primaries, observer.dimension)))
            idxs = np.array(list(combinations(range(len(observed_primaries)), observer.dimension)))

            idx, volume, efficacy = compute_max_pareto_vol_efficiency(
                cs, basis, sets_of_observed, idxs, np.array(spds), paretoPlot=f"{save_dir}/{str(observer)}_{basis}_primary_set_{pset_idx}.png")

            max_primaries = list(sets_of_observed)[idx]
            corresponding_max_peaks = corresponding_primaries[idx]
            max_primaries = np.array(max_primaries)
            key = (observer, basis, tuple(map(tuple, spds)))
            results_dict[key] = {
                "max_volume": volume,
                "efficacy": efficacy,
                "max_primaries": max_primaries,
                "corresponding_max_peaks": corresponding_max_peaks
            }

            print(
                f"IDX: {idx}, Observer: {observer}, Basis: {basis}, Primary Set Idx: {pset_idx}")
            print(f"Max Volume: {volume}")
            print(f"Efficacy: {efficacy}")
            print(f"Corresponding Max Peaks: {corresponding_max_peaks}")

output_file = os.path.join(save_dir, "all_results.pkl")
with open(output_file, "wb") as f:
    pickle.dump(results_dict, f)
