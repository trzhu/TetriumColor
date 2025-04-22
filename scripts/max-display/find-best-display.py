

import matplotlib.pyplot as plt
from re import I
import numpy as np

from TetriumColor.Observer import *
from TetriumColor import *
import pickle
import math
import pandas as pd
from scipy.optimize import lsq_linear


AVG_FWHM = 22.4


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def computeVolumes(points, cs, chrom_basis):
    chrom_points = cs.convert(points, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, cs.dim, cs.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(cs.dim)


def get_pareto_front(powers, volumes):
    points = list(zip(powers, volumes))
    pareto = []
    for i, (p_i, v_i) in enumerate(points):
        dominated = False
        for j, (p_j, v_j) in enumerate(points):
            if j != i and p_j <= p_i and v_j >= v_i and (p_j < p_i or v_j > v_i):
                dominated = True
                break
        if not dominated:
            pareto.append((p_i, v_i))
    return np.array(sorted(pareto))  # sorted by power for nice plotting


def computeMaxChromVolandEfficacy(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray,
                                  idxs: npt.NDArray, spds: npt.NDArray, peak_combinations: npt.NDArray, alpha=1.0):

    # compute total power needed to reach luminance
    spd_powers = np.array([np.trapz(spd) for spd in spds])
    weights = []
    for p in primary_candidates:
        p = p.T
        try:
            w = np.linalg.solve(p, np.ones(color_space.dim))
            # w = lsq_linear(p, np.ones(color_space.dim), bounds=(0, np.inf))
            # error = np.linalg.norm(p @ w.x - np.ones(color_space.dim))
            # if error < 0.5:
            # weights.append(np.ones(color_space.dim) * -1)
            # else:
            weights.append(w)
        except Exception as e:
            # Handle singular matrix case
            weights.append(np.ones(color_space.dim) * -1)
    efficacies = 1.0 / np.array([np.dot(w, spd_powers[idx])
                                 for w, idx in zip(weights, idxs)])  # power needed to reach luminance
    # efficacies = np.clip(efficacies, 0, None)  # clip to avoid negative efficaciess
    weights = np.array(weights)
    actual_candidates = weights[~np.any(np.array(weights) < 0, axis=1)]
    actual_idxs = idxs[~np.any(np.array(weights) < 0, axis=1)]
    peak_combinations_valid = peak_combinations[~np.any(np.array(weights) < 0, axis=1)]
    print(
        f"Peak combinations valid: {len(peak_combinations_valid)}/{len(primary_candidates)} = {len(peak_combinations_valid)/len(primary_candidates)}")
    efficacies[np.any(np.array(weights) < 0, axis=1)] = 0

    # efficacies = color_space.compute_efficiacies_per_primary(primary_candidates, ColorSpaceType.CONE)
    # efficacies = efficacies.mean(axis=1)  # or compute the min efficiency
    # efficacies = efficacies.min(axis=1)
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)

    print("Max / Min Volume: ", volumes.max(), volumes.min())
    print("Max / Min Efficacy: ", efficacies.max(), efficacies.min())

    v_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    e_norm = (efficacies - efficacies.min()) / (efficacies.max() - efficacies.min())
    # filtered_e_norm = e_norm[e_norm < 0.01 and (e_norm > 0)]
    # e_norm = filtered_e_norm / filtered_e_norm.max()
    # import pdb
    # pdb.set_trace()

    pareto = get_pareto_front(np.log(e_norm), np.log(v_norm))
    print(pareto)
    plt.plot(pareto[:, 0], pareto[:, 1], color='red', label='Pareto Front')
    plt.show()

    scores = alpha * v_norm + (1.0 - alpha) * (e_norm if alpha < 1.0 else 0)

    # # Plot the Pareto front
    plt.figure(figsize=(10, 6))
    plt.scatter(volumes, efficacies, c=scores, cmap='viridis', label='Candidates')
    plt.colorbar(label='Scores')
    plt.xlabel('Volume')
    plt.ylabel('Efficacy')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Pareto Front: Volume vs Efficacy')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Select the best
    idx = np.argmax(scores)
    return idx, volumes[idx], efficacies[idx]


def computeMaxChromGamut(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray,
                         idxs: npt.NDArray, spds: npt.NDArray, alpha=1.0):

    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)

    v_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())

    # Select the best
    idx = np.argmax(v_norm)
    return idx, volumes[idx], None


def compute3DSomething(color_space: ColorSpace, sets_of_primaries, alpha=0.5):
    efficacies = color_space.compute_efficiacies_per_primary(sets_of_primaries, ColorSpaceType.CONE)
    # efficacies = efficacies.mean(axis=1)  # or compute the min efficiency
    efficacies = efficacies.min(axis=1)
    volumes = np.array([np.linalg.det(np.array(p)) for p in sets_of_primaries])

    v_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    e_norm = (efficacies - efficacies.min()) / (efficacies.max() - efficacies.min())
    scores = alpha * v_norm + (1 - alpha) * e_norm

    # Select the best
    idx = np.argmax(scores)
    return idx, volumes[idx], efficacies[idx]


# given observers
wavelengths = np.arange(400, 701, 5)
observer_wavelengths = np.arange(380, 781, 1)
observers = [
    # Observer.custom_observer(observer_wavelengths, dimension=3),  # standard LMS observer
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
# Generate monochromatic lights from 380nm to 720nm
monochromatic_lights = [np.eye(1, len(wavelengths), np.abs(wavelengths - p).argmin()).flatten()
                        for p in peak_wavelengths]

primary_sets = [monochromatic_lights]  # , gaussian_primaries]

# set of possible perceptual functions
denoms = [1, 2.43, 3]

# set of Basis
bases = [ColorSpaceType.HERING_CHROM]  # , ColorSpaceType.CHROM, ColorSpaceType.CONE, ColorSpaceType.MAXBASIS]

results_dict = {}
# compute routine
denom = 1
alpha = 0.5
for observer in observers:
    corresponding_primaries = list(combinations(peak_wavelengths, observer.dimension))
    for basis in bases:
        # for denom in denoms:
        for pset_idx, spds in enumerate(primary_sets):
            cs = ColorSpace(observer)
            observed_primaries = np.array(
                [observer.observe(Spectra(wavelengths=wavelengths, data=primary)) for primary in spds])
            # observed_primaries = np.array([primary/primary.sum() for primary in observed_primaries])
            peak_combinations = np.array(list(combinations(peak_wavelengths, observer.dimension)))

            sets_of_observed = np.array(list(combinations(observed_primaries, observer.dimension)))
            idxs = np.array(list(combinations(range(len(observed_primaries)), observer.dimension)))

            idx, volume, efficacy = computeMaxChromVolandEfficacy(
                cs, basis, sets_of_observed, idxs, np.array(spds), peak_combinations, np.round(alpha, 1))

            max_primaries = list(sets_of_observed)[idx]
            corresponding_max_peaks = corresponding_primaries[idx]
            max_primaries = np.array(max_primaries)
            key = (observer, basis, alpha, denom, tuple(map(tuple, spds)))
            results_dict[key] = {
                "max_volume": volume,
                "efficacy": efficacy,
                "max_primaries": max_primaries,
                "corresponding_max_peaks": corresponding_max_peaks
            }

            print(f"Observer: {observer}, Basis: {basis}, Alpha: {alpha}, Denom: {denom}, Primary Set Idx: {pset_idx}")
            print(f"Max Volume: {volume}")
            print(f"Efficacy: {efficacy}")
            print(f"Max Primaries: {max_primaries}")
            print(f"Corresponding Max Peaks: {corresponding_max_peaks}")

# Save results_dict to a file
output_file = "/Users/jessicalee/Projects/generalized-colorimetry/code/TetriumColor/scripts/max-display/results_dict.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results_dict, f)

# print(f"Results saved to {output_file}")
