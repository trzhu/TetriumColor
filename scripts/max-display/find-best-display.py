

import numpy as np

from TetriumColor.Observer import *
from TetriumColor import *
import pickle

AVG_FWHM = 22.4


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# given observers
wavelengths = np.arange(380, 721, 5)
observers = [Observer.custom_observer(wavelengths, dimension=3),  # standard LMS observer
             #  Observer.custom_observer(wavelengths, dimension=3, l_cone_peak=551),  # ben-like observer
             #  Observer.custom_observer(wavelengths, dimension=3, l_cone_peak=547),  # Cda29's kid
             Observer.custom_observer(wavelengths, dimension=4),  # most likely functional tetrachromatic observer
             #  Observer.custom_observer(wavelengths, q_cone_peak=551, dimension=4),  # ben-like tetrachromatic observer
             #  Observer.custom_observer(wavelengths, q_cone_peak=555, dimension=4)
             ]  # ser180ala like observer

# set of primaries - monochromatic, gaussian, or discrete

fwhm = AVG_FWHM
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
peak_wavelengths = np.arange(380, 721, 5)  # Peaks every 10nm from 380nm to 720nm

gaussian_primaries = [gaussian(wavelengths, peak, sigma) for peak in peak_wavelengths]
# Generate monochromatic lights from 380nm to 720nm
monochromatic_lights = [np.eye(1, len(wavelengths), np.abs(wavelengths - p).argmin()).flatten()
                        for p in peak_wavelengths]

primary_sets = [monochromatic_lights, gaussian_primaries]

# set of possible perceptual functions
denoms = [1, 2.43, 3]

# set of Basis
bases = [ColorSpaceType.CONE, ColorSpaceType.MAXBASIS]

results_dict = {}
# compute routine
for observer in observers:
    corresponding_primaries = list(combinations(peak_wavelengths, observer.dimension))
    for basis in bases:
        for denom in denoms:
            for pset_idx, primary_set in enumerate(primary_sets):
                cs = ColorSpace(observer)
                primaries = np.array([observer.observe(primary) for primary in primary_set])
                perceptual_primaries = cs.convert_to_perceptual(primaries, ColorSpaceType.CONE, basis, denom)

                sets_of_dimension = np.array(list(combinations(perceptual_primaries, observer.dimension)))

                adjusted_primaries = []
                adjusted_idxs = []
                for i, p in enumerate(sets_of_dimension):
                    try:
                        adjusted_primaries += [p * np.linalg.solve(p, np.ones(observer.dimension))]
                        adjusted_idxs += [i]
                    except np.linalg.LinAlgError:
                        continue

                result = list(map(np.linalg.det, adjusted_primaries))
                idx = np.argmax(result)
                real_idx = adjusted_idxs[idx]
                max_primaries = list(sets_of_dimension)[real_idx]
                corresponding_max_peaks = corresponding_primaries[real_idx]
                max_primaries = np.array(max_primaries)
                key = (observer, basis, denom, tuple(map(tuple, primary_set)))
                results_dict[key] = {
                    "max_volume": result[idx],
                    "max_primaries": max_primaries,
                    "corresponding_max_peaks": corresponding_max_peaks
                }

                print(f"Observer: {observer}, Basis: {basis}, Denom: {denom}, Primary Set Idx: {pset_idx}")
                print(f"Max Volume: {result[idx]}")
                print(f"Max Primaries: {max_primaries}")
                print(f"Corresponding Max Peaks: {corresponding_max_peaks}")


# Save results_dict to a file
output_file = "/Users/jessicalee/Projects/generalized-colorimetry/code/TetriumColor/scripts/max-display/results_dict.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results_dict, f)

# print(f"Results saved to {output_file}")
