from typing import List, Tuple
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

from TetriumColor.Observer import Spectra


def load_primaries_from_csv(primaries_dir: str) -> List[Spectra]:
    """Load primaries from a csv file

    Args:
        primary_csv (str): path to the csv file

    Returns:
        List[Spectra]: list of Spectra objects representing the Primaries measured
    """

    return get_spectras_from_rgbo_list(primaries_dir, [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)])


def get_spectras_from_rgbo_list(
    directory: str, rgbo_list: List[Tuple[int, int, int, int]]
) -> List[Spectra]:
    """Given a list of (r, g, b, o) tuples, read the corresponding power data if available.

    Returns a list of power value lists in the same order as the input RGBO list.
    If a file is missing, the corresponding entry is None and a warning is printed.
    """
    wavelengths = np.arange(380, 781, 4)  # Assuming a fixed wavelength range
    results = []

    for rgbo in rgbo_list:
        r, g, b, o = rgbo
        filename = f"r{r}g{g}b{b}o{o}.csv"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            print(f"Warning: CSV file for {rgbo} not found at {filepath}")
            results.append(None)
            continue

        power_values = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header if present
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    power = float(row[1])
                    power_values.append(power)
                except ValueError:
                    continue  # Skip malformed rows

        results.append(Spectra(wavelengths=wavelengths, data=np.array(power_values)))

    return results


def get_rgbo_from_filename(filename: str) -> Tuple[int, int, int, int]:
    """Extract RGBO tuple from filename like r12g34b56o78.csv"""
    name = os.path.splitext(filename)[0]
    parts = name.strip("rgbop")
    r = int(name[name.index("r") + 1: name.index("g")])
    g = int(name[name.index("g") + 1: name.index("b")])
    b = int(name[name.index("b") + 1: name.index("o")])
    o = int(name[name.index("o") + 1:])
    return r, g, b, o


def compare_dataset_to_primaries(
    measurements_dir: str,
    rgbo_list: List[Tuple[int, int, int, int]],
    primary_spectra: List[Spectra],
    exclude_primaries: bool = True
) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    """
    Compares all spectra in a directory to their predicted linear combinations from primaries.

    Args:
        measurements_dir (str): Path to the directory containing measured CSVs
        primary_spectra (List[Spectra]): List of 4 Spectra objects for (R, G, B, O) at 255
        exclude_primaries (bool): Whether to exclude pure primaries from evaluation

    Returns:
        List[Tuple[RGBO, RMSE, diff_spectrum]]
    """
    spectrums = get_spectras_from_rgbo_list(measurements_dir, rgbo_list)
    results = []
    for rgbo, spectrum in zip(rgbo_list, spectrums):

        if exclude_primaries and rgbo in [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)]:
            continue

        scaling_factors = np.array(rgbo) / 255.0

        predicted = Spectra(wavelengths=primary_spectra[0].wavelengths, data=np.array(sum(
            scale * primary.data for scale, primary in zip(scaling_factors, primary_spectra)
        )))

        diff = spectrum.data - predicted.data

        rmse = np.sqrt(np.mean(diff ** 2))
        plot_measured_vs_predicted(rgbo, spectrum, predicted, rmse)
        results.append((rgbo, rmse, diff))

    return results


def plot_measured_vs_predicted(
    rgbo: Tuple[int, int, int, int],
    measured: Spectra,
    predicted: Spectra,
    rmse: float,
    save_path: str | None = None
):
    """
    Plot measured vs predicted spectra and their difference.

    Args:
        rgbo (Tuple): RGBO tuple
        measured (Spectra): Measured spectrum
        predicted (np.ndarray): Predicted spectrum from primaries
        rmse (float): Root mean square error
        save_path (str, optional): If given, save the plot to this path instead of showing
    """
    wavelengths = measured.wavelengths
    diff = measured.data - predicted.data

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, measured.data, label="Measured", color='blue')
    plt.plot(wavelengths, predicted.data, label="Predicted", color='orange')
    plt.plot(wavelengths, diff, label="Difference", color='red', linestyle='--')
    plt.title(f"RGBO {rgbo} - RMSE: {rmse:.4f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
