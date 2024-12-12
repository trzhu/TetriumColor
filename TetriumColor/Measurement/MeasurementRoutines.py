import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from typing import List
import numpy.typing as npt

from .MeasurementGUI import MeasureDisplay
from ..Observer import Spectra, convert_refs_to_spectras
from .PR650 import PR650
from scipy.optimize import curve_fit


def RemoveBlackLevelFromPrimaries(primaries: List[Spectra], summed_spectra: Spectra, subset=[0, 1, 2, 3]) -> List[Spectra]:
    subset_primaries = [primaries[i] for i in subset]
    summed_primaries = Spectra(data=np.sum(
        [primary.data for primary in subset_primaries], axis=0), wavelengths=primaries[0].wavelengths)
    black_level = Spectra(data=(summed_primaries.data - summed_spectra.data)/-(len(subset_primaries)-1),
                          wavelengths=summed_primaries.wavelengths)

    black_level_adjusted_spectras = [Spectra(data=s.data - black_level.data,
                                             wavelengths=s.wavelengths) for s in primaries]
    return black_level_adjusted_spectras


def SaveRGBOtoSixChannel(spectras: List[Spectra], save_filename: str):
    reformatted_spectras = np.array([spectras[0].wavelengths] + [spectras[i].data
                                                                 for i in range(len(spectras))]).T  # wavelengths + 4 spectra
    modified_spectras = np.insert(arr=reformatted_spectras, obj=[5, 5], values=0, axis=1)
    np.savetxt(save_filename, modified_spectras, delimiter=',', header='Wavelength,R,G,B,O,C,V')


def MeasurePrimaries(save_directory: str):

    # currently need to manually measure all of the primaries + all of them together.
    # fine for now, but would be nice to get rid of it by interfacing with Tetrium
    # load the pr650
    # only works on Jessica's Mac rn, can't be bothered
    mac_port_name = '/dev/cu.usbserial-A104D0XS'
    pr650 = PR650(mac_port_name)

    # Create the app and manually measure all of the primaries
    app = MeasureDisplay(pr650, save_directory=save_directory)
    app.run()

    spectras = np.load("../../measurements/2024-12-06/primaries/spectras.npy")

    wavelengths = np.arange(380, 781, 4)
    primaries = convert_refs_to_spectras(spectras[:, 1, :], wavelengths)

    black_adjusted_primaries = RemoveBlackLevelFromPrimaries(primaries[:4], primaries[4], subset=[0, 1, 2])
    SaveRGBOtoSixChannel(black_adjusted_primaries, os.path.join(save_directory, 'all_primaries.csv'))

    with open(os.path.join(save_directory, 'black_adjusted_primaries.pkl'), 'wb') as f:
        pickle.dump(black_adjusted_primaries, f)


def MeasureMetamers(metamer_display_weights: npt.NDArray, save_directory: str, primary_directory: str):
    # currently need to manually measure all of the primaries + all of them together.
    # fine for now, but would be nice to get rid of it by interfacing with Tetrium
    # load the pr650
    # only works on Jessica's Mac rn, can't be bothered
    mac_port_name = '/dev/cu.usbserial-A104D0XS'
    pr650 = PR650(mac_port_name)

    # Create the app and manually measure all of the primaries
    app = MeasureDisplay(pr650, save_directory='tmp')
    app.run()

    wavelengths = np.arange(380, 781, 4)
    metamers = convert_refs_to_spectras(np.array(app.spectras), wavelengths)

    with open(os.path.join(save_directory, 'metamers.pkl'), 'wb') as f:
        pickle.dump(metamers, f)

    with open(os.path.join(save_directory, "metamer_display_weights.pkl"), 'wb') as f:
        pickle.dump(metamer_display_weights, f)

    with open(os.path.join(primary_directory, 'black_adjusted_primaries.pkl'), 'rb') as f:
        primaries = pickle.load(f)

    return metamer_display_weights, metamers, primaries


def LoadMetamers(metamer_save_directory: str, primary_directory: str) -> tuple[npt.NDArray, List[Spectra], List[Spectra]]:
    """load the measured metamers and primaries

    Args:
        metamer_save_directory (str): metamer_save_directory
        primary_directory (str): primary directory saved

    Returns:
        _type_: return the metamer display weights, the metamers, and the primaries as Spectra
    """
    with open(os.path.join(metamer_save_directory, 'metamers.pkl'), 'rb') as f:
        metamers = pickle.load(f)

    with open(os.path.join(primary_directory, 'black_adjusted_primaries.pkl'), 'rb') as f:
        primaries = pickle.load(f)

    with open(os.path.join(metamer_save_directory, "metamer_display_weights.pkl"), 'rb') as f:
        metamer_display_weights = pickle.load(f)

    return metamer_display_weights, metamers, primaries


def LoadPrimaries(primary_directory: str) -> List[Spectra]:
    """Load primaries from a directory

    Args:
        primary_directory (str): directory to load the primaries from

    Returns:
        List[Spectra]: list of Spectra objects representing the Primaries measured
    """
    with open(os.path.join(primary_directory, 'black_adjusted_primaries.pkl'), 'rb') as f:
        primaries = pickle.load(f)
    return primaries


def GaussianSmoothPrimaries(primaries: List[Spectra]) -> List[Spectra]:
    """ Smooth the primaries by fitting a better curve to the data, and using that as the real primaries

    Args:
        primaries (List[Spectra]): measured primaries
        smoothing_factor (float, optional): . Defaults to 0.01.

    Returns:
        List[Spectra]: _description_
    """
    # want to upsample to at least 1 nm resolution because 4nm doesn't sample finely enough for the narrowband imo. We can tell with the red

    def generalized_normal(x, amp, cen, alpha, beta):
        return amp * np.exp(-((np.abs(x - cen) / alpha) ** beta))

    smoothed_primaries = []

    data_per_peak = [[630, 1], [530, 2.5], [450, 2], [590, 2]]
    upsampled_wavelengths = np.arange(380, 781, 1)

    for primary, data in zip(primaries, data_per_peak):
        popt, _ = curve_fit(generalized_normal, primary.wavelengths,
                            primary.data, p0=[np.max(primary.data), data[0], 1, data[1]])
        fitted_data = generalized_normal(upsampled_wavelengths, *popt)
        smoothed_primaries.append(Spectra(data=fitted_data, wavelengths=upsampled_wavelengths))

    return smoothed_primaries


def PerturbPrimaries(primaries: List[Spectra], wavelength_pertubation: int = 1, intensity_pertubation: float = 0.05) -> List[List[Spectra]]:
    """Perturb the primaries to find better models of the display primaries

    Args:
        primaries (List[Spectra]): measured primaries

    Returns:
        List[List[Spectra]]: list of color space transforms for each observer
    """

    perturbed_down_list = []
    perturbed_up_list = []
    shifted_left = []
    shifted_right = []
    for primary in primaries:
        perturbed_up_list.append(Spectra(data=np.clip(primary.data + (primary.data * intensity_pertubation),
                                                      0, None), wavelengths=primary.wavelengths))
        perturbed_down_list.append(Spectra(data=np.clip(primary.data - (primary.data * intensity_pertubation),
                                                        0, None), wavelengths=primary.wavelengths))
        shifted_left.append(Spectra(data=primary.data, wavelengths=primary.wavelengths + wavelength_pertubation))
        shifted_right.append(Spectra(data=primary.data, wavelengths=primary.wavelengths - wavelength_pertubation))

    return [perturbed_up_list, perturbed_down_list, shifted_left, shifted_right]


def PerturbSinglePrimary(primary_idx: int, primaries: List[Spectra], wavelength_pertubation: int = 1, intensity_pertubation: float = 0.05) -> List[List[Spectra]]:
    """Perturb the primaries to find better models of the display primaries

    Args:
        primaries (List[Spectra]): measured primaries

    Returns:
        List[List[Spectra]]: list of color space transforms for each observer
    """
    pertubations = []
    primary = primaries[primary_idx]
    pertubations.append(Spectra(data=np.clip(primary.data + (primary.data * intensity_pertubation),
                                             0, None), wavelengths=primary.wavelengths))
    pertubations.append(Spectra(data=np.clip(primary.data - (primary.data * intensity_pertubation),
                                             0, None), wavelengths=primary.wavelengths))
    pertubations.append(Spectra(data=primary.data, wavelengths=primary.wavelengths + wavelength_pertubation))
    pertubations.append(Spectra(data=primary.data, wavelengths=primary.wavelengths - wavelength_pertubation))

    final_primaries = []
    for perturbation in pertubations:
        final_primaries += [[perturbation if i == primary_idx else p for i, p in enumerate(primaries)]]
    return final_primaries


def PerturbWavelengthPrimaries(primary, wavelength_pertubation_range: List[int] = [1, 2, 3, 4, 5]) -> List[Spectra]:
    """Perturb the primaries to find better models of the display primaries

    Args:
        primaries (List[Spectra]): measured primaries

    Returns:
        List[List[Spectra]]: list of color space transforms for each observer
    """

    shifted_left = []
    for wavelength_pertubation in wavelength_pertubation_range:
        shifted_left.append(Spectra(data=primary.data, wavelengths=primary.wavelengths + wavelength_pertubation))

    return shifted_left
