import numpy as np
import matplotlib.pyplot as plt
import tqdm

import numpy.typing as npt
from typing import List
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.models import RGB_COLOURSPACE_BT709
from colour import XYZ_to_RGB, wavelength_to_XYZ

from . import Observer, Cone, Spectra, MaxBasis, MaxBasisFactory
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform


def GetsRGBfromWavelength(wavelength):
    return XYZ_to_RGB(wavelength_to_XYZ(wavelength), "sRGB")


def GetCustomObserver(wavelengths: npt.NDArray,
                      od: float = 0.5,
                      dimension: int = 4,
                      s_cone_peak: float = 419,
                      m_cone_peak: float = 530,
                      q_cone_peak: float = 547,
                      l_cone_peak: float = 559,
                      macular: float = 1,
                      lens: float = 1,
                      template: str = "neitz",
                      illuminant: Spectra | None = None,
                      verbose: bool = False, subset: List[int] = [0, 1, 3]):
    """Given specific parameters, return an observer model with Q cone peaked at 547

    Args:
        wavelengths (_type_): Array of wavelengths
        od (float, optional): optical density of photopigment. Defaults to 0.5.
        m_cone_peak (int, optional): peak of the M-cone. Defaults to 530.
        l_cone_peak (int, optional): peak of the L-cone. Defaults to 560.
        template (str, optional): cone template function. Defaults to "neitz".

    Returns:
        _type_: Observer of specified paramters and 4 cone types
    """

    l_cone = Cone.cone(l_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    l_cone_555 = Cone.cone(555, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    l_cone_551 = Cone.cone(551, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    l_cone_547 = Cone.cone(547, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    q_cone = Cone.cone(q_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    m_cone = Cone.cone(m_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    s_cone = Cone.cone(s_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
    # s_cone = Cone.s_cone(wavelengths=wavelengths)
    if dimension == 3:
        set_cones = [s_cone, m_cone, q_cone, l_cone]
        return Observer([set_cones[i] for i in subset], verbose=verbose, illuminant=illuminant)
    elif dimension == 2:
        return Observer([s_cone, m_cone], verbose=verbose)
    elif dimension == 4:
        return Observer([s_cone, m_cone, q_cone, l_cone], verbose=verbose)
    elif dimension == 6:
        return Observer([s_cone, m_cone, l_cone_547, l_cone_551, l_cone_555, l_cone], verbose=verbose)
    else:
        raise NotImplementedError


def GetStockmanObserver(wavelengths: npt.NDArray):
    """Get the standard stockman & sharpe 2 degrees observer model

    Args:
        wavelengths (_type_): Array of wavelengths

    Returns:
        _type_: Stockman Observer Model
    """
    l_cone = Cone.l_cone(wavelengths=wavelengths)
    q_cone = Cone.q_cone(wavelengths=wavelengths)
    m_cone = Cone.m_cone(wavelengths=wavelengths)
    s_cone = Cone.s_cone(wavelengths=wavelengths)
    return Observer([s_cone, m_cone, q_cone, l_cone], verbose=False)


def GetSubsetIdentifyingObservers(peaks=((530, 559), (530, 555), (533, 559), (533, 555),
                                         (530, 551), (533, 551), (530, 552), (533, 552)),
                                  od=0.5,
                                  macular=[0.5, 1, ],
                                  lens=1,
                                  template='neitz',
                                  wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(380, 781, 1)

    all_observers = []
    i = 0
    for m_cone_peak, l_cone_peak in peaks:
        per_observer = []
        avg_observer = GetCustomObserver(wavelengths, od=0.5,
                                         m_cone_peak=m_cone_peak, l_cone_peak=l_cone_peak,
                                         macular=1, lens=1, template=template)
        per_observer.append(avg_observer)
        for od in [0.4, 0.5, 0.6]:
            for macular in [0.5, 0.75, 1.0, 1.5, 2.0]:  # 1.0 is standard, 4.0 is 1.2/0.35, which is the max peak
                for lens in [0.75, 1, 1.25]:  # vary 25% in young observers
                    per_observer.append(GetCustomObserver(wavelengths, od=od,
                                                          m_cone_peak=m_cone_peak, l_cone_peak=l_cone_peak,
                                                          macular=macular, lens=lens, template=template))
                    i += 1
        all_observers.append(per_observer)
    return all_observers, peaks


def GetPeakPrevalentObservers(peaks=((530, 559), (530, 555), (530, 551),  # (533, 547)
                                     #   ((530, 559), (530, 555), (533, 559), (533, 555),
                                     #  (530, 551), (533, 551), (530, 552), (533, 552)
                                     ),
                              od=0.5,
                              macular=1,
                              lens=1,
                              template='neitz',
                              wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(380, 781, 1)

    all_observers = []
    i = 0
    for m_cone_peak, l_cone_peak in peaks:
        all_observers.append(GetCustomObserver(wavelengths, od=od,
                                               m_cone_peak=m_cone_peak, l_cone_peak=l_cone_peak,
                                               macular=macular, lens=lens, template=template))
    return all_observers, peaks


def GetPrevalentObservers(peaks=((530, 559), (530, 555), (533, 559), (533, 555),
                                 (530, 551), (533, 551), (530, 552), (533, 552)),
                          od=0.5,
                          macular=[0.5, 1, ],
                          lens=1,
                          template='neitz',
                          wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(380, 781, 1)

    all_observers = []
    i = 0
    for m_cone_peak, l_cone_peak in peaks:
        per_observer = []
        avg_observer = GetCustomObserver(wavelengths, od=0.5,
                                         m_cone_peak=m_cone_peak, l_cone_peak=l_cone_peak,
                                         macular=1, lens=1, template=template)
        per_observer.append(avg_observer)
        for od in [0.4, 0.5, 0.6]:
            for macular in [0.5, 0.75, 1.0, 1.5, 2.0]:  # 1.0 is standard, 4.0 is 1.2/0.35, which is the max peak
                for lens in [0.75, 1, 1.25]:  # vary 25% in young observers
                    per_observer.append(GetCustomObserver(wavelengths, od=od,
                                                          m_cone_peak=m_cone_peak, l_cone_peak=l_cone_peak,
                                                          macular=macular, lens=lens, template=template))
                    i += 1
        all_observers.append(per_observer)
    return all_observers, peaks


def GetAllObservers(
        ods=[0.5, 0.6],
        peaks=((530, 559), (530, 555), (533, 559), (533, 555)),
        macular_pigment_density=[0.5, 1.0, 2.0],  # 1.0 is standard, 4.0 is 1.2/0.35, which is the max peak
        lens_density=[0.75, 1, 1.25],  # vary 25% in young observers
        template='neitz',
        wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(380, 781, 1)

    all_observers = []
    i = 0
    for od in ods:
        for m_cone_peak, l_cone_peak in peaks:
            for macular in macular_pigment_density:
                for lens in lens_density:
                    # with open("observer_parameters.txt", "a") as file:
                    #     file.write(
                    #         f"idx:{i} OD: {od}, M-Cone Peak: {m_cone_peak}, L-Cone Peak: {l_cone_peak}, Macular: {macular}, Lens: {lens}\n")
                    all_observers.append(GetCustomObserver(wavelengths, od=od,
                                                           m_cone_peak=m_cone_peak, l_cone_peak=l_cone_peak, macular=macular, lens=lens, template=template))
                    i += 1
    return all_observers


def GetConeToDisplay(observer: Observer, led_spectrums: List[Spectra]):
    primary_intensities = np.array(
        [observer.observe_normalized(s) for s in led_spectrums])
    mat_cone_to_primaries = np.linalg.pinv(primary_intensities)
    return mat_cone_to_primaries


def GetDisplayToCone(observer: Observer, led_spectrums: List[Spectra] | npt.NDArray):
    primary_intensities = np.array(
        [observer.observe_normalized(s) for s in led_spectrums])
    return primary_intensities


def GetParalleletopeBasis(observer: Observer, led_spectrums: List[Spectra]):
    disp = GetDisplayToCone(observer, led_spectrums)
    intensities = disp.T
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    return (intensities@np.diag(white_weights)).T


def GetConeTosRGBPrimaries(observer: Observer, metameric_axis: int = 2):
    if observer.dimension > 3:
        subset = [0, 1, 2]
    else:
        # really this only works for LMS subset so this is a bit stupid
        subset = [i for i in range(4) if i != metameric_axis]

    xyz_cmfs = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].values
    white_pt = np.diag(observer.get_whitepoint(wavelengths=observer.wavelengths)[subset])

    M_XYZ_to_RGB = RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB
    M_Cone_To_Primaries = M_XYZ_to_RGB@(xyz_cmfs.T@np.linalg.pinv(observer.sensor_matrix[subset]) @ white_pt / 100)

    if observer.dimension > 3:
        M_Cone_To_Primaries = np.insert(M_Cone_To_Primaries, metameric_axis, 0, axis=1)
    return M_Cone_To_Primaries


def GetColorSpaceTransformTosRGB(observer: Observer, metameric_axis: int = 2,
                                 subset_leds: List[int] = [0, 1, 2, 3]) -> ColorSpaceTransform:
    """ONLY WORKS FOR 3D OBSERVERS. ColorSpaceTransform Object for Observer to sRGB

    Args:
        observer (Observer): observer object
        metameric_axis (int, optional): axis along metamers. Defaults to 2.
        subset_leds (List[int], optional): subset_led. Defaults to [0, 1, 2, 3].

    Returns:
        ColorSpaceTransform: ColorSpaceTransform object
    """
    # ONLY WORKS FOR 3D in general because that transform is only defined for 3 dimensions
    max_basis = MaxBasisFactory.get_object(observer, verbose=False)

    M_Cone_To_Primaries = GetConeTosRGBPrimaries(observer, metameric_axis)

    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        subset_leds,
        np.ones(observer.dimension),
        M_Cone_To_Primaries
    )


def GetColorSpaceTransformWODisplay(observer: Observer, metameric_axis: int = 2) -> ColorSpaceTransform:
    """Given an observer and display primaries, return the ColorSpaceTransform

    Args:
        observer (Observer): Observer object
        metameric_axis: axis to be metameric over
    Returns:
         ColorSpaceTransform
    """
    max_basis = MaxBasisFactory.get_object(observer, verbose=False)

    M_Cone_To_Primaries = np.eye(observer.dimension)  # dummy matrix
    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        [i for i in range(observer.dimension)],  # dummy LEDs
        np.ones(observer.dimension),  # dummy white point
        GetConeTosRGBPrimaries(observer, metameric_axis)
    )


def GetColorSpaceTransform(observer: Observer, display_primaries: List[Spectra] | npt.NDArray,
                           scaling_factor: float = 1000, metameric_axis: int = 2,
                           subset_leds: List[int] = [0, 1, 2, 3]) -> ColorSpaceTransform:
    """Given an observer and display primaries, return the ColorSpaceTransform

    Args:
        observer (Observer): Observer object
        display_primaries (List[Spectra]): List of Spectra objects representing the display primaries
        scaling_factor (float, optional): factor to scale the display primaries by -- they are pretty low by default. Defaults to 1000.

    Returns:
        _type_: ColorSpaceTransform
    """
    max_basis = MaxBasisFactory.get_object(observer, verbose=False)
    disp = GetDisplayToCone(observer, display_primaries)

    intensities = disp.T * scaling_factor
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    rescaled_white_weights = white_weights / np.max(white_weights)
    new_intensities = intensities * white_weights

    M_Cone_To_Primaries = np.linalg.inv(new_intensities)  # something is fucked
    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        subset_leds,
        rescaled_white_weights,
        GetConeTosRGBPrimaries(observer, metameric_axis)
    )


def GetColorSpaceTransformsOverObservers(observers: List[Observer], primaries: List[Spectra],
                                         scaling_factor: float = 10000, metameric_axis: int = 2,
                                         subset_leds: List[int] = [0, 1, 2, 3]) -> List[ColorSpaceTransform]:
    """Get Color Space Transforms for a List of Observers, with a Fixed Primary Set.

    Args:
        observers (List[Observer]): List of observers.
        primaries (List[Spectra]): List of LED Spectra
        scaling_factor (float, optional): Scale the LEDs arbitarily. Defaults to 10000.
        metameric_axis (int, optional): Axis to be metameric over. Defaults to 2.
        subset_leds (List[int], optional): leds to use in a subset of 6. Defaults to [0, 1, 2, 3].

    Returns:
        List[ColorSpaceTransform]: List of ColorSpaceTransforms
    """
    transforms = []
    for observer in tqdm.tqdm(observers):
        transforms.append(GetColorSpaceTransform(observer, primaries, scaling_factor, metameric_axis, subset_leds))
    return transforms


def GetColorSpaceTransforms(observers: List[Observer], display_primaries: List[List[Spectra]],
                            scaling_factor: float = 10000, metameric_axis: int = 2,
                            subset_leds: List[int] = [0, 1, 2, 3]) -> List[List[ColorSpaceTransform]]:
    """Given a list of observers and display primary Spectra, return a List of associated ColorSpaceTransforms

    Args:
        observers (List[Observer]): List of Observer Objects
        primaries (List[Spectra]): List of Spectra objects representing the display primaries
        scaling_factor (float, optional): factor to scale the display primaries by -- they are pretty low by default. Defaults to 1000.

    Returns:
        List[List[ColorSpaceTransform]]: List of Lists of ColorSpaceTransforms indexed first by observer, and then by primary
    """
    # TODO: cache users
    transforms = []

    for observer in tqdm.tqdm(observers):
        per_observer = []
        max_basis = MaxBasisFactory.get_object(observer, verbose=False)

        for primaries in display_primaries:
            per_observer.append(GetColorSpaceTransform(observer, primaries,
                                scaling_factor, metameric_axis, subset_leds))
        transforms.append(per_observer)
    return transforms


def GetMaxBasisToDisplayTransform(color_space_transform: ColorSpaceTransform) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate from a 4x4 matrix that converts from RYGB to display primaries, a two 4x3 matrices
    that represent the conversion from max basis to display primaries in order to interface with Tetrium Paint Program

    Args:
        color_space_transform (ColorSpaceTransform): the colorspace transform to use.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: a tuple of 4x3 matrices that converts from max basis to display primaries
    """
    disp_weights = np.identity(4) * color_space_transform.white_weights
    # go from bgyr to rygb
    rygb_to_rgbo = disp_weights @ np.flip(color_space_transform.maxbasis_to_disp, axis=1)

    rygb_to_rgb = np.zeros((3, 4))
    rygb_to_rgb = rygb_to_rgbo[:3]

    rygb_to_ocv = np.zeros((3, 4))
    rygb_to_ocv[0] = rygb_to_rgbo[3:]

    return rygb_to_rgb.T, rygb_to_ocv.T


def GetRGBOCVToConeBasis(color_space_transform: ColorSpaceTransform) -> npt.NDArray:
    """Generate from a 4x4 matrix that converts from RYGB to display primaries, a two 4x3 matrices
    that represent the conversion from max basis to display primaries in order to interface with Tetrium Paint Program

    Args:
        color_space_transform (ColorSpaceTransform): the colorspace transform to use.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: a tuple of 4x3 matrices that converts from max basis to display primaries
    """
    disp_weights = np.identity(4) * color_space_transform.white_weights
    # go from bgyr to rygb
    rgbo_to_smql = np.linalg.inv(disp_weights @ color_space_transform.cone_to_disp)

    return rgbo_to_smql


def spectralDeliveryObserverSensitivity(observer: Observer, used_primaries: List[Spectra], measured_metamers: List[List[Spectra]], metamer_display_weights: npt.NDArray):
    """Given an array of used primaries, and multiple measured metamers, plot the predicted and measured metamersf

    Args:
        observer (Observer): observer model
        used_primaries (npt.NDArray): 4 primaries
        measured_metamers (npt.NDArray): measured metamers M x 2 x 4 where M is the number of metamer pairs
        metamer_display_weights (npt.NDArray): M x 2 x 4 array of weights for each primary in the metamer display
    """
    # plot all comparisons between the measurements

    smql_observed = np.zeros((len(metamer_display_weights), 3, 4))
    smql_predicted = np.zeros((len(metamer_display_weights), 3, 4))
    fig, axs = plt.subplots(len(metamer_display_weights), 2, figsize=(10, 5 * len(metamer_display_weights)))
    if len(axs.shape) < 2:
        axs = [axs]
    for i, weights in enumerate(metamer_display_weights):
        for k in range(2):

            data = np.sum([weights[k][j] * primary.data for j, primary in enumerate(used_primaries)], axis=0)
            predicted_metamer = Spectra(data=data,
                                        wavelengths=used_primaries[0].wavelengths)
            observed_metamer = measured_metamers[i][k]
            axs[i][k].plot(predicted_metamer.wavelengths, predicted_metamer.data, label='Predicted Metamer')
            axs[i][k].plot(observed_metamer.wavelengths, observed_metamer.data,
                           label='Measured Metamer', c=measured_metamers[i][k].to_rgb())
            axs[i][k].set_xlabel('Wavelength (nm)')
            axs[i][k].set_ylabel('Intensity')
            axs[i][k].legend()
            smql_observed[i][k] = observer.observe_normalized(observed_metamer)
            smql_predicted[i][k] = observer.observe_normalized(predicted_metamer)
        smql_observed[i][2] = np.abs(smql_observed[i][0] - smql_observed[i][1])
        smql_predicted[i][2] = np.abs(smql_predicted[i][0] - smql_predicted[i][1])

    print("For this observer:")
    print("Observed SMQL metamer 1, metamer 2, and difference:")
    print(smql_observed)
    print("Predicted SMQL metamer 1, metamer 2, and difference:")
    print(smql_predicted)

    plt.tight_layout()
    plt.show()
    return smql_observed, smql_predicted


def getDisplayedSpectraFromWeights(weights: npt.NDArray, primaries: List[Spectra]):
    weighted_spectra = np.sum([w * p.data for w, p in zip(weights, primaries)], axis=0)
    return Spectra(data=weighted_spectra, wavelengths=primaries[0].wavelengths)


def observerDisplaySensitivity(observer: Observer, weights: npt.NDArray, primaries: List[Spectra], perturbations: float = 1e-5):

    normalization = np.max([np.max(p.data) for p in primaries]) * 2  # give some headroom
    # perturb the primaries
    normalized_primaries = [Spectra(data=primary.data / normalization,
                                    wavelengths=primary.wavelengths) for primary in primaries]
    over_perturbed_primaries = [Spectra(data=(primary.data * (1 + perturbations/normalization)),
                                        wavelengths=primary.wavelengths) for primary in normalized_primaries]
    under_perturbed_primaries = [Spectra(data=np.clip(primary.data * (1 - perturbations/normalization), 0, 1),
                                         wavelengths=primary.wavelengths) for primary in normalized_primaries]

    current_primaries_to_cones = GetDisplayToCone(observer, normalized_primaries)
    over_current_to_cones = GetDisplayToCone(observer, over_perturbed_primaries)
    under_current_to_cones = GetDisplayToCone(observer, under_perturbed_primaries)

    actual = getDisplayedSpectraFromWeights(weights, normalized_primaries)
    perturbed_up = getDisplayedSpectraFromWeights(weights, over_perturbed_primaries)
    perturbed_down = getDisplayedSpectraFromWeights(weights, under_perturbed_primaries)

    estimated_metamers = [actual, perturbed_up, perturbed_down]

    fig = plt.figure()
    for i, metamer in enumerate(estimated_metamers):
        plt.plot(metamer.wavelengths, metamer.data, label=f"Metamer {i}", c=metamer.to_rgb())
    plt.show()
    cone_respones = [observer.observe_normalized(s) for s in estimated_metamers]

    over_white_pt = over_current_to_cones.sum(axis=0)
    white_pt = current_primaries_to_cones.sum(axis=0)
    under_white_pt = under_current_to_cones.sum(axis=0)

    # Plot the estimated metamers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metamer in zip(axes, estimated_metamers):
        rgb = metamer.to_rgb()
        ax.imshow([[rgb]])
        ax.axis('off')
    plt.show()

    print("White point over and under")
    print("Over White: ", over_white_pt)
    print("Actual: ", white_pt)
    print("Under White:", under_white_pt)

    print("White point differences over - actual")
    print(np.abs(over_white_pt - white_pt))
    print("White point differences under - actual")
    print(np.abs(under_white_pt - white_pt))
    return over_current_to_cones, under_current_to_cones
