import pandas as pd
import numpy as np
import json, os, math
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import lsq_linear


# dude i couldnt get it to find TetriumColor.Observer so I guess I have to do this
# even messing with .env and .vscode/settings.json???? it just didnt want to work?? am i stupid?
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# from TetriumColor.Observer import Spectra, Illuminant, Observer
from TetriumColor.Observer import Spectra

def load_reflectance_data():
    """
    returns list of wavelengths and dictionary like 
    pigment_spectra["pigment name"][concentration] = spectra
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "oilpaints.json")
    
    with open(json_path, "r") as f:
        loaded_data = json.load(f)
    
    wavelengths = np.array(loaded_data["wavelengths"])
    pigment_data = loaded_data["pigments, concentrations, reflectances"]
    
    # create Spectra objects from oil paint data
    pigment_spectra = defaultdict(dict)
    for pigment, concentration, reflectance in pigment_data:  
        # e.g. of an entry in pigment_data
        # ["pigment name", concentration, [reflectance at 380nm, reflectance at 390nm, ... reflectance at 730 nm]]    
        reflectance = np.array(reflectance)
        spec = Spectra(wavelengths=wavelengths, data=reflectance)
        pigment_spectra[pigment][concentration] = spec
    
    return wavelengths, pigment_spectra

def visualize(name: str, spec: Spectra):
    rgb = spec.to_rgb()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot spectrum
    spec.plot(name="Spectrum", ax=ax[0])

    # Plot color swatch
    ax[1].imshow([[rgb]])
    ax[1].axis('off')
    ax[1].set_title("Swatch")
    plt.tight_layout()
    plt.legend()
    plt.title(name)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.show()

# reminder to self: the left side is short wavelengths/purple and the right side is long wavelengths/red
def plot_all_pigments(pigment_data: dict):
    # Number of pigments
    pigments = list(pigment_data.keys())
    num_pigments = len(pigments)

    # Set up subplot grid (e.g., 3 columns)
    cols = 2
    rows = num_pigments
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.5 * rows), constrained_layout=True)

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Plot each pigment
    for i, pigment in enumerate(pigments):
        ax_plot = axes[i, 0]
        rgb = pigment_data[pigment][100].to_rgb()
        
        # plot spectrum
        spec = pigment_data[pigment][100]
        spec.plot(name=pigment, ax=ax_plot, normalize=True, color = rgb)
        ax_plot.set_title(pigment)
        ax_plot.set_xlabel("Wavelength")
        ax_plot.set_ylabel("Reflectance")
        ax_plot.grid(True)

        # Plot swatch
        ax_swatch = axes[i, 1]
        ax_swatch.imshow([[rgb]])
        ax_swatch.axis("off")
        ax_swatch.set_title("Swatch")
    
    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "oilpaint_swatches.png")
    plt.savefig(path, dpi=300)
    plt.close()
    
    print("saved swatches image")

# returns K/S = (1-R)^2 / 2R for every lambda
def KS_ratio(spec: Spectra) -> np.ndarray:
    reflectance = spec.data
    return ((1-reflectance)**2) / (2 * reflectance)

# returns reflectance predicted by KM
# as an array like the wavelength arrays
# R_bg = reflectance of background substrate
# I want to be able to pass in an arbitrary number of pigments 
# and their weights
def mixture_KM_reflectance(weights: np.array, K_pigments: np.array, S_pigments: np.array, d=1.0, R_bg=1.0) -> np.array:
    if not len(weights) == len(K_pigments) == len(S_pigments):
        print("number of weights should be the same as number of pigments")
        return
    
    # normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # literally just a linear combination i love kubelka munk theory
    K_mix = K_pigments @ weights
    S_mix = S_pigments @ weights
    
    Q = K_mix / S_mix
    # let a = K/S, then R_infinity = 1 + a - sqrt((1+a)^2 - 1)
    R_inf = 1 + Q - np.sqrt((1+Q)**2 - 1)
    return R_inf

def solve_KS(Q_array: list, c_array: list):
    """
    Solves for K_w, S_w, K_p, S_p at a single wavelength using linear least squares.
    
    Q_array: array of Q values for each concentration (length n)
    c_array: array of concentrations as decimals (length n)
    
    Returns: (K_w, S_w, K_p, S_p)
    """
    A = []
    for Q, c in zip(Q_array, c_array):
        A.append([
            -(1 - c),         # coefficient of K_w
             Q * (1 - c),     # coefficient of S_w
            -c,               # coefficient of K_p
             Q * c            # coefficient of S_p
        ])
    A = np.array(A)
    b = np.zeros_like(Q_array)
    # print(f"A {A}")
    
    # Solve Ax = b in least squares sense
    bounds = (1.0e-16, np.inf)
    result = lsq_linear(A, b, bounds=bounds)
    # print(f"K_w, S_w, K_p, S_p = {result.x}")
    return result.x

def Q_to_R(Q):
    return (1 + Q - np.sqrt(Q**2 + 2 * Q)) / (1 + Q + np.sqrt(Q**2 + 2 * Q))


def fit_KS_all_wavelengths(refls_by_c: dict, wavelengths: np.ndarray):
    concentrations = sorted(refls_by_c.keys())
    c_array = np.array(concentrations) / 100  # Convert to 0-1

    # Initialize arrays
    K_w_spectrum = []
    S_w_spectrum = []
    K_p_spectrum = []
    S_p_spectrum = []

    for i in range(len(wavelengths)):
        # Reflectance values at this wavelength
        R_array = np.array([refls_by_c[c][i] for c in concentrations])
        R_clipped = np.clip(R_array, 0.01, 0.99)
        Q_array = ((1 - R_clipped) ** 2) / (2 * R_clipped)

        K_w, S_w, K_p, S_p = solve_KS(Q_array, c_array)
        K_w_spectrum.append(K_w)
        S_w_spectrum.append(S_w)
        K_p_spectrum.append(K_p)
        S_p_spectrum.append(S_p)

    return (
        np.array(K_w_spectrum),
        np.array(S_w_spectrum),
        np.array(K_p_spectrum),
        np.array(S_p_spectrum),
    )

def plot_values(Ks: list, Ss: list, pigment: str, wavelengths: list):
    if not len(Ks) == len(Ss) == len(wavelengths):
        print(f"len(Ks), Ss = {len(Ks)}, len(Ss) = {len(Ss)}")
        print(f"pigment is {pigment}")
        return
    i = range(len(wavelengths))
    plt.plot(wavelengths[i], Ks, marker='o', label = 'K')
    plt.plot(wavelengths[i], Ss, marker='x', label = 'S')
    plt.xlabel('wavelength')
    plt.ylabel('Value of K/S')
    plt.title(pigment)
    plt.grid(True)
    plt.show()
    
def plot_computed_KS(computed_KS: dict, wavelengths: np.array):
    n_pigments = len(computed_KS)
    fig, axes = plt.subplots(n_pigments, 2, figsize=(10, 3 * n_pigments), sharex=True)

    if n_pigments == 1:
        axes = [axes]  # handle the single-row edge case

    for i, (pigment, data) in enumerate(computed_KS.items()):
        ax_left = axes[i][0]
        ax_right = axes[i][1]

        # Left: K_p and S_p
        ax_left.plot(wavelengths, data["K_p"], label="K_p", color="blue")
        ax_left.plot(wavelengths, data["S_p"], label="S_p", color="green")
        ax_left.set_title(f"{pigment} - Pigment Coefficients")
        ax_left.legend()
        ax_left.set_ylabel("Value")

        # Right: K_w and S_w as derived via this pigment
        ax_right.plot(wavelengths, data["K_w"], label="K_w", color="red")
        ax_right.plot(wavelengths, data["S_w"], label="S_w", color="orange")
        ax_right.set_title(f"White Coefficients (via {pigment})")
        ax_right.legend()

    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (nm)")

    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "KS_plots.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("saved plots")
    
def main():
    print("it started")
    wvls, pigment_spectra = load_reflectance_data()
    # if want smaller steps maybe do something like e.g.
    # spec.interpolate(385, 395, )etc
    
    # plot_all_pigments(pigment_spectra)
    
    # K/S for titanium white 
    Q_white = KS_ratio(pigment_spectra["titanium white"][100])
    
    # compute K/S ratio at each wavelength for each mixture we have :)
    KS_ratios = defaultdict(dict)
    for p in pigment_spectra.keys():
        if p == "titanium white":
            KS_ratios[p][100] = Q_white
            continue
        # for everything except white, its 0% reflectance is just titanium white
        pigment_spectra[p][0] = pigment_spectra["titanium white"][100]
        for c in pigment_spectra[p]:
            if c == 0:
                KS_ratios[p][0] = Q_white
                continue
            KS_ratios[p][c] = KS_ratio(pigment_spectra[p][c])
    
    # dictionary that will store values for K_w, S_w, K_p, S_p as computed by each pigment's data
    computed_values = {}
    
    for p in KS_ratios.keys():
        if p == "titanium white":
            continue
        # K_w, S_w, K_p, S_p as computed for this pigment across each wavelength
        computed_K_ws = []
        computed_S_ws = []
        computed_K_ps = []
        computed_S_ps = []
        for i in range(len(wvls)):
            Q_vals = []
            c_vals = []
            for c in [10 * i for i in range(11)]:
                Q = KS_ratios[p][c][i]  # K/S at this wavelength and concentration
                Q_vals.append(Q)
                c_vals.append(c / 100)  # convert to [0, 1] range

            K_w, S_w, K_p, S_p = solve_KS(np.array(Q_vals), np.array(c_vals))
            
            computed_K_ws.append(K_w)
            computed_S_ws.append(S_w)
            computed_K_ps.append(K_p)
            computed_S_ps.append(S_p)
        
        computed_values[p] = {
        "K_w": np.array(computed_K_ws),
        "S_w": np.array(computed_S_ws),
        "K_p": np.array(computed_K_ps),
        "S_p": np.array(computed_S_ps), 
        }
        
    plot_computed_KS(computed_values, wvls)
    
    
# btw FOR SOME REASON, in Artist_paint_spectra.xlsx phthalo blue is spelled pthalo blue

if __name__ == "__main__":
    main()
