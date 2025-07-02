import pandas as pd
import numpy as np
import json, os, math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.optimize import lsq_linear


# dude i couldnt get it to find TetriumColor.Observer so I guess I have to do this
# even messing with .env and .vscode/settings.json???? it just didnt want to work?? am i stupid?
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from TetriumColor.Observer import Spectra, Inks

img_dpi = 48 # make smaller or larger for desired size of pngs

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

# reminder to self: the left side is short wavelengths/purple and the right side is long wavelengths/red
def plot_all_pigments(pigment_data: dict):
    # Number of pigments
    pigments = list(pigment_data.keys())
    num_pigments = len(pigments)

    # Set up subplot grid
    cols = 2
    rows = num_pigments
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.5 * rows), constrained_layout=True)

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
    plt.savefig(path, dpi=img_dpi)
    plt.close()
    
    print("saved swatches image")

# returns K/S = (1-R)^2 / 2R for every lambda
def KS_ratio(spec: Spectra) -> np.ndarray:
    reflectance = spec.data
    return ((1-reflectance)**2) / (2 * reflectance)

# returns reflectance predicted by KM
# as an array like the wavelength arrays
# R_bg = reflectance of background substrate
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

def solve_KS(Q_array: list, c_array: list, K_w):
    """
    Solves for K_p, S_p at a single wavelength using linear least squares.
    
    Q_array: array of Q values for each concentration (length n)
    c_array: array of concentrations as decimals (length n)
    
    Returns: (K_p, S_p)
    """
    A = []
    b = []
    # could add -K + Q*S = 0 equations but when I tried it ruined everything
    for Q, c in zip(Q_array, c_array):
        A.append([
            Q * c,     # coefficient for S_p
            -c         # coefficient for K_p
        ])
        b.append((1 - c) * (K_w - Q))  # Known terms: K_w, S_w=1

    A = np.array(A)
    b = np.array(b)
      
    cond = np.linalg.cond(A.T @ A)
    if cond > 1e8:
        print(f"Warning: Ill-conditioned matrix (cond={cond:.2e})")
        print(A)
        try:
            # Safe least squares with bounds
            result = lsq_linear(A, b, bounds=(1e-8, np.inf))
            S_p, K_p = result.x
            return K_p, S_p
        except Exception as e:
            print("Least squares failed:", e)
            return np.nan, np.nan
    
    # Walowit 1987 least squares method (I think)
    AtA_inv = np.linalg.inv(np.dot(A.T, A))
    Atb = np.dot(A.T, b)
    S_p, K_p = np.dot(AtA_inv, Atb)
    
    return K_p, S_p

def Q_to_R(Q: np.array) -> np.array:
    return 1 / (1 + Q + np.sqrt(Q**2 + 2 * Q))
    
def plot_real_vs_predicted_reflectance(pigment_spectra: dict, predicted_reflectances: dict):
    cols = 14 # 1 graph and 11 swatches but first plot takes 2 columns
    rows = 3 * len(predicted_reflectances)
    fig = plt.figure(figsize=(cols * 1.5, rows * 1.0))
    height_ratios = [1.5, 1.5, 0.3] * len(predicted_reflectances)
    gs = gridspec.GridSpec(rows, cols, figure=fig, height_ratios=height_ratios)
    
    for i, pigment in enumerate(predicted_reflectances.keys()):
        if pigment == "titanium white":
            continue
        
        # plot real spectra
        ax_plot_real = fig.add_subplot(gs[3*i, 0:2])
        ax_plot_real.set_title(f"{pigment} real reflectance")
        
        # white swatch
        white_swatch = fig.add_subplot(gs[3*i, 2])
        white_rgb = pigment_spectra["titanium white"][100].to_rgb()
        white_swatch.imshow([[white_rgb]])
        white_swatch.axis("off")
        white_swatch.set_title("0%")
        
        for j, conc in enumerate(sorted(pigment_spectra[pigment].keys())):
            spec = pigment_spectra[pigment][conc]
            rgb = spec.to_rgb()
            
            spec.plot(name=pigment, ax=ax_plot_real, normalize=True, color = rgb)
            ax_plot_real.set_xlabel("Wavelength")
            ax_plot_real.set_ylabel("Reflectance")
            ax_plot_real.grid(True)
            # plot real swatches
            ax_swatch = fig.add_subplot(gs[3*i,j+3])
            ax_swatch.imshow([[rgb]])
            ax_swatch.axis("off")
            ax_swatch.set_title(f"{conc}%")
        # plot predicted spectra
        ax_plot_predicted = fig.add_subplot(gs[3*i + 1, 0:2])
        ax_plot_predicted.set_title(f"{pigment} predicted reflectance")
        
        # white swatch
        white_swatch = fig.add_subplot(gs[3*i + 1, 2])
        white_rgb = predicted_reflectances["titanium white"][100].to_rgb()
        white_swatch.imshow([[white_rgb]])
        white_swatch.axis("off")
        white_swatch.set_title("0%")

        for k, conc in enumerate(sorted(pigment_spectra[pigment].keys())):
            spec = predicted_reflectances[pigment][conc]
            rgb = spec.to_rgb()
            # predicted swatches
            spec.plot(name=pigment, ax=ax_plot_predicted, normalize=True, color = rgb)
            ax_plot_predicted.set_xlabel("Wavelength")
            ax_plot_predicted.set_ylabel("Reflectance")
            ax_plot_predicted.grid(True)
            
            ax_swatch = fig.add_subplot(gs[3*i + 1,k+3])
            ax_swatch.imshow([[rgb]])
            ax_swatch.axis("off")
            ax_swatch.set_title(f"{conc}%")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "real_vs_predicted.png")
    plt.savefig(path, dpi=img_dpi)
    plt.close()
    
    print("saved real vs predicted image")

def reverse_saunderson(R: Spectra, K1=0.035, K2=0.6) -> Spectra:
    """
    converts R_measured to R_inf
    """
    R_m = R.data
    R_inf = (R_m - K1) / (1 - K1 - K2 + K2 * R_m)	
    R_inf = np.clip(R_inf, 1e-4, 1)
    return Spectra(wavelengths=R.wavelengths, data = R_inf)

def saunderson_correction(R_inf: np.array, K1=0.035, K2=0.6):
    """
    converts R_inf to R_measured
    K1 = the fraction of light coming from outside the film that is reflected back out without ever entering the surface
    i.e. gloss, specular highlight. Berns assumed 0.035
    K2 = the fraction of light coming from inside the film that is reflected back inside at the surface. 
    theoretical value = 0.6
    """
    R_m = K1 + ((1 - K1) * (1 - K2) * R_inf) / (1 - K2 * R_inf)
    return np.clip(R_m, 0, 1)

def save_KS(KS_values: dict):
    data_to_save = {
    pigment: {
        variable: np.array(values).tolist()
        for variable, values in inner_dict.items()
    }
    for pigment, inner_dict in KS_values.items()
}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "oilpaints_KS.json")
    
    with open(json_path, "w") as f:
        json.dump(data_to_save, f)
        
    print("saved to oilpaints_KS.json")

def main():
    print("it started")
    wvls, pigment_spectra = load_reflectance_data()
    # if want smaller steps maybe do something like e.g.
    # spec.interpolate(385, 395, )etc
    
    # apply Saunderson corretion
    R_inf = defaultdict(dict)
    for p in pigment_spectra.keys():
        # TODO: "The internal reflectance of white was scaled by 1.005"
        for c in pigment_spectra[p].keys():
            # SAUNDERSON CORRECTION DOESNT WORK
            # R_inf[p][c] = pigment_spectra[p][c]
            R_inf[p][c] = reverse_saunderson(pigment_spectra[p][c])
    
    Q_white = KS_ratio(R_inf["titanium white"][100])
    
    # compute K/S ratio at each wavelength for each mixture we have
    KS_ratios = defaultdict(dict)
    for p in R_inf.keys():
        if p == "titanium white":
            KS_ratios[p][100] = Q_white
            continue
        # for everything except white, its 0% reflectance is just titanium white
        R_inf[p][0] = R_inf["titanium white"][100]
        KS_ratios[p][0] = Q_white
        for c in R_inf[p]:
            if c == 0:
                continue
            KS_ratios[p][c] = KS_ratio(R_inf[p][c])
    
    # assume S = 1.0 uniformly for white 
    K_white = Q_white
    S_white = np.ones_like(K_white)
    
    # dictionary that will store values for K_p, S_p
    KS_values = {}
    
    for p in KS_ratios.keys():
        if p == "titanium white":
            KS_values[p] = {"K": K_white,
                            "S": np.ones_like(wvls)
            }
            continue
        # K_p, S_p across each wavelength
        computed_K_ps = []
        computed_S_ps = []
        for i in range(len(wvls)):
            Q_vals = []
            c_vals = []
            for c in [10 * i for i in range(11)]:
                # cooked
                Q = KS_ratios[p][c][i]  # K/S at this wavelength and concentration
                Q_vals.append(Q)
                c_vals.append(c / 100)  # convert to [0, 1] range

            K_p, S_p = solve_KS(np.array(Q_vals), np.array(c_vals), K_white[i])
            
            computed_K_ps.append(K_p)
            computed_S_ps.append(S_p)
        
        KS_values[p] = {
        "K": np.array(computed_K_ps),
        "S": np.array(computed_S_ps), 
        }
        
    # compute predicted reflectance of each pigment, at each concentration
    predicted_reflectances = defaultdict(dict)
    # white isn't in compute_KS_values so I'll just add it manually. is that a bit disgusting?
    predicted_reflectances["titanium white"][100] = Spectra(wavelengths=wvls, data=Q_to_R(K_white / S_white))
    for p in KS_values:
        for c in pigment_spectra[p].keys():
            K_mix = c * KS_values[p]["K"] + (100 - c) * K_white
            S_mix = c * KS_values[p]["S"] + (100 - c) * S_white
            Q_mix = K_mix / S_mix
            # saunderson correction DOESNT WORK
            predicted_reflectances[p][c] = Spectra(wavelengths=wvls, data=saunderson_correction(Q_to_R(Q_mix)))
            # predicted_reflectances[p][c] = Spectra(wavelengths=wvls, data=Q_to_R(Q_mix))
    
    plot_real_vs_predicted_reflectance(pigment_spectra, predicted_reflectances)
    
    # save_KS(KS_values)

if __name__ == "__main__":
    main()
