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

img_dpi = 72 # make smaller or larger for desired size of pngs

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

def apply_saunderson_and_scale_white(pigment_spectra: dict, useSaunderson: bool, scaleWhite: bool):
    R_inf = defaultdict(dict)
    for p in pigment_spectra.keys():
        for c in pigment_spectra[p].keys():
            # SAUNDERSON CORRECTION
            if useSaunderson:
                R_inf[p][c] = reverse_saunderson(pigment_spectra[p][c])
            else:
                R_inf[p][c] = pigment_spectra[p][c]
    
    # "The internal reflectance of white was scaled by 1.005"
    if scaleWhite:
        R_inf["titanium white"][100] *= 1.005
    return R_inf

def compute_all_KS_ratios(R_inf):
    # Q for quotient lol
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

    return KS_ratios

# returns K/S = (1-R)^2 / 2R for every lambda
def KS_ratio(spec: Spectra) -> np.ndarray:
    reflectance = spec.data
    return ((1-reflectance)**2) / (2 * reflectance)

# returns reflectance predicted by KM
# as an array like the wavelength arrays
# R_bg = reflectance of background substrate
def mix(weights: np.array, K_pigments: np.array, S_pigments: np.array, d=1.0, R_bg=1.0) -> np.array:
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
    
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        print("Invalid result: NaN or inf")
      
    # least square version
    # result = lsq_linear(A, b, bounds=(1e-32, np.inf))
    # S_p, K_p = result.x
    
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

def solve_all_KS(wavelengths, KS_ratios):
    # assume S = 1.0 uniformly for white 
    K_white = KS_ratios["titanium white"][100]
    S_white = np.ones_like(K_white)
    
    # dictionary that will store values for K_p, S_p
    KS_values = {}
    
    for p in KS_ratios.keys():
        if p == "titanium white":
            KS_values[p] = {"K": K_white,
                            "S": np.ones_like(wavelengths)
            }
            continue
        # K_p, S_p across each wavelength
        computed_K_ps = []
        computed_S_ps = []
        for i in range(len(wavelengths)):
            Q_vals = []
            c_vals = []
            for c in [10 * i for i in range(11)]:
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
    return KS_values


def Q_to_R(Q: np.array) -> np.array:
    return 1 / (1 + Q + np.sqrt(Q**2 + 2 * Q))

# compute predicted reflectance of each pigment, at each concentration
def predict_all_reflectance(KS_values, wvls, pigment_spectra, useSaunderson):
    K_white = KS_values["titanium white"]["K"]
    S_white = KS_values["titanium white"]["S"]
    
    predicted_reflectances = defaultdict(dict)
    # white isn't in compute_KS_values so I'll just add it manually. is that a bit disgusting?
    predicted_reflectances["titanium white"][100] = Spectra(wavelengths=wvls, data=Q_to_R(K_white / S_white))
    for p in KS_values:
        for c in pigment_spectra[p].keys():
            K_mix = c * KS_values[p]["K"] + (100 - c) * K_white
            S_mix = c * KS_values[p]["S"] + (100 - c) * S_white
            Q_mix = K_mix / S_mix
            # saunderson correction
            if useSaunderson:
                predicted_reflectances[p][c] = Spectra(wavelengths=wvls, data=saunderson_correction(Q_to_R(Q_mix)))
            else:
                predicted_reflectances[p][c] = Spectra(wavelengths=wvls, data=Q_to_R(Q_mix))

    return predicted_reflectances
    
def plot_real_vs_predicted_reflectance(pigment_spectra: dict, predicted_reflectances: dict):
    cols = 14 # 1 graph and 11 swatches but first plot takes 2 columns
    rows = 3 * len(predicted_reflectances) # every 3rd row is empty (padding)
    fig = plt.figure(figsize=(cols * 1.5, rows * 2.0))
    height_ratios = [2.0, 2.0, 0.1] * len(predicted_reflectances)
    gs = gridspec.GridSpec(rows, cols, figure=fig, height_ratios=height_ratios, hspace=0.4)
    
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
            # ax_plot_real.set_xlabel("Wavelength") # brugg it overlaps with titles
            ax_plot_real.tick_params(axis='x', labelbottom=False) # also overlaps with titles
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
    
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.01, right=0.99)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "real_vs_predicted.png")
    plt.savefig(path, dpi=img_dpi, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print("saved real vs predicted image")

def plot_reflectance_per_pigment(pigment_spectra: dict, predicted_reflectances: dict, useSaunderson=None, scaleWhite=None):
    cols = 14  # 1 graph + 11 swatches, first plot takes 2 columns

    for pigment in predicted_reflectances:
        if pigment == "titanium white":
            continue

        # help i dont need the 3rd row spacer anymore
        # but leaving it there makes the graph a bit taller which might be better idk
        rows = 3
        height_ratios = [2.0, 2.0, 0.1]

        fig = plt.figure(figsize=(cols * 1.5, rows * 2.0))
        gs = gridspec.GridSpec(rows, cols, figure=fig, height_ratios=height_ratios)

        ### --- REAL REFLECTANCE ---
        ax_plot_real = fig.add_subplot(gs[0, 0:2])
        ax_plot_real.set_title(f"{pigment} real reflectance", fontsize=9, pad=5)

        white_rgb = pigment_spectra["titanium white"][100].to_rgb()
        white_swatch = fig.add_subplot(gs[0, 2])
        white_swatch.imshow([[white_rgb]])
        white_swatch.axis("off")
        white_swatch.set_title("0%", fontsize=6)

        for j, conc in enumerate(sorted(pigment_spectra[pigment].keys())):
            spec = pigment_spectra[pigment][conc]
            rgb = spec.to_rgb()
            spec.plot(name=pigment, ax=ax_plot_real, normalize=True, color=rgb)
            ax_plot_real.tick_params(axis='x', labelbottom=False)
            ax_plot_real.set_ylabel("Reflectance")
            ax_plot_real.grid(True)

            ax_swatch = fig.add_subplot(gs[0, j + 3])
            ax_swatch.imshow([[rgb]])
            ax_swatch.axis("off")
            ax_swatch.set_title(f"{conc}%", fontsize=6)

        ### --- PREDICTED REFLECTANCE ---
        ax_plot_pred = fig.add_subplot(gs[1, 0:2])
        ax_plot_pred.set_title(f"{pigment} predicted reflectance", fontsize=9, pad=5)

        white_rgb_pred = predicted_reflectances["titanium white"][100].to_rgb()
        white_swatch_pred = fig.add_subplot(gs[1, 2])
        white_swatch_pred.imshow([[white_rgb_pred]])
        white_swatch_pred.axis("off")
        white_swatch_pred.set_title("0%", fontsize=6)

        for k, conc in enumerate(sorted(predicted_reflectances[pigment].keys())):
            spec = predicted_reflectances[pigment][conc]
            rgb = spec.to_rgb()
            spec.plot(name=pigment, ax=ax_plot_pred, normalize=True, color=rgb)
            ax_plot_pred.set_xlabel("Wavelength")
            ax_plot_pred.set_ylabel("Reflectance")
            ax_plot_pred.grid(True)

            ax_swatch = fig.add_subplot(gs[1, k + 3])
            ax_swatch.imshow([[rgb]])
            ax_swatch.axis("off")
            ax_swatch.set_title(f"{conc}%", fontsize=6)

        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.01, right=0.99)

        # make folder if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(current_dir, f"real vs predicted (saunderson {useSaunderson} scaleWhite {scaleWhite})")
        os.makedirs(folder, exist_ok=True)  # Creates the folder if it doesn't exist

        # Save per-pigment
        safe_name = pigment.lower().replace(" ", "_").replace("/", "_")
        path = os.path.join(current_dir, folder, f"{safe_name} (saunderson {useSaunderson} scaleWhite {scaleWhite}).png")
        plt.savefig(path, dpi=img_dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"Saved: {path}")

def reverse_saunderson(R: Spectra, K1=0.035, K2=0.6) -> Spectra:
    """
    converts R_measured to R_inf
    """
    R_m = R.data
    R_inf = (R_m - K1) / (1 - K1 - K2 + K2 * R_m)
    # clipping is ruining my life
    R_inf = np.clip(R_inf, 1e-4, 1)
    # let's try clipping negative but not > 1 values
    # R_inf = np.clip(R_inf, 1e-4, None)
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

def save_KS(KS_values: dict, useSaunderson=None, scaleWhite=None):
    data_to_save = {
    pigment: {
        variable: np.array(values).tolist()
        for variable, values in inner_dict.items()
    }
    for pigment, inner_dict in KS_values.items()
}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, f"oilpaints_KS_saunderson{useSaunderson}_scaleWhite{scaleWhite}.json")
    
    with open(json_path, "w") as f:
        json.dump(data_to_save, f)
        
    print("saved to oilpaints_KS.json")

def load_oilpaint_KS_data():
    """
    returns list of wavelengths and dictionary like 
    KS_values[pigment]["K"] = K values over every wavelength
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "oilpaints_KS.json")
    
    with open(json_path, "r") as f:
        loaded_data = json.load(f)
    
    KS_values = defaultdict(dict)
    for pigment in loaded_data.keys():
        for variable in loaded_data[pigment].keys():
            KS_values[pigment][variable] = np.array(loaded_data[pigment][variable])
    
    return KS_values

def main():
    useSaunderson = True
    scaleWhite = True
    
    print("it started")
    wvls, pigment_spectra = load_reflectance_data()
    
    R_inf = apply_saunderson_and_scale_white(pigment_spectra, useSaunderson, scaleWhite)
    
    KS_ratios = compute_all_KS_ratios(R_inf)
    
    KS_values = solve_all_KS(wvls, KS_ratios)
    
    predicted_reflectances = predict_all_reflectance(KS_values, wvls, pigment_spectra, useSaunderson)
            
    # plot_real_vs_predicted_reflectance(pigment_spectra, predicted_reflectances)
    plot_reflectance_per_pigment(pigment_spectra, predicted_reflectances, useSaunderson, scaleWhite)
    
    # save_KS(KS_values, useSaunderson, scaleWhite)

if __name__ == "__main__":
    main()
