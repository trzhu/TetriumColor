import pandas as pd
import numpy as np
import json, os, math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.optimize import lsq_linear

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from TetriumColor.Observer import Spectra, Inks

img_dpi = 48

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
    # let Q = K/S, then R_infinity = 1 + Q - sqrt((1+Q)^2 - 1)
    R_inf = 1 + Q - np.sqrt((1+Q)**2 - 1)
    return R_inf

def main():
    KS_values = load_oilpaint_KS_data()
    print(KS_values)
    
    # ACTUALLY i can probably make pigment objects now
    

# btw FOR SOME REASON, in Artist_paint_spectra.xlsx phthalo blue is spelled pthalo blue
if __name__ == "__main__":
    main()