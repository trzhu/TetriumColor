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

from TetriumColor.Observer import Spectra

img_dpi = 48 # make smaller or larger for desired size of pngs

def load_KS_data():
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
    KS_values = load_KS_data()
    print(KS_values)
    
if __name__ == "__main__":
    main()