import pandas as pd
import numpy as np
import json, os, math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from oilpaint_mixing import *
from TetriumColor.Observer import Spectra, Inks, Pigment

img_dpi = 48

def main():
    wavelengths, pigment_spectra = load_reflectance_data()
    # TODO: make a function that automatically loads spectra objects
    KS_values = load_oilpaint_KS_data()
    print(reflectances)
    # print(KS_values)
    
    pigments = {}
    
    print(reflectances["titanium white"][100])
    
    for p in KS_values:
        pigments[p] = Pigment(array=reflectances[p][100], k=KS_values[p]["K"], s=KS_values[p]["S"])
    
    
    
    # ACTUALLY i can probably make pigment objects now

if __name__ == "__main__":
    main()