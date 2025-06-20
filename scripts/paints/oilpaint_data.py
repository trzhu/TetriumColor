# script for getting the wavelength information from Artist_paint_spectra.xlsx and putting it in a json

import pandas as pd
import numpy as np
import json
import os

# dude i couldnt get it to find TetriumColor.Observer so I guess I have to do this
# even messing with .env and .vscode/settings.json???? it just didnt want to work?? am i stupid?
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from TetriumColor.Observer import Spectra, Illuminant, Observer

# getting the data from the xls was kinda slow so i'll save the important stuff to a json first
def save_reflectances_as_json(filePath):
    excel_file = pd.ExcelFile(filePath)
    df = pd.read_excel(excel_file, sheet_name="Oil")
    
    reflectance_cols = [col for col in df.columns if col.startswith("R")]
    wavelengths = [int(col[1:]) for col in reflectance_cols]
    
    # if wavelengths != wavelengths.sorted():
    #     print("yikes")
    
    grouped = df.groupby("Pigment")
    processed = []
    
    for pigment, group in grouped:
        # assume the lowest luminance (darkest) is the highest concentration
        group = group.sort_values(by="L* D65, 10°", ascending=True).reset_index(drop=True)
        for i, row in group.iterrows():
            label = f"{pigment.strip().strip("'")}"
            # if something has only one entry, like titanium white, it will automatically be 100% concentration
            concentration = 100 - 10 * i
            reflectance = row[6:].to_list() #R380 is on the 6th column
            processed.append((label, concentration, reflectance))
            # e.g.
            # ["pigment name", concentration, [reflectance at 380nm, reflectance at 390nm, ... reflectance at 730 nm]]
    
    data_to_save = {
        "wavelengths": wavelengths,
        "pigments, concentrations, reflectances": processed
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "oilpaints.json")
    
    with open(json_path, "w") as f:
        json.dump(data_to_save, f)
        
    print("saved to oilpaints.json")

if __name__ == "__main__":
    save_reflectances_as_json("C:\\Users\\User\\Documents\\Documents2\\tetrachromacy\\Berns archiving files\\Artist_paint_spectra.xlsx")
