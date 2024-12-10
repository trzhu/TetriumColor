from screeninfo import get_monitors

import numpy as np
import numpy.typing as npt
from typing import List
import tkinter as tk
import platform
import matplotlib.pyplot as plt
import os

from .PR650 import PR650


class MeasureDisplay:
    def __init__(self, pr650, save_directory: str = 'tmp', debug: bool = False, mode: str = 'RGB'):
        self.debug = debug
        # Create the main screen window
        self.main_window = tk.Tk()
        self.main_window.title("Main Window")
        self.main_window.geometry("400x300+100+100")

        if mode == 'RGB':
            # Create the second fullscreen window
            self.second_window = tk.Toplevel(self.main_window)
            self.second_window.withdraw()  # Start hidden until positioned

            self.current_color = [255, 0, 0]

            # Get monitor information and setup the second window
            self.setup_second_window()
            self.show_second_window()

            self.change_color_button = tk.Button(self.main_window, text="Set Background Color",
                                                 command=self.open_rgb_input_window)
            self.change_color_button.pack(pady=10)

        os.makedirs(save_directory, exist_ok=True)
        self.save_directory_var = tk.StringVar()
        self.save_directory_var.set(save_directory)

        self.new_dir_button = tk.Button(self.main_window, text="New Directory", command=self.create_new_directory)
        self.new_dir_button.pack(pady=10)

        self.measure_button = tk.Button(self.main_window, text="Measure", command=self.measure_spectra)
        self.measure_button.pack(pady=10)

        self.quit_button = tk.Button(self.main_window, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)
        self.main_window.protocol("WM_DELETE_WINDOW", self.quit_app)

        # need measuremnt device
        self.pr650: PR650 | None = None if debug else pr650
        # result tables
        self.spectras: List[npt.NDArray] = []
        self.luminances: List[npt.NDArray] = []

        self.save_directory = save_directory
        self.save_directory_label = tk.Label(self.main_window, textvariable=self.save_directory_var)
        self.save_directory_label.pack(pady=10)

    def create_new_directory(self):
        def submit_text():
            text = entry.get()
            self.save_directory = text
            self.save_directory_var.set(text)
            os.makedirs(self.save_directory, exist_ok=True)
            print(f"New directory created: {self.save_directory}")
            entry.delete(0, tk.END)  # Clear the entry field
            window.destroy()

        window = tk.Tk()
        window.title("Directory Input")

        label = tk.Label(window, text="Enter Text:")
        label.pack(pady=10)

        entry = tk.Entry(window)
        entry.pack()

        submit_button = tk.Button(window, text="Submit", command=submit_text)
        submit_button.pack(pady=10)

    def setup_second_window(self):
        # Get monitor information
        monitors = get_monitors()

        first_monitor = monitors[0]
        first_x = first_monitor.x

        if len(monitors) > 1:
            second_monitor = monitors[1]  # Assuming the second monitor is at index 1
            width = second_monitor.width
            height = second_monitor.height
            x = second_monitor.x
            y = second_monitor.y
            self.second_window.geometry(f"{width}x{height}+{x}+{y}")

            # Set fullscreen based on platform
            current_platform = platform.system()
            if current_platform == "Windows":
                self.second_window.overrideredirect(True)
                self.second_window.state("zoomed")
                self.second_window.bind("<F11>", lambda event: self.second_window.attributes("-zoomed",
                                        not self.second_window.attributes("-zoomed")))
                self.second_window.bind("<Escape>", lambda event: self.second_window.attributes("-zoomed", False))
            elif current_platform == "Darwin":  # macOS
                self.second_window.attributes("-fullscreen", True)
                self.second_window.bind("<F11>", lambda event: self.second_window.attributes("-fullscreen",
                                        not self.second_window.attributes("-fullscreen")))
                self.second_window.bind("<Escape>", lambda event: self.second_window.attributes("-fullscreen", False))
        else:
            print("Only one monitor detected. The second window will open on the main screen.")
            self.second_window.geometry("800x600")
        self.change_background_color(self.second_window, self.current_color)

    def show_second_window(self):
        # Show the second window
        self.second_window.deiconify()

    def quit_app(self):
        def submit_text():
            text = entry.get()
            if text:
                with open(os.path.join(self.save_directory, "description.txt"), "w") as f:
                    f.write(text)
            entry.delete(0, tk.END)  # Clear the entry field
            window.destroy()

        window = tk.Tk()
        window.title("Summary of Measurements")

        label = tk.Label(window, text="Enter Text:")
        label.pack(pady=10)

        entry = tk.Entry(window)
        entry.pack()

        submit_button = tk.Button(window, text="Submit", command=submit_text)
        submit_button.pack(pady=10)

        np.save(os.path.join(self.save_directory, "spectras.npy"), self.spectras)
        np.save(os.path.join(self.save_directory, "luminances.npy"), self.luminances)
        print(f"Saved Spectras and Luminances to {self.save_directory}")

        # Close both windows
        self.second_window.destroy()
        self.main_window.destroy()

    def open_rgb_input_window(self):
        def submit_rgb():
            try:
                r = int(entry_r.get())
                g = int(entry_g.get())
                b = int(entry_b.get())
                if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                    self.current_color = [r, g, b]
                    self.change_background_color(self.second_window, self.current_color)
                    window.destroy()
                else:
                    error_label.config(text="Values must be between 0 and 255")
            except ValueError:
                error_label.config(text="Please enter valid integers")

        window = tk.Tk()
        window.title("RGB Input")

        tk.Label(window, text="R:").pack(pady=5)
        entry_r = tk.Entry(window)
        entry_r.pack()

        tk.Label(window, text="G:").pack(pady=5)
        entry_g = tk.Entry(window)
        entry_g.pack()

        tk.Label(window, text="B:").pack(pady=5)
        entry_b = tk.Entry(window)
        entry_b.pack()

        submit_button = tk.Button(window, text="Submit", command=submit_rgb)
        submit_button.pack(pady=10)

        error_label = tk.Label(window, text="", fg="red")
        error_label.pack(pady=5)

    def rgb_to_hex(self, rgb):
        # Convert RGB tuple to hex color code
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def change_background_color(self, window, rgb):
        # Change the background color of the given window using RGB input
        hex_color = self.rgb_to_hex(rgb)
        window.configure(bg=hex_color)

    def measure_spectra(self):
        if self.pr650 is not None:
            print("Measuring Spectra...")
            spectra, lum = self.pr650.measureSpectrum()
            print(f"Done Measuring, Lum: {lum}")
            self.spectras.append(spectra)
            self.luminances.append(lum)

            plot_resulting_spectras(self.spectras)

    def show_next_color(self):
        self.current_color = self.current_color[2:] + self.current_color[:2]
        print(f"Current Color Showing is {self.current_color}")
        self.change_background_color(self.second_window, self.current_color)

    def run(self):
        self.main_window.mainloop()


def plot_resulting_spectras(spectras):
    for i, spectra in enumerate(spectras):
        plt.plot(spectra[0], spectra[1], label=f'Spectra {i+1}')

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Spectra Measurements')
    plt.legend()
    plt.show()


def save_spectrum_to_csv(filename):
    spectras = np.load(filename)
    nm, power = spectras[:, 0], spectras[:, 1]
    data = np.row_stack((nm[:1], power))
    np.savetxt('spectras.csv', data.T, delimiter=',', header='Wavelength,R,O,C,V', comments='')


def load_spectrum_from_npy(filename):
    data = np.load(filename)
    nm, power = data[:, 0], data[:, 1]
    return nm, power


# Example usage:
if __name__ == "__main__":
    mac_port_name = '/dev/cu.usbserial-A104D0XS'
    # pr650 = PR650(mac_port_name)
    app = MeasureDisplay(None, save_directory='tmp', debug=True)
    app.run()
    print(app.spectras)
