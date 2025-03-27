from colour.models import rgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer import Observer
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries


def test_color_space():
    print("Testing ColorSpace functionality...")

    # Create an observer with default parameters
    observer = Observer.tetrachromat(wavelengths=np.arange(360, 831, 1))

    # Test 1: Initialize ColorSpace with default display
    print("\nTest 1: Initialize ColorSpace with default display")
    cs_default = ColorSpace(observer)
    print(f"ColorSpace dimensions: {cs_default.dim}")
    print(f"Metameric axis: {cs_default.metameric_axis}")

    # Test 2: Initialize ColorSpace with sRGB display
    print("\nTest 2: Initialize ColorSpace with sRGB display")
    try:
        cs_sRGB = ColorSpace(observer, display='sRGB')
    except Exception as e:
        print(f"Could not initialize ColorSpace with sRGB display: {e}")

    primaries = LoadPrimaries("../measurements/2025-01-16/primaries")
    gaussian_smooth_primaries = GaussianSmoothPrimaries(primaries)

    cs_4d = ColorSpace(observer, display=gaussian_smooth_primaries)

    # Test 3: Color space conversion
    print("\nTest 3: Color space conversion")
  # Create a test point in VSH space (value, saturation, hue)
    vsh_point = np.array([[0.5, 0.5, 0.0]])  # For 3D
    if cs_4d.dim == 4:
        vsh_point = np.array([[0.5, 0.5, 0.0, 0.0]])  # For 4D

    print(f"Original VSH point: {vsh_point}")

    # Convert to HERING
    hering_point = cs_4d.convert(vsh_point, ColorSpaceType.VSH, ColorSpaceType.HERING)
    print(f"HERING point: {hering_point}")

    # Convert to DISPLAY
    display_point = cs_4d.convert(vsh_point, ColorSpaceType.VSH, ColorSpaceType.MAXBASIS)
    print(f"DISPLAY point: {display_point}")

    # Convert to CONE
    cone_point = cs_4d.convert(vsh_point, ColorSpaceType.VSH, ColorSpaceType.CONE)
    print(f"CONE point: {cone_point}")

    # Convert to RGB_OCV
    rgb_ocv_point = cs_4d.convert(vsh_point, ColorSpaceType.VSH, ColorSpaceType.RGB_OCV)
    print(f"RGB_OCV point: {rgb_ocv_point}")

    # Test 4: Round-trip conversion
    print("\nTest 4: Round-trip conversion")
    vsh_round_trip = cs_4d.convert(hering_point, ColorSpaceType.HERING, ColorSpaceType.VSH)
    print(f"Round-trip VSH point: {vsh_round_trip}")
    print(f"Round-trip error: {np.abs(vsh_point - vsh_round_trip).sum()}")

    # Test 5: Sampling equiluminant plane
    print("\nTest 5: Sampling equiluminant plane")
    equi_points = cs_4d.sample_equiluminant_plane(luminance=0.5, num_points=1000, remap_to_gamut=False)
    print(f"Sampled {len(equi_points)} points on equiluminant plane")
    print(f"Sample points shape: {equi_points.shape}")

    print("\nTest 6: Gamut operations")
    # Generate a point outside the gamut with high saturation
    out_of_gamut_point = vsh_point.copy()
    out_of_gamut_point[0, 1] = 2.0  # Increase saturation beyond gamut

    print(f"Original point: {vsh_point}")
    print(f"Out of gamut point: {out_of_gamut_point}")

    # Check if points are in gamut
    in_gamut_orig = cs_4d.is_in_gamut(vsh_point)
    in_gamut_high = cs_4d.is_in_gamut(out_of_gamut_point)

    print(f"Original point in gamut: {in_gamut_orig}")
    print(f"High saturation point in gamut: {in_gamut_high}")

    # Remap out-of-gamut point
    remapped_point = cs_4d.remap_to_gamut(out_of_gamut_point)
    print(f"Remapped point: {remapped_point}")
    print(f"Remapped point in gamut: {cs_4d.is_in_gamut(remapped_point)}")

    # Test 7: Conversion to TetraColor and PlateColor
    print("\nTest 7: Conversion to TetraColor and PlateColor")
    tetra_colors = cs_4d.to_tetra_color(vsh_point)
    print(f"TetraColor: {tetra_colors[0]}")

    plate_color = cs_4d.to_plate_color(vsh_point[0], background_luminance=0.2)
    print(f"PlateColor: {plate_color}")

    # Test 8: Get metameric axis
    print("\nTest 8: Get metameric axis")
    metameric_axis = cs_4d.get_metameric_axis()
    print(f"Metameric axis: {metameric_axis}")

    # Visualize sampled equiluminant plane if matplotlib is available
    try:
        visualize_equiluminant_plane(cs_4d, equi_points)
    except Exception as e:
        print(f"Could not visualize equiluminant plane: {e}")

    print("\nAll tests completed successfully!")


def visualize_equiluminant_plane(cs, points):
    """Visualize points on an equiluminant plane by plotting in 2D"""
    # Convert points to display space for RGB values
    rgb_values = np.clip(cs.convert(points, ColorSpaceType.VSH, ColorSpaceType.sRGB), 0, 1)
    hering_points = cs.convert(points, ColorSpaceType.VSH, ColorSpaceType.HERING)
    # Normalize for RGB display (assuming first 3 dimensions are RGB)

    # Extract x-y coordinates from VSH (assuming these are the last two dimensions)
    # We'll create a 2D plot using the saturation and hue
    # Create a scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hering_points[:, 1], hering_points[:, 2], hering_points[:, 3], s=100, c=rgb_values)
    plt.title("Hering Equiluminant Points")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    test_color_space()
