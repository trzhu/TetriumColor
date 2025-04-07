from colour.models import rgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
from PIL import Image

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

    # Test the new cubemap functionality
    test_gamut_cubemap(cs_4d)

    print("\nAll tests completed successfully!")


def visualize_equiluminant_plane(cs, points):
    """Visualize points on an equiluminant plane by plotting in 2D"""
    # Convert points to display space for RGB values
    rgb_values = np.clip(cs.convert(points, ColorSpaceType.VSH, ColorSpaceType.SRGB), 0, 1)
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


def test_gamut_cubemap(cs: ColorSpace):
    """Test the new cubemap-based gamut mapping functionality"""
    print("\nTest: Gamut cubemap generation and visualization")

    # Generate the gamut cubemap
    # Save individual cubemap faces
    output_dir = "cubemap_output"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize each cubemap face
    plt.figure(figsize=(15, 10))
    for i in range(6):
        if i in cs._gamut_cubemap:
            # Save the cubemap face as an image
            img_path = os.path.join(output_dir, f"face_{i}.png")
            cs._gamut_cubemap[i].save(img_path)

            # Display in the subplot
            plt.subplot(2, 3, i + 1)
            plt.imshow(np.array(cs._gamut_cubemap[i]))
            plt.title(f"Face {i}")
            plt.axis('off')

    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "cubemap_faces.png"))
    plt.show()
    print(f"Saved cubemap faces to {output_dir}/cubemap_faces.png")

    # Create a cross visualization
    combined_image = create_cubemap_cross_visualization(cs._gamut_cubemap)
    combined_image.save(os.path.join(output_dir, "cubemap_cross.png"))
    print(f"Saved cubemap cross visualization to {output_dir}/cubemap_cross.png")

    # Test gamut boundary interpolation
    test_interpolation(cs, output_dir)


def create_cubemap_cross_visualization(cubemap_images):
    """Create a cross layout visualization of the cubemap faces"""
    if not all(i in cubemap_images for i in range(6)):
        raise ValueError("Missing cubemap faces")

    # Get the size of a single face
    face_size = cubemap_images[0].width

    # Create a blank image for the cross layout
    width = 4 * face_size
    height = 3 * face_size
    combined_image = Image.new("RGB", (width, height), (0, 0, 0))

    # Paste faces into the cross layout
    # +X (0)
    combined_image.paste(cubemap_images[0], (2 * face_size, face_size))
    # -X (1)
    combined_image.paste(cubemap_images[1], (0, face_size))
    # +Y (2)
    combined_image.paste(cubemap_images[2], (face_size, 2 * face_size))
    # -Y (3)
    combined_image.paste(cubemap_images[3], (face_size, 0))
    # +Z (4)
    combined_image.paste(cubemap_images[4], (face_size, face_size))
    # -Z (5)
    combined_image.paste(cubemap_images[5], (3 * face_size, face_size))

    return combined_image


def test_interpolation(cs: ColorSpace, output_dir):
    """Test the interpolation of gamut boundaries from the cubemap"""
    print("\nTest: Gamut boundary interpolation")

    # Sample equiluminant planes at different luminance values and visualize
    luminance_values = [0.2, 0.5, 0.8]

    # Create plots for each luminance value
    fig = plt.figure(figsize=(15, 5))

    for i, lum in enumerate(luminance_values):
        # Sample the equiluminant plane
        points = cs.sample_equiluminant_plane(luminance=lum, num_points=100, remap_to_gamut=True)

        # Convert to Hering space for plotting
        hering_points = cs.convert(points, ColorSpaceType.VSH, ColorSpaceType.HERING)

        # Convert to sRGB for coloring (if available)
        try:
            rgb_values = np.clip(cs.convert(points, ColorSpaceType.VSH, ColorSpaceType.SRGB), 0, 1)
        except Exception:
            rgb_values = np.zeros((len(points), 3))
            # Use a simple coloring scheme based on hue angle
            for j in range(len(points)):
                hue = points[j, 2]
                rgb_values[j] = [0.5 * (1 + np.cos(hue)),
                                 0.5 * (1 + np.cos(hue + 2*np.pi/3)),
                                 0.5 * (1 + np.cos(hue + 4*np.pi/3))]

        # Plot the equiluminant plane
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        if cs.dim == 3:  # 2D gamut boundary
            ax.scatter(hering_points[:, 1], hering_points[:, 2], hering_points[:, 3], c=rgb_values)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        else:  # 3D gamut boundary (project to 2D)
            ax.scatter(hering_points[:, 1], hering_points[:, 2], hering_points[:, 3], c=rgb_values)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        ax.set_title(f'Luminance = {lum}')
        ax.set_aspect('equal')
        ax.grid(True)

    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "gamut_boundaries.png"))
    plt.show()
    print(f"Saved gamut boundary visualization to {output_dir}/gamut_boundaries.png")

    # Test specific angle interpolation
    if cs.dim == 3:  # For 2D case (one angle)
        test_angles = [(0.0,), (np.pi/4,), (np.pi/2,), (3*np.pi/4,), (np.pi,),
                       (5*np.pi/4,), (3*np.pi/2,), (7*np.pi/4,)]
    else:  # For 3D case (two angles)
        test_angles = [(0.0, 0.0), (np.pi/4, np.pi/4), (np.pi/2, np.pi/2),
                       (3*np.pi/4, np.pi/4), (np.pi, 0.0)]

    print("\nInterpolated gamut boundaries for specific angles:")
    for angle in test_angles:
        lum_cusp, sat_cusp = cs._interpolate_from_cubemap(angle)
        print(f"Angle: {angle}, Luminance cusp: {lum_cusp:.4f}, Saturation cusp: {sat_cusp:.4f}")


if __name__ == "__main__":
    test_color_space()
