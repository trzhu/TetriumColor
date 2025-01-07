"""
Linear programming-based metamer mismatch body calculation.

All Taken from Paul Centore on his website - https://www.munsellcolourscienceforpainters.com/ColourSciencePapers/ColourSciencePapers.html
Translated to Python by ChatGPT and Jessica Lee. 
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy.typing as npt

from TetriumColor.Observer import GetCustomObserver
from TetriumColor.Observer.Spectra import Illuminant


def draw_metamer_mismatch_body(response_functions1: npt.NDArray, response_functions2: npt.NDArray,
                               illuminant1: npt.NDArray, illuminant2: npt.NDArray, z: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the metamer mismatch body for two 3-dimensional observers and two illuminants.

    Args:
        response_functions1 (npt.NDArray):  sensor matrix for observer 1, dim x n
        response_functions2 (npt.NDArray): sensor matrix for observer 2, dim x n
        illuminant1 (npt.NDArray): illuminant for observer 1, n
        illuminant2 (npt.NDArray): illuminant for observer 2, n
        z (npt.NDArray): color that we want to compute the MMB on, dim

    Raises:
        ValueError: if z does not have three entries, or if linear programming fails

    Returns:
        tuple[npt.NDArray, npt.NDArray]: returns the vertices and the bounding reflectance spectra
    """
    # Initialize variables
    vertices = []
    bounding_reflectance_spectra = []
    display_figure = True

    if z.size != 3:
        raise ValueError("z must have three entries.")

    zvert = z.reshape(3, 1)

    illuminant1 = illuminant1[np.newaxis]
    illuminant2 = illuminant2[np.newaxis]

    # Construct transformations Phi and Psi
    phi_vecs = response_functions1 * illuminant1
    psi_vecs = response_functions2 * illuminant2

    # Loop over directions using spherical coordinates
    fineness_index = 5
    thetas = np.linspace(0, 360, 8 * fineness_index + 8)
    phis = np.linspace(0, 90, 2 ** (fineness_index + 1) + 1)

    for theta in thetas:
        for phi in phis:
            alpha_1 = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
            alpha_2 = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
            alpha_3 = np.cos(np.radians(phi))

            F = np.array([alpha_1, alpha_2, alpha_3])
            psi_star_F = np.sum(psi_vecs * F[np.newaxis].T, axis=0)

            # Find minimum vertex
            x_min = solve_linear_program(psi_star_F, phi_vecs, zvert)
            vertex_min = np.dot(psi_vecs, x_min)
            vertices.append(vertex_min)
            bounding_reflectance_spectra.append(x_min)

            # Find maximum vertex
            x_max = solve_linear_program(-psi_star_F, phi_vecs, zvert)
            vertex_max = np.dot(psi_vecs, x_max)
            vertices.append(vertex_max)
            bounding_reflectance_spectra.append(x_max)

    vertices = np.array(vertices)
    bounding_reflectance_spectra = np.array(bounding_reflectance_spectra)

    # Remove duplicate vertices using convex hull
    hull = ConvexHull(vertices)
    new_vertices = vertices[hull.vertices]
    bounding_reflectance_spectra = bounding_reflectance_spectra[hull.vertices]

    if display_figure:
        # Plot the points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(new_vertices[:, 0], new_vertices[:, 1], new_vertices[:, 2], marker='o')

        # Plot the convex hull
        for simplex in hull.simplices:
            # Get the vertices of the simplex
            simplex_points = vertices[simplex]
            # Add a 3D polygon for the simplex
            poly = Poly3DCollection([simplex_points], alpha=0.3, edgecolor='k')
            ax.add_collection3d(poly)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    return vertices, bounding_reflectance_spectra


def solve_linear_program(c, A, b):
    """
    Placeholder for linear programming solver to minimize c^T x subject to Ax = b,
    0 <= x <= 1.
    """
    from scipy.optimize import linprog
    bounds = [(0, 1)] * A.shape[1]
    res = linprog(c, A_eq=A, b_eq=b.flatten(), bounds=bounds, method='highs')
    if not res.success:
        raise ValueError("Linear programming failed.")
    return res.x


if __name__ == "__main__":
    # Example usage (replace with actual inputs)
    wavelengths = np.arange(400, 710, 2)
    observer1 = GetCustomObserver(wavelengths=wavelengths, dimension=3)
    observer2 = GetCustomObserver(wavelengths=wavelengths, dimension=3,
                                  s_cone_peak=547, m_cone_peak=551, l_cone_peak=555)
    illuminant1 = np.ones(len(wavelengths))
    illuminant2 = np.ones(len(wavelengths))  # Illuminant.get("D65").interpolate_values(wavelengths).data
    z = np.random.rand(3)
    draw_metamer_mismatch_body(observer1.normalized_sensor_matrix,
                               observer2.normalized_sensor_matrix, illuminant1, illuminant2, np.array([0.5, 0.5, 0.5]))
