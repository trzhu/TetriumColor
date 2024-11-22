import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


def DisplayParallepiped(ax, V):
    """
    Display the Parallepiped defined by the spanning vectors
    """
    # Generate all vertices of the parallelepiped
    num_vectors = V.shape[1]
    vertices = np.array([
        np.dot(V, np.array(t))
        # Generate all 2^n combinations of 0 and 1
        for t in np.ndindex(*(2,) * num_vectors)
    ])

    # Compute the convex hull of the vertices
    hull = ConvexHull(vertices)

    # Plot the vertices
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=hull.simplices, color='cyan', alpha=0.1)

    # Plot the vectors
    for i in range(num_vectors):
        ax.quiver(0, 0, 0, V[0, i], V[1, i], V[2, i], color='k')

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([-0.1, 1])
    ax.set_zlim([-0.1, 1])


def Display2DPlane(ax, v1, v2):
    """
    Display the 2D Plane defined by the spanning vectors
    """
    # Create a grid of points
    x = np.linspace(-0.1, 1.1, 10)
    y = np.linspace(-0.1, 1.1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute the plane points
    plane_points = np.outer(v1, X.flatten()) + np.outer(v2, Y.flatten())

    # Plot the plane
    ax.plot_trisurf(plane_points[0], plane_points[1], plane_points[2], color='cyan', alpha=0.1)

    # Plot the vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='g')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
