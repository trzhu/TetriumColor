import numpy.typing as npt
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform


def ProjectPointOntoPlane(p: npt.NDArray, q: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """
    Projects a point p onto a plane defined by a point q and a normal vector n.

    Parameters:
    - p: (N,) array, point to project
    - q: (N,) array, a point on the plane
    - n: (N,) array, normal vector of the plane (should be normalized)

    Returns:
    - p_proj: (N,) array, projected point on the plane
    - v_orthogonal: (N,) array, orthogonal vector from p to the plane
    """
    # Ensure n is normalized
    n = n / np.linalg.norm(n)

    # Compute the vector from p to q
    v = p - q

    # Compute the scalar projection
    d = np.dot(v, n)

    # Compute the projected point
    p_proj = p - d * n

    # Compute the orthogonal vector
    # v_orthogonal = d * n

    return p_proj


def ParallelepipedToInequalities(p0, V):
    """
    Convert a parallelepiped defined by an origin and spanning vectors into linear inequalities.

    Parameters:
    - p0: (N,) array, the origin of the parallelepiped
    - V: (N x N) array, spanning vectors (columns)

    Returns:
    - A: (M x N) array of inequality coefficients
    - b: (M,) array of inequality bounds
    """
    # Generate all vertices of the parallelepiped
    num_vectors = V.shape[1]
    vertices = np.array([
        p0 + np.dot(V, np.array(t))
        # Generate all 2^n combinations of 0 and 1
        for t in np.ndindex(*(2,) * num_vectors)
    ])

    # Compute the convex hull of the vertices
    hull = ConvexHull(vertices)

    # Extract inequalities A, b from the convex hull
    A = hull.equations[:, :-1]  # Coefficients of the facets
    b = -hull.equations[:, -1]  # Offsets (note the sign)

    return A, b


def MaximizeDimensionOnSubspace(A, b, p0, V, k):
    """
    find the maximum value along the k-th dimension of a plane's parameter space
    that intersects with an n-dimensional parallelepiped.

    Parameters:
    - A: (P x N) matrix of parallelepiped constraints (A @ x <= b)
    - b: (P,) vector of parallelepiped bounds
    - p0: (N,) vector, the origin of the subspace (plane)
    - V: (N x M) matrix of spanning vectors for the M-dimensional subspace
    - k: Index of the dimension in the subspace to maximize

    Returns:
    - max_value: Maximum value of t_k
    - result.x: Values of [t1, t2, ..., tM] that maximize t_k
    """
    # Number of spanning vectors
    M = V.shape[1]  # Dimensionality of the subspace

    # Transform parallelepiped constraints: A @ (p0 + V @ t) <= b
    A_plane = np.dot(A, V)  # Transform spanning vectors
    b_plane = b - np.dot(A, p0)  # Shift bounds by the plane's origin

    # Objective function: Maximize t_k (k-th parameter of the subspace)
    c = np.zeros(M)
    c[k] = -1  # Negate to maximize (linprog minimizes by default)

    # Solve linear program
    result = linprog(c, A_ub=A_plane, b_ub=b_plane,
                     bounds=(None, None), method="highs")

    if result.success:
        max_value = -result.fun  # Negate to get the maximum
        return max_value, result.x
    else:
        raise ValueError("Optimization failed. No feasible solution found.")


def FindMaximalSaturation(hue_direction: npt.NDArray, paralleletope_vecs: npt.NDArray) -> npt.NDArray:
    """Find the Maximal Point that lies along a given hue direction in a paralleletope.

    Args:
        hue_direction (npt.NDArray): point in display basis space to represent the hue dimension in (theta, phi,.. etc)
        paralleletope_vecs (npt.NDArray): vectors in display basis space in R^N that define the paralleletope

    Returns:
        npt.NDArray: The maximal point in the paralleletope that lies along the hue direction
    """
    N = paralleletope_vecs.shape[1]  # Dimension of parallelepiped

    p0 = np.zeros(N)  # Origin
    A, b = ParallelepipedToInequalities(p0, paralleletope_vecs)

    v1 = np.ones(N)/np.linalg.norm(np.ones(N))  # Luminance vector
    v2 = ProjectPointOntoPlane(hue_direction, p0, v1)  # Hue Vector
    V = np.column_stack((v1, v2))  # Combine spanning vectors

    # Maximize along t2 Hue dimension, which should correspond to saturation
    k = 1
    max_t2, optimal_t = MaximizeDimensionOnSubspace(A, b, p0, V, k)
    # Back-transform to original space:
    x_max = p0 + np.dot(V, optimal_t)
    return x_max


def FindMaximumIn1DimDirection(point: npt.NDArray, metameric_direction: npt.NDArray, paralleletope_vecs: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Find maximal point that lies in one direction inside of a linear constraint set

    Args:
        display_space_direction (npt.NDArray): _description_
        paralleletope_vecs (npt.NDArray): _description_

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Returns in both positive and negative directions the maximal point
    """
    N = paralleletope_vecs.shape[1]  # Dimension of parallelepiped

    p0 = np.zeros(N)
    A, b = ParallelepipedToInequalities(p0, paralleletope_vecs)
    V = metameric_direction[np.newaxis, :].T
    # Maximize in the only dimension of the subspace
    k = 0
    # positive direction optimization
    max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, point, V, k)
    x_max = point + np.dot(V, optimal_t)

    # negative direction optimization
    max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, point, -1 * V, k)
    x_min = point + np.dot(-1 * V, optimal_t)

    return x_max, x_min


if __name__ == "__main__":
    paralleletope_vecs = np.eye(3)
    ans = FindMaximalSaturation(np.array([0.5, 1, 0]), paralleletope_vecs)
    print(ans)

    paralleletope_vecs = np.eye(4)
    ans = FindMaximalSaturation(np.array([1, 0.4, 0.1, 0.1]), paralleletope_vecs)
    print(ans)
