from colour.models.rgb.datasets import canon_cinema_gamut
import numpy as np
import numpy.typing as npt
from typing import List
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def stretch_matrix_along_direction(d, scale):
    d = d / np.linalg.norm(d)
    P = np.outer(d, d)        # projection matrix onto d
    I = np.eye(3)
    S = I + (scale - 1) * P   # stretch along d by 'scale'
    return S


def yz_plane_equiangular_error(vectors):
    """
    Given a list of 3D vectors, compute how evenly spaced they are in the YZ-plane.
    Assumes vectors are originating from the same point.

    Parameters:
    - vectors: np.ndarray of shape (n, 3)

    Returns:
    - angular_error: float (lower is better, 0 is perfect)
    """
    # Project to YZ-plane
    yz_vectors = vectors[:, 1:3]

    # Normalize to unit vectors (in YZ-plane)
    norms = np.linalg.norm(yz_vectors, axis=1, keepdims=True)
    yz_unit_vectors = yz_vectors / np.clip(norms, 1e-8, None)

    # Compute angles in YZ-plane (atan2 gives angle from Y axis)
    angles = np.arctan2(yz_unit_vectors[:, 1], yz_unit_vectors[:, 0])  # angle from Y (y=0) counter-clockwise
    angles = np.mod(angles, 2 * np.pi)  # Wrap to [0, 2pi)

    # Sort angles
    angles = np.sort(angles)

    # Compute angle differences between adjacent vectors
    angle_diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))  # Close the circle

    # Ideal angle spacing
    n = len(vectors)
    ideal_angle = 2 * np.pi / n

    # Total squared deviation from ideal spacing
    error = np.sum((angle_diffs - ideal_angle) ** 2)

    return error


def solve_transformation_matrix(source_points, target_points):
    """
    Solve for a transformation matrix that maps source points to target points
    while ensuring (1,0,0) maps to (1,0,0) and preserving angular relationships.

    Parameters:
    - source_points: numpy array of shape (n, 3), each row is a source point
    - target_points: numpy array of shape (n, 3), each row is a target point

    Returns:
    - transformation_matrix: numpy array of shape (3, 3)
    - error: final optimization error
    """
    # Initial guess for the 6 free parameters (a12, a13, a22, a23, a32, a33)
    initial_guess = np.zeros(6)

    def objective_function(params):
        a12, a13, a22, a23, a32, a33 = params

        # Build matrix A where (1,0,0) stays fixed
        A = np.array([
            [1.0, a12, a13],
            [0.0, a22, a23],
            [0.0, a32, a33]
        ])

        transformed_points = np.array([A @ p for p in source_points])
        error = np.mean(np.sum((transformed_points - target_points) ** 2))
        luminance_error = np.mean(np.sum((source_points[:, 0] - transformed_points[:, 0])**2))

        # compute the perpendicular distance from the line (1, 0, 0)
        # Compute the perpendicular distance from the line (1, 0, 0)
        chromas = np.linalg.norm(transformed_points[:, 1:], axis=1)  # distance from (0, 0)
        actual_chromas = np.linalg.norm(target_points[:, 1:], axis=1)
        print("Actual chromas: ", actual_chromas)
        print("Derived chromas: ", chromas)
        chroma_error = np.mean((actual_chromas - chromas) ** 2)

        # Compute the y-z angle between vectors
        should_be_low = yz_plane_equiangular_error(target_points)
        angular_error = yz_plane_equiangular_error(transformed_points)
        print("Should be low: ", should_be_low)
        print("Actual error: ", angular_error)

        print(
            f"Current params: {params}, Error: {error:.4f}, Luminance error: {luminance_error:.4f}, Chroma error: {chroma_error:.4f}")
        combined_error = error + angular_error + 10 * chroma_error
        return combined_error * 100
    # # Solve the optimization problems
    # result = minimize(objective_function, initial_guess, method='BFGS', options={'maxiter': 100000})

    from scipy.optimize import differential_evolution

    bounds = [(-2, 2)] * 6  # Reasonable bounds for a12, a13, a22, ..., a33
    result = differential_evolution(objective_function, bounds, strategy='best1bin',
                                    maxiter=10000, polish=True, disp=True)

    # Extract the optimized parameters
    a12, a13, a22, a23, a32, a33 = result.x

    # Construct the final transformation matrix
    transformation_matrix = np.array([
        [1.0, a12, a13],
        [0.0, a22, a23],
        [0.0, a32, a33]
    ])

    print("Optimization result status:", result.message)
    print("Optimization error:", result.fun)

    # Validate the transformation
    # validate_transformation(source_points, target_points, transformation_matrix)
    visualize_transformation(source_points, target_points, source_points @ transformation_matrix.T)

    return transformation_matrix, result.fun


def find_interpolated_vector(v, h, c, projection_vector=np.ones(3)):
    """
    Find a vector that lies on the interpolation between the ones vector and a given vector v,
    with projection length h onto the ones vector and rejection magnitude c.

    Parameters:
    ----------
    v : numpy.ndarray
        The reference vector (e.g., [1, 0, 0])
    h : float
        Desired projection length onto the ones vector
    c : float
        Desired magnitude of the rejection component

    Returns:
    -------
    numpy.ndarray
        The constructed vector with the specified properties
    """
    dimensions = len(v)
    if dimensions != len(projection_vector):
        raise ValueError("The dimensions of the input vector and projection vector must match.")
    # Create and normalize the ones vector
    ones = projection_vector  # np.ones(dimensions)
    ones_unit = ones / np.linalg.norm(ones)  # Unit ones vector

    # Project v onto the ones vector
    v_proj = np.dot(v, ones_unit) * ones_unit

    # Find the rejection (perpendicular component)
    v_rej = v - v_proj

    # Normalize the rejection vector (if not zero)
    v_rej_norm = np.linalg.norm(v_rej)
    if v_rej_norm > 0:
        v_rej_unit = v_rej / v_rej_norm
    else:
        raise ValueError("The input vector v is parallel to the ones vector, no perpendicular component")

    # Construct the desired vector: h * ones_unit + c * v_rej_unit
    result = h * ones_unit + c * v_rej_unit

    return result


def construct_angle_basis(dim: int, white_point: npt.NDArray, heights: List[float], chromas: List[float]) -> npt.NDArray:
    """In the order of the color basis (BGYR), construct the new basis vectors that they should map to

    Args:
        dim (int): dimension of the basis
        heights (List[float]): height of the basis vectors
        chromas (List[float]): chromas of the basis vectors

    Returns:
        npt.NDArray: the constructed basis vectors
    """
    basis = np.zeros((dim, dim))
    # vec = np.zeros(dim)
    # vec[0] = 1
    # mat = rotation_and_scale_to_point_nd(np.ones(dim), vec)

    # canonical_basis = np.eye(dim)@mat.T

    canonical_basis = np.eye(dim)
    for i in range(dim):
        basis[i] = find_interpolated_vector(canonical_basis[i], heights[i], chromas[i], projection_vector=white_point)
    return basis


def generate_max_angle_points(luminances, chromas, seed=42):
    """
    Places points on a cone in 3D space as far apart in angle as possible,
    constrained by luminance and chroma values.

    Parameters:
        luminances (np.ndarray): Projection onto (1, 0, 0), between 0 and 1.
        chromas (np.ndarray): Distance from the x-axis in 3D space.
        seed (int): For reproducibility.

    Returns:
        np.ndarray: Nx3 array of points on the unit sphere satisfying constraints.
    """
    np.random.seed(seed)
    luminances = np.asarray(luminances)
    chromas = np.asarray(chromas)
    # assert np.allclose(luminances**2 + chromas**2, 1.0, atol=1e-6), \
    #     "Each point must lie on the unit sphere: L^2 + C^2 = 1"

    N = len(luminances)
    points = np.zeros((N, 3))

    # Start with a random orthonormal frame for the plane orthogonal to (1,0,0)
    def random_orthonormal_basis():
        a = np.array([0, 1, 0])
        b = np.cross((1, 0, 0), a)
        b = b / np.linalg.norm(b)
        c = np.cross((1, 0, 0), b)
        return b, c

    y_dir, z_dir = random_orthonormal_basis()

    # Greedily assign angles that maximize angular spread
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    np.random.shuffle(angles)  # randomness for better spreading

    for i in range(N):
        x = luminances[i]
        r = chromas[i]
        y = r * np.cos(angles[i])
        z = r * np.sin(angles[i])
        point = x * np.array([1.0, 0, 0]) + y * y_dir + z * z_dir
        point /= np.linalg.norm(point)  # ensure on unit sphere
        points[i] = point

    return points


def find_transformation_matrix(source_points, target_points):
    """
    Find the transformation matrix that maps source_points to target_points.

    For 2D points with homogeneous coordinates:
    - source_points and target_points should be arrays of shape (n, 3)
    - Each point is represented as [x, y, 1]

    Returns:
    - transformation_matrix: 3x3 matrix that transforms source to target points
    """
    # Ensure we have enough points (need at least 3 for 2D transformation)
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("Need at least 3 point pairs for a 2D transformation")

    # Convert to numpy arrays if they aren't already
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    # Solve the equation: source_points * transformation_matrix = target_points
    # Using the least squares method: T = (A^T A)^(-1) A^T B
    A = source_points
    B = target_points

    # Calculate transformation matrix
    # We're solving for X in A * X = B
    # Least squares solution is X = (A^T A)^(-1) A^T B
    AT = A.T
    ATA = AT @ A
    ATA_inv = np.linalg.inv(ATA)
    ATB = AT @ B
    transformation_matrix = ATA_inv @ ATB

    return transformation_matrix.T


def get_transform_to_angle_basis(basis_vectors: npt.NDArray, white_point: npt.NDArray, heights: List[float], chromas: List[float]) -> npt.NDArray:
    """Get the transformation matrix to the angle basis

    Args:
        basis_vectors (npt.NDArray): the basis vectors written row by row
        white_point (npt.NDArray): the white point
        heights (List[float]): the heights of the basis vectors
        chromas (List[float]): the chromas of the basis vectors

    Returns:
        npt.NDArray: the transformation matrix to the angle basis
    """
    # Construct the new basis vectors that are a certain distance & height from the ones vector
    new_basis = construct_angle_basis(
        basis_vectors.shape[1], white_point, heights, chromas)

    # Compute the transform from a set of points to these new basis vectors
    vecs_to_angle_basis = find_transformation_matrix(basis_vectors, new_basis)

    return vecs_to_angle_basis


def rotation_and_scale_to_point_nd(source_point, target_point=None):
    """
    Finds a transformation matrix that maps source_point to target_point using
    rotation and scaling (no translation) in N-dimensional space.

    If target_point is not specified, it defaults to a vector of ones [1,1,...,1]
    with the same dimension as source_point.

    Parameters:
    - source_point: The original point as an N-dimensional vector
    - target_point: The desired point as an N-dimensional vector (default: [1,1,...,1])

    Returns:
    - transform_matrix: NxN matrix combining rotation and scaling
    """
    # Convert inputs to numpy arrays
    source = np.array(source_point, dtype=float)

    # Set default target if not provided
    if target_point is None:
        target = np.ones(len(source))
    else:
        target = np.array(target_point, dtype=float)

    # Ensure dimensions match
    if source.shape != target.shape:
        raise ValueError(f"Source and target must have same dimension. Got {source.shape} and {target.shape}")

    # Get dimension
    n = len(source)

    # Check for zero vector
    source_norm_value = np.linalg.norm(source)
    if source_norm_value < 1e-10:
        raise ValueError("Source vector has zero length, can't determine rotation")

    # We'll construct a transformation matrix M such that M @ source = target
    # This will be a combination of rotation and scaling

    # Normalize source vector
    u = source / source_norm_value

    # Initialize identity matrix
    M = np.eye(n)

    # Direct construction method:
    # 1. Set the first column to be proportional to target
    scaling_factor = np.linalg.norm(target) / source_norm_value

    # The first column of M will transform u directly to the normalized target
    M[:, 0] = target / source_norm_value

    # 2. Complete the basis using Gram-Schmidt orthogonalization
    basis = [u]

    # Create a basis that includes u
    for i in range(1, n):
        # Start with standard basis vector
        e_i = np.zeros(n)
        e_i[i] = 1.0

        # Make it orthogonal to all previous basis vectors
        for b in basis:
            e_i = e_i - np.dot(e_i, b) * b

        # Normalize if not zero
        norm = np.linalg.norm(e_i)
        if norm > 1e-10:
            e_i = e_i / norm
            basis.append(e_i)
        else:
            # Try a different standard basis vector
            continue

    # If we couldn't find enough basis vectors, try a different approach
    if len(basis) < n:
        # Complete the basis with random vectors
        while len(basis) < n:
            # Generate random vector
            v = np.random.randn(n)

            # Orthogonalize against existing basis
            for b in basis:
                v = v - np.dot(v, b) * b

            # Normalize
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                basis.append(v / norm)

    # Use these basis vectors to create a transformation matrix
    # M will map the source to target, and preserve orthogonality

    # Create an orthogonal matrix Q whose first column is u
    Q = np.column_stack(basis)

    # Create an orthogonal matrix R whose first column is normalized target
    target_norm = target / np.linalg.norm(target)

    # Create a new basis for the target space starting with target_norm
    target_basis = [target_norm]

    # Similar to above, complete the basis
    for i in range(1, n):
        e_i = np.zeros(n)
        e_i[i] = 1.0

        for b in target_basis:
            e_i = e_i - np.dot(e_i, b) * b

        norm = np.linalg.norm(e_i)
        if norm > 1e-10:
            target_basis.append(e_i / norm)

    # Complete with random vectors if needed
    if len(target_basis) < n:
        while len(target_basis) < n:
            v = np.random.randn(n)
            for b in target_basis:
                v = v - np.dot(v, b) * b
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                target_basis.append(v / norm)

    # Create R
    R = np.column_stack(target_basis)

    # The transformation is R @ Q.T
    rotation = R @ Q.T

    # Add the scaling component
    scaling = np.eye(n) * scaling_factor

    # Final transformation
    transform_matrix = rotation @ scaling

    # Verify the transformation
    transformed = transform_matrix @ source
    error = np.linalg.norm(transformed - target)

    if error > 1e-8:
        # If there's an error, try a direct construction method
        # Create a matrix that directly maps source to target
        A = np.eye(n)
        A[:, 0] = target / source_norm_value

        # Calculate an orthogonal matrix that has source/||source|| as its first column
        u_matrix = np.eye(n)
        u_matrix[:, 0] = u

        # The transformation matrix is A @ u_matrix.T
        transform_matrix = A @ u_matrix.T

    return transform_matrix


def rotation_and_scale_to_point(source_point, target_point=(1, 1, 1)):
    """
    Finds a transformation matrix that maps source_point to the target_point using
    rotation and scaling (no translation).
    By default, the target point is (1, 1, 1).

    Parameters:
    - source_point: The original 3D point as [x, y, z]
    - target_point: The desired 3D point as [x, y, z], default is [1, 1, 1]

    Returns:
    - transform_matrix: 3x3 matrix combining rotation and scaling
    """
    # Convert inputs to numpy arrays
    source = np.array(source_point, dtype=float)
    target = np.array(target_point, dtype=float)

    # Check for zero vector
    if np.linalg.norm(source) < 1e-10:
        raise ValueError("Source vector has zero length, can't determine rotation")

    # Normalize source to unit vector
    source_norm = source / np.linalg.norm(source)

    # Calculate the rotation part using the same technique as before
    # First, try to find the rotation from source_norm to the unit vector in target direction
    target_norm = target / np.linalg.norm(target)

    # Find the axis of rotation (cross product)
    v = np.cross(source_norm, target_norm)

    # If vectors are parallel or anti-parallel, we need a different approach
    if np.linalg.norm(v) < 1e-10:
        # Check if they're parallel or anti-parallel
        if np.dot(source_norm, target_norm) > 0:
            # Vectors are parallel, no rotation needed
            R = np.eye(3)
        else:
            # Vectors are anti-parallel, rotate 180 degrees around a perpendicular axis
            # Find any perpendicular vector
            if abs(source_norm[0]) < abs(source_norm[1]) and abs(source_norm[0]) < abs(source_norm[2]):
                perp = np.array([0, -source_norm[2], source_norm[1]])
            elif abs(source_norm[1]) < abs(source_norm[2]):
                perp = np.array([-source_norm[2], 0, source_norm[0]])
            else:
                perp = np.array([-source_norm[1], source_norm[0], 0])

            perp = perp / np.linalg.norm(perp)

            # Rodrigues formula for 180 degree rotation around perp
            R = np.zeros((3, 3))
            R[0, 0] = 2 * perp[0]**2 - 1
            R[0, 1] = 2 * perp[0] * perp[1]
            R[0, 2] = 2 * perp[0] * perp[2]
            R[1, 0] = 2 * perp[1] * perp[0]
            R[1, 1] = 2 * perp[1]**2 - 1
            R[1, 2] = 2 * perp[1] * perp[2]
            R[2, 0] = 2 * perp[2] * perp[0]
            R[2, 1] = 2 * perp[2] * perp[1]
            R[2, 2] = 2 * perp[2]**2 - 1
    else:
        # Normalize the rotation axis
        v = v / np.linalg.norm(v)

        # Calculate cosine of the rotation angle
        c = np.dot(source_norm, target_norm)

        # Calculate sine of the rotation angle
        s = np.linalg.norm(np.cross(source_norm, target_norm))

        # Skew-symmetric cross-product matrix of v
        v_cross = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        # Rodrigues' rotation formula
        R = np.eye(3) + s * v_cross + (1 - c) * (v_cross @ v_cross)

    # Calculate the scaling factor
    # We need to scale the rotated source vector to match the target length
    rotated_source = R @ source

    # Calculate the scaling matrix
    # We want: S * (R * source) = target
    scaling_factor = np.linalg.norm(target) / np.linalg.norm(rotated_source)
    S = scaling_factor * np.eye(3)

    # Combine rotation and scaling: T = S * R
    transform_matrix = S @ R

    return transform_matrix


def visualize_transformation(source_points, target_points, transformed_points):
    """Visualize the original points, target points, and transformed points."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points in red
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2],
               color='red', s=100, label='Source Points')

    # Plot target points in blue
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
               color='blue', s=100, label='Target Points')

    # Plot transformed points in green
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2],
               color='green', s=50, alpha=0.7, label='Transformed Points')

    # Draw lines connecting source to transformed
    for src, trans in zip(source_points, transformed_points):
        ax.plot([src[0], trans[0]], [src[1], trans[1]], [src[2], trans[2]],
                color='gray', linestyle='--', alpha=0.5)

    # Draw lines connecting transformed to target
    for trans, tgt in zip(transformed_points, target_points):
        ax.plot([trans[0], tgt[0]], [trans[1], tgt[1]], [trans[2], tgt[2]],
                color='black', linestyle=':', alpha=0.5)

    # Highlight the (1,0,0) point
    ax.scatter([1], [0], [0], color='yellow', s=200, edgecolor='black', label='(1,0,0)')

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Visualization of Transformation')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    np.set_printoptions(suppress=True, precision=3)

    dim = 3
    basis_vectors = np.eye(dim)
    vec = np.zeros(dim)
    vec[0] = 1
    mat = rotation_and_scale_to_point_nd(np.ones(dim), vec)

    heights = [0.577/2, 0.577, 0.577]
    chromas = [0.816, 0.816/4, 0.816/4]

    print("basis", basis_vectors@mat.T)
    print("ones", np.ones(dim)@mat.T)

    new_basis = construct_angle_basis(
        basis_vectors.shape[1], mat@np.ones(dim), heights, chromas)

    # new_basis = generate_max_angle_points(heights, chromas)
    # print("new_basis", new_basis)

    transform = get_transform_to_angle_basis(basis_vectors@mat.T, np.ones(dim)@mat.T, heights, chromas)

    print("Transform to angle basis:")
    print(transform)

    import matplotlib.pyplot as plt

    # Extract the transformed basis vectors
    transformed_vectors = basis_vectors@mat.T@transform.T

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw arrows from the origin to the transformed basis vectors
    origin = np.zeros(3)
    for vec in transformed_vectors:
        if dim == 4:
            vec = vec[1:]
        ax.quiver(*origin, *vec, color='b', arrow_length_ratio=0.1)

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Transformed Basis Vectors')

    plt.show()
