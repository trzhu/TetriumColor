import numpy as np
import numpy.typing as npt
from typing import List


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
    vec = np.zeros(dim)
    vec[0] = 1
    mat = rotation_and_scale_to_point_nd(np.ones(dim), vec)

    canonical_basis = np.eye(dim)@mat.T
    for i in range(dim):
        basis[i] = find_interpolated_vector(canonical_basis[i], heights[i], chromas[i], projection_vector=white_point)
    return basis


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


if __name__ == "__main__":
    # Example usage
    np.set_printoptions(suppress=True, precision=3)

    dim = 4
    basis_vectors = np.eye(dim)
    vec = np.zeros(dim)
    vec[0] = 1
    mat = rotation_and_scale_to_point_nd(np.ones(dim), vec)

    heights = [0.577, 0.577, 0.577, 0.577]
    chromas = [0.816, 0.816, 0.816, 0.816]

    print("basis", basis_vectors@mat.T)
    print("ones", np.ones(dim)@mat.T)

    new_basis = construct_angle_basis(
        basis_vectors.shape[1], mat@np.ones(dim), heights, chromas)

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
