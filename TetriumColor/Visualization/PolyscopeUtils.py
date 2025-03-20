import numpy as np
import numpy.typing as npt
from typing import List

import tetrapolyscope as ps

from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, DisplayBasisType
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransformWODisplay
from TetriumColor.ColorMath.GamutMath import GenerateMaximalHueSpherePoints

from .Geometry import GeometryPrimitives
from ..Observer import Observer, MaxBasisFactory, GetHeringMatrix
from .Animation import AnimationUtils
from scipy.spatial import ConvexHull
from itertools import combinations


def OpenVideo(filename: str):  # dunno how to type a fd
    """Open a video file for writing.

    Args:
        filename (str): filename

    Returns:
        file descriptor: file descriptor for the video file
    """
    return ps.open_video_file(filename, fps=30)


def CloseVideo(fd) -> None:
    """Close the video file descriptor

    Args:
        fd (dunno): file descriptor
    """
    ps.close_video_file(fd)
    return


def RenderVideo(fd, total_frames: int, target_fps: int = 30):
    """
    Renders a video by updating animations frame by frame.

    :param fd: File descriptor or path to save the video.
    :param total_frames: Total number of frames to render.
    :param target_fps: Target frames per second for the video (default: 30).
    """
    delta_time: float = 1 / target_fps

    for i in range(total_frames):
        # Update animations via AnimationUtils
        AnimationUtils.UpdateObjects(delta_time)

        # Render the current frame to the video
        ps.write_video_frame(fd, transparent_bg=False)


def GetHeringMatrixLumYDir(dim: int) -> npt.NDArray:
    """Get the Hering Matrix with the Luminance Direction as the Y direction

    Args:
        dim (int): dimension of the matrix

    Returns:
        npt.NDArray: Hering Matrix with Luminance Direction as the Y direction
    """
    h_mat = GetHeringMatrix(dim)
    lum_dir = np.copy(h_mat[0])
    h_mat[0] = h_mat[1]
    h_mat[1] = lum_dir
    return h_mat


def Render2DMesh(name: str, points: npt.NDArray, rgb: npt.NDArray) -> float:
    """Create a 2D mesh from a list of vertices (N x 2) and RGB colors (1 x 3)

    Args:
        name (str): Name of the mesh
        points (npt.ArrayLike): N x 2 array of vertices
        rgbs (npt.ArrayLike): 1 x 3 array of RGB colors
    """
    # Compute the convex hull of the points
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]

    # rerun to get the right triangle indices
    hull = ConvexHull(hull_vertices)
    hull_vertices = np.hstack((hull_vertices[hull.vertices], np.zeros((len(hull_vertices), 1))))
    hull_triangles = np.hstack((hull.simplices, np.ones((hull.simplices.shape[0], 1)) * len(hull_vertices)))
    # add center coordinate to make it a triangle fan
    hull_vertices = np.vstack((hull_vertices, [0, 0, 0]))
    # Register the convex hull mesh with Polyscope
    ps_hull_mesh = ps.register_surface_mesh(f"{name}", hull_vertices, hull_triangles, back_face_policy='identical')
    ps_hull_mesh.add_color_quantity(f"{name}_colors",
                                    np.tile(rgb, (len(hull_vertices), 1)), defined_on='vertices', enabled=True)
    return hull.volume


def Render3DMesh(name: str, points: npt.ArrayLike, rgbs: npt.ArrayLike) -> float:
    """Create a 3D mesh from a list of vertices (N x 3) and RGB colors (N x 3)

    Args:
        name (str): Name of the mesh
        points (npt.ArrayLike): N x 3 array of vertices
        rgbs (npt.ArrayLike): N x 3 array of RGB colors
    """
    mesh = GeometryPrimitives.Create3DMesh(points, rgbs)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)
    hull = ConvexHull(points)
    return hull.volume


def Render3DCone(name: str, points: npt.NDArray, line_colors: npt.NDArray, mesh_color: npt.NDArray, mesh_alpha: float = 1, arrow_alpha: float = 1) -> None:
    """Create a 3D cone from a list of vertices (N x 3) and a color (3)

    Args:
        name (str): basename for the cone asset
        points (npt.NDArray): N x 3 array of vertices
        color (npt.ArrayLike): 1 x 3 Array of RGB color or N x 3 array of RGB colors
        mesh_alpha (float, optional): Transparency of the Mesh. Defaults to 1.
        arrow_alpha (float, optional): Transparency of the Arrows. Defaults to 1.
    """
    mesh_colors = np.tile(mesh_color, (len(points)+1, 1))
    if len(line_colors) == 3:
        line_colors = np.tile(line_colors, (len(points), 1))

    center_vertex = np.zeros(points.shape[1])
    vertices = np.concatenate([[center_vertex], points])
    triangles = [[0, i, i + 1] for i in range(1, len(vertices) - 1)]
    ps_mesh = ps.register_surface_mesh(f"{name}", np.asarray(vertices), np.asarray(
        triangles), transparency=mesh_alpha, material='wax', smooth_shade=True)
    ps_mesh.add_color_quantity(f"{name}_colors", mesh_colors, defined_on='vertices', enabled=True)

    arrow_mesh = []
    for i in range(len(points)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [[0, 0, 0], points[i]]), color=np.array([0, 0, 0]), radius=0.025/100)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    ps_arrows = ps.register_surface_mesh(f"{name}_arrows", np.asarray(
        arrow_mesh.vertices), np.asarray(arrow_mesh.triangles), transparency=arrow_alpha, smooth_shade=True)
    ps_arrows.add_color_quantity(f"{name}_arrows_colors", np.asarray(
        arrow_mesh.vertex_colors), defined_on='vertices', enabled=True)

    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points))])
    ps_net = ps.register_curve_network(f"{name}_curve", points, edges)
    ps_net.add_color_quantity(f"{name}_curve_colors", line_colors, defined_on='nodes', enabled=True)


def Render3DLine(name: str, points: npt.NDArray, color: npt.NDArray, radius=None) -> None:
    """Create a 3D line from a list of vertices (N x 3) and a color (3)

    Args:
        name (str): basename for the line asset
        points (npt.NDArray): N x 3 array of vertices
        color (npt.ArrayLike): 1 x 3 Array of RGB color
        line_alpha (float, optional): Transparency of the Line. Defaults to 1.
    """
    if len(color) == 3:
        color = np.tile(color, (len(points), 1))
    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points)-1)])
    ps_net = ps.register_curve_network(f"{name}", points, edges, radius=radius)
    ps_net.add_color_quantity(f"{name}_colors", color, defined_on='nodes', enabled=True)


def RenderSimplexElements(name: str, dim: int, simplex_coords: npt.NDArray, simplex_colors: npt.NDArray, mesh_color=np.array([0.25, 0, 1]) * 0.5, isColored=False):
    """Create a Simplex from the Elements

    Args:
        name (str): Name of the simplex
        dim (int): Dimension of the simplex
        simplex_coords (npt.NDArray): N x 3 array of vertices
        simplex_colors (npt.NDArray): N x 3 array of RGB colors
        isColored (bool, optional): Whether the simplex is colored. Defaults to False.
    """
    names = []
    # GENERATE POINTS
    if dim < 4:
        simplex_coords = np.hstack((simplex_coords, np.zeros((simplex_coords.shape[0], 1))))
    RenderPointCloud("simplex_points", simplex_coords, simplex_colors, radius=0.1)
    names.append(("simplex_points", "point_cloud"))
    # GENERATE EDGES
    edges = list(combinations(range(len(simplex_coords)), 2))

    for i, edge in enumerate(edges):
        if isColored:
            color = np.sum(simplex_colors[list(edge)], axis=0)
        else:
            color = np.zeros(3)
        Render3DLine(f'simplex_edge_{i}', simplex_coords[list(edge)], color=color)
        names.append((f"simplex_edge_{i}", "curve_network"))

    # GENERATE FACES
    faces = list(combinations(range(len(simplex_coords)), 3))
    if dim == 3:
        if isColored:
            color = np.sum(simplex_colors[list(faces[0])], axis=0)
        else:
            color = mesh_color
        RenderTriangle(f'simplex_face', simplex_coords[list(faces[0])], color)
        names.append((f"simplex_face", "surface_mesh"))
        ps.get_surface_mesh(f'simplex_face').set_transparency(0.4)

    elif dim == 4:
        for i, face in enumerate(faces):
            if isColored:
                color = np.sum(simplex_colors[list(faces[0])], axis=0)
            else:
                color = mesh_color
            color = np.sum(simplex_colors[list(face)], axis=0)
            RenderTriangle(f'simplex_face_{i}', simplex_coords[list(face)], color)
            ps.get_surface_mesh(f'simplex_face_{i}').set_transparency(0.4)
            names.append((f"simplex_face_{i}", "surface_mesh"))
    return names


def RenderSphere(name: str, radius):
    mesh = GeometryPrimitives.CreateSphere(radius)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)


def RenderSimplexGamut(name: str, dim: int, points: npt.NDArray, colors: npt.NDArray, mesh_color: npt.NDArray):
    """Generate a Simplex Gamut
    Args:
        name (str): Name of the simplex
        dim (int): Dimension of the observer
        points (npt.NDArray): N x 3 array of vertices
        colors (npt.NDArray): N x 3 array of RGB colors
        mesh_color (npt.NDArray): 1 x 3 array of RGB color
    """
    if dim < 4:
        simplex_coords = np.hstack((points, np.zeros((points.shape[0], 1))))

    def render_edges():
        edges = list(combinations(range(len(points)), 2))
        for i, edge in enumerate(edges):
            Render3DLine(f'{name}_edge_{i}', points[list(edge)], color=np.zeros(3))

    if dim == 3:
        RenderPointCloud(name, points, colors, radius=0.1)
        render_edges()
        RenderTriangle(f'{name}_face', points, mesh_color)
    else:
        RenderPointCloud(name, points, colors, radius=0.1)
        render_edges()
        Render3DMesh(f'{name}_mesh', points, np.tile(mesh_color, (len(points), 1)))
        ps.get_surface_mesh(f'{name}_mesh').set_transparency(0.4)


def RenderTriangle(name: str, points: npt.NDArray, color: npt.NDArray) -> None:
    """Render a 3 point element as a triangle surface mesh

    Args:
        name (str): Name of the triangle
        points (npt.NDArray): 3 x 3 array of vertices
        color (npt.NDArray): 1 x 3 array of RGB color
    """
    if points.shape[0] != 3 or points.shape[1] != 3:
        raise ValueError("Points array must be of shape (3, 3)")

    # Create the triangle mesh
    triangles = np.array([[0, 1, 2]])
    colors = np.tile(color, (3, 1))

    # Register the triangle mesh with Polyscope
    ps_triangle_mesh = ps.register_surface_mesh(name, points, triangles)
    ps_triangle_mesh.add_color_quantity(f"{name}_colors", colors, defined_on='vertices', enabled=True)


def RenderMetamericDirection(name: str, observer: Observer, display_basis: DisplayBasisType,
                             metameric_axis: int, color: npt.ArrayLike, line_alpha: float = 1, scale: float = 1) -> None:

    length = 1 * 0.05
    basisLMSQ = np.zeros((1, observer.dimension))
    basisLMSQ[:, metameric_axis] = 1
    basisLMSQ = basisLMSQ * length
    if display_basis == DisplayBasisType.Cone:
        basisLMSQ = basisLMSQ
    elif display_basis == DisplayBasisType.MaxBasis:
        maxbasis = MaxBasisFactory.get_object(observer)
        T = maxbasis.GetConeToMaxBasisTransform()
        basisLMSQ = basisLMSQ@(T.T)
    else:  # display_basis == DisplayBasisType.Hering
        basisLMSQ = basisLMSQ@GetHeringMatrix(observer.dimension).T

    if observer.dimension > 3 and display_basis != DisplayBasisType.Hering:
        basisLMSQ = (basisLMSQ@GetHeringMatrix(observer.dimension).T)[:, 1:]
    elif observer.dimension > 3 and display_basis == DisplayBasisType.Hering:
        basisLMSQ = basisLMSQ[:, 1:]

    normalizedLMSQ = basisLMSQ[0] / np.linalg.norm(basisLMSQ[0]) * scale
    Render3DLine(name, np.array([np.zeros(3), normalizedLMSQ]), color)


def RenderConeOBS(name: str, observer: Observer) -> None:
    """Create an Object Color Solid in the Cone Basis in Polyscope

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
    """
    chrom_points, rgbs = observer.get_optimal_colors()
    if observer.dimension > 3:
        chrom_points = (GetHeringMatrix(observer.dimension)@chrom_points.T).T[:, 1:]
    elif observer.dimension == 2:
        # First we want to get the boundary, and then get the max pt along the saturation

        # then we want to color in the rest of the solid. We should do this using subspace intersection

        raise NotImplementedError("2D Observer not supported")
    Render3DMesh(f"{name}", chrom_points, rgbs)


def RenderOBSTransform(name: str, observer: Observer, T: npt.NDArray) -> None:
    """Render an Object Color Solid by transforming points before.

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
        T (npt.NDArray): Transformation matrix to apply to the points
    """
    chrom_points, rgbs = observer.get_optimal_colors()
    chrom_points = chrom_points@T.T

    if observer.dimension > 3:
        chrom_points = (GetHeringMatrix(observer.dimension)@chrom_points.T).T[:, 1:]

    Render3DMesh(f"{name}", chrom_points, rgbs)


def RenderMaxBasisOBS(name: str, observer: Observer) -> None:
    """Render Object Color Solid in Max Basis

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
    """
    maxbasis = MaxBasisFactory.get_object(observer)
    T = maxbasis.GetConeToMaxBasisTransform()
    RenderOBSTransform(name, observer, T)


def RenderHeringBasisOBS(name: str, observer: Observer) -> None:
    """Render Object Color Solid in Hering Basis

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
    """
    maxbasis = MaxBasisFactory.get_object(observer)
    T = maxbasis.GetConeToMaxBasisTransform()
    if observer.dimension > 3:  # will be transformed either way in next function call
        T = np.identity(observer.dimension)
    else:
        T = GetHeringMatrixLumYDir(observer.dimension)@T
    RenderOBSTransform(name, observer, T)


def RenderOBS(name: str, observer: Observer, display_basis: DisplayBasisType, num_samples=10000) -> None:
    """Render Object Color Solid in Specified Basis

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
        display_basis (DisplayBasisType): Basis to render the object in
    """
    color_space_transform: ColorSpaceTransform = GetColorSpaceTransformWODisplay(observer)

    boundary_points = GenerateMaximalHueSpherePoints(num_samples, color_space_transform, observer)
    cones = boundary_points@color_space_transform.hering_to_cone.T
    sRGBs = np.clip(cones@color_space_transform.cone_to_sRGB.T, 0, 1)

    T = GetBasisConvert(observer, display_basis)
    chrom_points = cones@T.T
    Render3DMesh(f"{name}", chrom_points, sRGBs)


def RenderMaxBasis(name: str, observer: Observer, display_basis: DisplayBasisType = DisplayBasisType.MaxBasis) -> None:
    """Render Max Basis Objects of Points and Lines - A Luminance Projected Parallelotope.

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
        display_basis (DisplayBasisType, optional): Display Basis to Render in. Defaults to DisplayBasisType.MaxBasis.
    """
    maxbasis = MaxBasisFactory.get_object(observer)
    _, points, rgbs, lines = maxbasis.GetDiscreteRepresentation()
    # go into hering if dim is > 3
    if display_basis == DisplayBasisType.Hering:
        points = points@GetHeringMatrix(observer.dimension).T
    elif display_basis == DisplayBasisType.MaxBasis:
        points = points
    else:  # display_basis == DisplayBasisType.Cone
        T = np.linalg.inv(maxbasis.GetConeToMaxBasisTransform())
        points = points@T.T

    if observer.dimension > 3 and display_basis != DisplayBasisType.Hering:
        points = points@GetHeringMatrix(observer.dimension).T[:, 1:]
    elif observer.dimension > 3 and display_basis == DisplayBasisType.Hering:
        points = points[:, 1:]

    mesh = GeometryPrimitives.CreateMaxBasis(points, rgbs, lines)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)


def RenderDisplayGamut(name: str, basis_vectors: npt.NDArray, T: npt.NDArray = np.eye(3)) -> None:
    """Render Display Gamut in Polyscope

    Args:
        name (str): Name of the gamut to be registered with polyscope
        basis_vectors (npt.NDArray): basis vectors of the paralleletope gamut
    """

    gamut_edges = GeometryPrimitives.CreateParallelotopeEdges(basis_vectors, color=[1, 1, 1], T=T)
    gamut = GeometryPrimitives.CreateParallelotopeMesh(basis_vectors, color=[1, 1, 1], T=T)

    two_mesh = GeometryPrimitives.CollapseMeshObjects([gamut_edges, gamut])
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, two_mesh)


def GetBasisConvert(observer: Observer, display_basis: DisplayBasisType) -> npt.NDArray:
    """Convert 4D points to the basis specified.

    Args:
        observer (Observer): observer object
        display_basis (DisplayBasisType): basis to display points in.

    Returns:
        npt.NDArray: _description_
    """
    T = np.eye(observer.dimension)
    if display_basis == DisplayBasisType.Cone:
        pass
    elif display_basis == DisplayBasisType.MaxBasis:
        maxbasis = MaxBasisFactory.get_object(observer)
        T = maxbasis.GetConeToMaxBasisTransform()@T
    elif display_basis == DisplayBasisType.Hering:
        maxbasis = MaxBasisFactory.get_object(observer)
        T = maxbasis.GetConeToMaxBasisTransform()@T
        if observer.dimension < 4:
            T = GetHeringMatrixLumYDir(observer.dimension)@T
    elif display_basis == DisplayBasisType.ConeHering:
        if observer.dimension < 4:
            T = GetHeringMatrixLumYDir(observer.dimension)@T
    if observer.dimension > 3:
        T = GetHeringMatrix(observer.dimension)[1:]@T
    return T


def ConvertPointsToBasis(points: npt.NDArray, observer: Observer, display_basis: DisplayBasisType) -> npt.NDArray:
    """Convert 4D points to the basis specified.

    Args:
        points (npt.NDArray): points Nx4
        observer (Observer): observer object
        display_basis (DisplayBasisType): basis to display points in.

    Returns:
        npt.NDArray: _description_
    """
    if display_basis == DisplayBasisType.Cone:
        points = points
    elif display_basis == DisplayBasisType.MaxBasis:
        maxbasis = MaxBasisFactory.get_object(observer)
        T = maxbasis.GetConeToMaxBasisTransform()
        points = points@T.T
    elif display_basis == DisplayBasisType.Hering:
        maxbasis = MaxBasisFactory.get_object(observer)
        T = maxbasis.GetConeToMaxBasisTransform()
        if observer.dimension < 4:
            T = GetHeringMatrixLumYDir(observer.dimension)@T
        points = points@T.T
        # projected anyways in the next step
    elif display_basis == DisplayBasisType.ConeHering:
        if observer.dimension < 4:
            points = points@GetHeringMatrixLumYDir(observer.dimension).T

    if observer.dimension > 3:
        return points@GetHeringMatrix(observer.dimension).T[:, 1:]
    elif observer.dimension == 2:  # fill all points with zeroes
        return np.hstack((points, np.zeros((points.shape[0], 1))))
    else:
        return points


def ConvertMaxBasisToDisplayBasis(points: npt.NDArray, observer: Observer, display_basis: DisplayBasisType) -> npt.NDArray:
    if display_basis == DisplayBasisType.Cone:
        maxbasis = MaxBasisFactory.get_object(observer)
        T = maxbasis.GetConeToMaxBasisTransform()
        points = points@np.linalg.inv(T).T
    elif display_basis == DisplayBasisType.MaxBasis:
        points = points
    elif display_basis == DisplayBasisType.Hering:
        if observer.dimension < 4:
            T = GetHeringMatrixLumYDir(observer.dimension)@T
        points = points@T.T
        # projected anyways in the next step
    elif display_basis == DisplayBasisType.ConeHering:
        if observer.dimension < 4:
            points = points@GetHeringMatrixLumYDir(observer.dimension).T

    if observer.dimension > 3:
        return points@GetHeringMatrix(observer.dimension).T[:, 1:]
    elif observer.dimension == 2:  # fill all points with zeroes
        return np.hstack((points, np.zeros((points.shape[0], 1))))
    else:
        return points


def ConvertPointsToChromaticity(points: npt.NDArray, observer: Observer, idxs: List[int]) -> npt.NDArray:
    """Transform coordinates into display chromaticity coordinates

    Args:
        points (npt.NDArray): points to transform (most likely spectral locus)
        observer (Observer): Observer object
        display_basis (DisplayBasisType): displayBasis to transform to before projecting
        idxs (List[int]): indices of the dimension to return

    Returns:
        npt.NDArray: chromaticity points
    """
    maxbasis = MaxBasisFactory.get_object(observer)
    T = maxbasis.GetConeToMaxBasisTransform()
    basis_points = points@T.T
    return (GetHeringMatrix(observer.dimension)@(basis_points.T / (np.sum(basis_points.T, axis=0) + 1e-9)))[idxs].T


def ConvertMaxBasisPointsToChromaticity(points: npt.NDArray, observer: Observer, idxs: List[int]) -> npt.NDArray:
    return (GetHeringMatrix(observer.dimension)@(points.T / (np.sum(points.T, axis=0) + 1e-9)))[idxs].T


def Render4DPointCloud(name: str, points: npt.NDArray, observer: Observer,
                       display_basis: DisplayBasisType = DisplayBasisType.MaxBasis,
                       rgb: npt.NDArray | None = None) -> None:
    """Render a point cloud in Polyscope

    Args:
        name (str): Name of the point cloud
        points (npt.Array): N x 4 array of vertices
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
    """
    points = ConvertPointsToBasis(points, observer, display_basis)
    pcl = ps.register_point_cloud(name, points)
    if rgb is not None:
        pcl.add_color_quantity(f"{name}_colors", rgb, enabled=True)


def RenderPointCloud(name: str, points: npt.NDArray, rgb: npt.NDArray | None = None, radius: float = 0.01) -> None:
    """Render a point cloud in Polyscope

    Args:
        name (str): Name of the point cloud
        points (npt.Array): N x 3 array of vertices
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
        radius (float, optional): Radius of the points. Defaults to 0.01.
    """
    pcl = ps.register_point_cloud(name, points, radius=0.01)
    if rgb is not None:
        pcl.add_color_quantity(f"{name}_colors", rgb, enabled=True)


def RenderSetOfArrows(name: str, endpoints: List[tuple], rgb: npt.NDArray | None = None, radius: float = 0.025/20) -> None:
    """Render a set of basis vectors as arrows in Polyscope

    Args:
        name (str): Name of the basis
        basis (npt.NDArray): N x 3 array of basis vectors
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
    """
    if rgb is None:
        rgb = np.zeros((len(endpoints), 3))

    arrow_mesh = []
    for i in range(len(endpoints)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [endpoints[i][0], endpoints[i][1]]), color=rgb[i], radius=radius)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)


def RenderBasisArrows(name: str, basis: npt.NDArray, rgb: npt.NDArray | None = None, radius: float = 0.025/20) -> None:
    """Render a set of basis vectors as arrows in Polyscope

    Args:
        name (str): Name of the basis
        basis (npt.NDArray): N x 3 array of basis vectors
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
    """
    if rgb is None:
        rgb = np.zeros((len(basis), 3))

    arrow_mesh = []
    for i in range(len(basis)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [[0, 0, 0], basis[i]]), color=rgb[i], radius=radius)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)


def RenderGridOfArrows(name: str):
    """Render a grid of arrows in Polyscope

    Args:
        name (str): Name to be registered with polyscope
    """
    grid_size = 10
    arrow_length = 1.0
    grid_range = np.linspace(-arrow_length/2, arrow_length/2, grid_size)
    arrow_mesh = []
    objs = GeometryPrimitives()
    for x in grid_range:
        for y in grid_range:
            objs.add_obj(GeometryPrimitives.CreateArrow(np.array([[x, y, -arrow_length/2], [x, y, arrow_length/2]])))
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(objs.objects)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)
