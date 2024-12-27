from typing import Callable
from matplotlib.pyplot import arrow
import numpy as np
import numpy.typing as npt

import tetrapolyscope as ps

from TetriumColor.Utils.CustomTypes import DisplayBasisType

from .Geometry import GeometryPrimitives
from ..Observer import Observer, MaxBasisFactory, GetHeringMatrix
from .Animation import AnimationUtils


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


def Render3DMesh(name: str, points: npt.ArrayLike, rgbs: npt.ArrayLike) -> None:
    """Create a 3D mesh from a list of vertices (N x 3) and RGB colors (N x 3)

    Args:
        name (str): Name of the mesh
        points (npt.ArrayLike): N x 3 array of vertices
        rgbs (npt.ArrayLike): N x 3 array of RGB colors
    """
    mesh = GeometryPrimitives.Create3DMesh(points, rgbs)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)


def Render3DCone(name: str, points: npt.NDArray, color: npt.NDArray, mesh_alpha: float = 1, arrow_alpha: float = 1) -> None:
    """Create a 3D cone from a list of vertices (N x 3) and a color (3)

    Args:
        name (str): basename for the cone asset
        points (npt.NDArray): N x 3 array of vertices
        color (npt.ArrayLike): 1 x 3 Array of RGB color
        mesh_alpha (float, optional): Transparency of the Mesh. Defaults to 1.
        arrow_alpha (float, optional): Transparency of the Arrows. Defaults to 1.
    """
    center_vertex = np.zeros(points.shape[1])
    vertices = np.concatenate([[center_vertex], points])
    triangles = [[0, i, i + 1] for i in range(1, len(vertices) - 1)]
    ps_mesh = ps.register_surface_mesh(f"{name}", np.asarray(vertices), np.asarray(
        triangles), transparency=mesh_alpha, material='wax', smooth_shade=True)
    ps_mesh.add_color_quantity(f"{name}_colors", np.tile(
        color, (len(vertices), 1)), defined_on='vertices', enabled=True)

    arrow_mesh = []
    for i in range(len(points)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [[0, 0, 0], points[i]]), color=color)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    ps_arrows = ps.register_surface_mesh("cone_arrows", np.asarray(
        arrow_mesh.vertices), np.asarray(arrow_mesh.triangles), transparency=arrow_alpha, smooth_shade=True)
    ps_arrows.add_color_quantity("cone_arrows_colors", np.asarray(
        arrow_mesh.vertex_colors), defined_on='vertices', enabled=True)

    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points))])
    ps_net = ps.register_curve_network(f"{name}_curve", points, edges)
    ps_net.add_color_quantity(f"{name}_curve_colors", np.tile(
        color, (len(points), 1)), defined_on='nodes', enabled=True)


def Render3DLine(name: str, points: npt.NDArray, color: npt.ArrayLike, line_alpha: float = 1) -> None:
    """Create a 3D line from a list of vertices (N x 3) and a color (3)

    Args:
        name (str): basename for the line asset
        points (npt.NDArray): N x 3 array of vertices
        color (npt.ArrayLike): 1 x 3 Array of RGB color
        line_alpha (float, optional): Transparency of the Line. Defaults to 1.
    """
    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points))])
    ps_net = ps.register_curve_network(f"{name}_curve", points, edges)
    ps_net.add_color_quantity(f"{name}_curve_colors", np.tile(
        color, (len(points), 1)), defined_on='nodes', enabled=True)


def RenderConeOBS(name: str, observer: Observer) -> None:
    """Create an Object Color Solid in the Cone Basis in Polyscope

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
    """
    chrom_points, rgbs = observer.get_optimal_colors()
    if observer.dimension > 3:
        chrom_points = (GetHeringMatrix(observer.dimension)@chrom_points.T).T[:, 1:]

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
        T = GetHeringMatrix(observer.dimension)@T
    RenderOBSTransform(name, observer, T)


def RenderOBS(name: str, observer: Observer, display_basis: DisplayBasisType) -> None:
    """Render Object Color Solid in Specified Basis

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
        display_basis (DisplayBasisType): Basis to render the object in
    """
    if display_basis == DisplayBasisType.Cone:
        RenderConeOBS(name, observer)
    elif display_basis == DisplayBasisType.MaxBasis:
        RenderMaxBasisOBS(name, observer)
    else:
        RenderHeringBasisOBS(name, observer)


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


def RenderDisplayGamut(name: str, basis_vectors: npt.NDArray):
    """Render Display Gamut in Polyscope

    Args:
        name (str): Name of the gamut to be registered with polyscope
        basis_vectors (npt.NDArray): basis vectors of the paralleletope gamut
    """
    gamut_edges = GeometryPrimitives.CreateParallelotopeEdges(basis_vectors, color=[1, 1, 1])
    gamut = GeometryPrimitives.CreateParallelotopeMesh(basis_vectors, color=[1, 1, 1])

    two_mesh = GeometryPrimitives.CollapseMeshObjects([gamut_edges, gamut])
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, two_mesh)


def RenderPointCloud(name: str, points: npt.NDArray, rgb: npt.NDArray | None = None) -> None:
    """Render a point cloud in Polyscope

    Args:
        name (str): Name of the point cloud
        points (npt.Array): N x 3 array of vertices
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
    """
    pcl = ps.register_point_cloud(name, points)
    if rgb is not None:
        pcl.add_color_quantity(f"{name}_colors", rgb, enabled=True)


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
