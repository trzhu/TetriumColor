from typing import Callable
from colour.recovery.jakob2019 import dimensionalise_coefficients
from matplotlib.pyplot import arrow
import numpy as np
import numpy.typing as npt

import tetrapolyscope as ps

from .Geometry import GeometryPrimitives
from ..Observer import Observer, MaxBasisFactory, GetHeringMatrix


def OpenVideo(filename):
    return ps.open_video_file(filename, fps=30)


def CloseVideo(fd):
    ps.close_video_file(fd)
    return


def RenderVideo(updateAnimation: Callable[[float], None], fd, total_frames: int, target_fps: int = 30):
    delta_time: float = 1 / target_fps
    for i in range(total_frames):
        updateAnimation(delta_time)
        ps.write_video_frame(fd, transparent_bg=False)


def Render3DMesh(name: str, points: npt.ArrayLike, rgbs: npt.ArrayLike, mesh_alpha: float = 1) -> None:
    """Create a 3D mesh from a list of vertices (N x 3) and RGB colors (N x 3)

    Args:
        name (str): Name of the mesh
        points (npt.ArrayLike): N x 3 array of vertices
        rgbs (npt.ArrayLike): N x 3 array of RGB colors

    Returns:
        _type_: _description_
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
    chrom_points, rgbs = observer.get_optimal_colors()
    if observer.dimension > 3:
        chrom_points = (GetHeringMatrix(observer.dimension)@chrom_points.T).T[:, 1:]

    Render3DMesh(f"{name}_mesh", chrom_points, rgbs)


def RenderMaxBasisOBS(name: str, observer: Observer) -> None:
    chrom_points, rgbs = observer.get_optimal_colors()
    maxbasis = MaxBasisFactory.get_object(observer)
    T = maxbasis.get_transformation_matrix()
    chrom_points = chrom_points@T.T

    if observer.dimension > 3:
        chrom_points = (GetHeringMatrix(observer.dimension)@chrom_points.T).T[:, 1:]

    Render3DMesh(f"{name}_mesh", chrom_points, rgbs)


def RenderDisplayGamut(name: str, basis_vectors: npt.NDArray):
    gamut_edges = GeometryPrimitives.CreateParallelotopeEdges(basis_vectors, color=[1, 1, 1])
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name + "_edges", gamut_edges)

    gamut = GeometryPrimitives.CreateParallelotopeMesh(basis_vectors, color=[1, 1, 1])
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name + "_mesh", gamut)


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
    grid_size = 10
    arrow_length = 1.0
    grid_range = np.linspace(-arrow_length/2, arrow_length/2, grid_size)
    arrow_mesh = []
    for x in grid_range:
        for y in grid_range:
            arrow = GeometryPrimitives.CreateArrow(np.array([[x, y, -arrow_length/2], [x, y, arrow_length/2]]))
            GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow)
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)
