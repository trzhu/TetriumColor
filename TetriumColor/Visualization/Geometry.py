import numpy as np
from typing import List
import numpy.typing as npt

import open3d as o3d
import glm
from itertools import combinations
import tetrapolyscope as ps


def GetCylinderTransform(endpoints):
    a = endpoints[1]-endpoints[0]
    a = glm.vec3(a[0], a[1], a[2])
    b = glm.vec3(0, 0, 1)

    mat = glm.mat4()
    # translate
    mat = glm.translate(mat, glm.vec3(
        endpoints[0][0], endpoints[0][1], endpoints[0][2]))
    # rotate
    v = glm.cross(b, a)
    if v != glm.vec3(0, 0, 0):
        angle = glm.acos(glm.dot(b, a) / (glm.length(b) * glm.length(a)))
        mat = glm.rotate(mat, angle, v)

    # scale
    scale_factor = glm.length(a)
    mat = glm.scale(mat, glm.vec3(scale_factor, scale_factor, scale_factor))

    return mat


class GeometryPrimitives:

    def __init__(self) -> None:
        self.objects = []

    def add_obj(self, obj: o3d.geometry.TriangleMesh) -> None:
        self.objects.append(obj)

    @staticmethod
    def CollapseMeshObjects(objects: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh()
        for obj in objects:
            mesh += obj
        return mesh

    @staticmethod
    def CreateSphere(radius: float = 0.025, center: List | tuple | npt.NDArray = [0, 0, 0], resolution: float = 20, color: List | tuple | npt.NDArray = [0, 0, 0]) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius, resolution=resolution)
        mesh.translate(center)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([color]*len(mesh.vertices)))
        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def CreateCylinder(endpoints: List | tuple | npt.NDArray, radius: float = 0.025/2, resolution: float = 20, color: List | tuple | npt.NDArray = [0, 0, 0]) -> o3d.geometry.TriangleMesh:
        # canonical cylinder is along z axis with height 1 and centered
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=1, resolution=resolution)
        mesh.translate([0, 0, 1/2])

        matrix = GetCylinderTransform(endpoints)
        mesh.transform(matrix)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([color]*len(mesh.vertices)))
        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def calculate_zy_rotation_for_arrow(vec):
        gamma = np.arctan2(vec[1], vec[0])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        vec = Rz.T @ vec

        beta = np.arctan2(vec[0], vec[2])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        return Rz, Ry

    @staticmethod
    def GetArrow(endpoint: List | tuple | npt.NDArray, origin: List | tuple | npt.NDArray = np.array([0, 0, 0]), scale: float = 1):
        assert (not np.all(endpoint == origin))
        vec = np.array(endpoint) - np.array(origin)
        size = np.sqrt(np.sum(vec**2))
        ratio_cone_cylinder = 0.15
        radius = 60
        ratio_cone_bottom_to_cylinder = 2

        Rz, Ry = GeometryPrimitives.calculate_zy_rotation_for_arrow(vec)
        mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1/radius * ratio_cone_bottom_to_cylinder * scale,
                                                      cone_height=size * ratio_cone_cylinder * scale,
                                                      cylinder_radius=1/radius * scale,
                                                      cylinder_height=size * (1 - ratio_cone_cylinder * scale))
        mesh.rotate(Ry, center=np.array([0, 0, 0]))
        mesh.rotate(Rz, center=np.array([0, 0, 0]))
        mesh.translate(origin)
        return (mesh)

    @staticmethod
    def CreateArrow(endpoints: npt.NDArray, radius: float = 0.025/2, resolution: float = 20, scale: float = 1, color: List | tuple | npt.NDArray = np.array([0, 0, 0])) -> o3d.geometry.TriangleMesh:
        mesh = GeometryPrimitives.GetArrow(
            endpoints[1], endpoints[0], scale=scale)
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([color]*len(mesh.vertices)))
        return mesh

    @staticmethod
    def CreateCoordinateBasis(basis: npt.NDArray, radius: float = 0.025/2, resolution: float = 20, scale: float = 1, color: List | tuple | npt.NDArray = [0, 0, 0]) -> o3d.geometry.TriangleMesh:
        meshes = []
        for i, b in enumerate(basis):
            mesh = GeometryPrimitives.CreateArrow(
                endpoints=np.array([[0, 0, 0], b]), radius=radius, resolution=resolution, color=color[i], scale=scale)
            meshes.append(mesh)
        mesh = GeometryPrimitives.CollapseMeshObjects(meshes)
        return mesh

    @staticmethod
    def CreateMaxBasis(points: npt.NDArray, rgbs: npt.NDArray | List[List], lines: npt.NDArray | List[List | tuple], ball_radius: float = 0.025) -> o3d.geometry.TriangleMesh:
        objs = []
        for rgb, point in zip(rgbs, points):
            objs += [GeometryPrimitives.CreateSphere(
                center=point, color=rgb, radius=ball_radius)]
        for line in lines:
            objs += [GeometryPrimitives.CreateCylinder(
                endpoints=[points[line[0]], points[line[1]]], color=[0, 0, 0])]
        return GeometryPrimitives.CollapseMeshObjects(objs)

    @staticmethod
    def CreateParallelotopeEdges(basis: npt.NDArray, color: List | tuple | npt.NDArray = [0, 0, 0], line_color: List | tuple | npt.NDArray = [0, 0, 0], line_alpha: float = 1) -> o3d.geometry.TriangleMesh:
        dim = basis.shape[1]
        if dim < 3:
            raise ValueError("Basis must be at least 3D")
        elif dim > 4:
            raise ValueError("Basis must be at most 4D")

        alllines = []
        for i in range(dim):
            alllines += list(combinations(range(dim), i + 1))

        lines = []
        for i, x in enumerate(alllines):
            if len(x) <= 1:
                lines += [[0, x[0] + 1]]  # connected to black
            else:  # > 1
                madeupof = list(combinations(x, len(x)-1))
                lines += [[alllines.index(elem) + 1, i + 1]
                          for elem in madeupof]  # connected to each elem

        if dim == 3:
            # # Create points of the parallelepiped
            points = np.array([[0, 0, 0]] + [basis[i].tolist() for i in range(3)] +
                              [(basis[i] + basis[j]).tolist() for i in range(3) for j in range(i+1, 3)] +
                              [(basis[0] + basis[1] + basis[2]).tolist()])
        else:
            # Create points of the 4D paralleletope
            points = np.array([[0, 0, 0, 0]] + [basis[i].tolist() for i in range(4)] +
                              [(basis[i] + basis[j]).tolist() for i in range(4) for j in range(i+1, 4)] +
                              [(basis[i] + basis[j] + basis[k]).tolist() for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)] +
                              [(basis[0] + basis[1] + basis[2] + basis[3]).tolist()])
        rgbs = np.array([color]*len(points))
        return GeometryPrimitives.CreateMaxBasis(points, rgbs, lines, ball_radius=0.025)

    @staticmethod
    def CreateParallelotopeMesh(basis: npt.NDArray, color: List | tuple | npt.NDArray = [0, 0, 0], line_color: List | tuple | npt.NDArray = [0, 0, 0], line_alpha: float = 1) -> o3d.geometry.TriangleMesh:
        dim = basis.shape[1]
        if dim < 3:
            raise ValueError("Basis must be at least 3D")
        elif dim > 4:
            raise ValueError("Basis must be at most 4D")

        if dim == 3:
            # # Create points of the parallelepiped
            points = np.array([[0, 0, 0]] + [basis[i].tolist() for i in range(3)] +
                              [(basis[i] + basis[j]).tolist() for i in range(3) for j in range(i+1, 3)] +
                              [(basis[0] + basis[1] + basis[2]).tolist()])
        else:
            # Create points of the 4D paralleletope
            points = np.array([[0, 0, 0, 0]] + [basis[i].tolist() for i in range(4)] +
                              [(basis[i] + basis[j]).tolist() for i in range(4) for j in range(i+1, 4)] +
                              [(basis[i] + basis[j] + basis[k]).tolist() for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)] +
                              [(basis[0] + basis[1] + basis[2] + basis[3]).tolist()])
        rgbs = np.array([color]*len(points))
        return GeometryPrimitives.Create3DMesh(points, rgbs)

    @staticmethod
    def Create3DMesh(points, rgbs):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        mesh, point_indices = pcl.compute_convex_hull()
        mesh.compute_vertex_normals()

        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs[point_indices])
        return mesh

    @staticmethod
    def ConvertTriangleMeshToPolyscope(name: str, mesh: o3d.geometry.TriangleMesh) -> None:
        ps_mesh = ps.register_surface_mesh(f"{name}", np.asarray(mesh.vertices), np.asarray(
            mesh.triangles), material='wax', smooth_shade=True)
        ps_mesh.add_color_quantity(f"{name}_colors", np.asarray(
            mesh.vertex_colors), defined_on='vertices', enabled=True)
