import math
import glm
import numpy as np

import tetrapolyscope as ps


class AnimationUtils:
    objects = {}  # Centralized object store

    @staticmethod
    def RotateObject(rotation_matrix, angle_degrees, axis):
        angle_radians = math.radians(angle_degrees)
        return rotation_matrix * glm.rotate(glm.mat4(1.0), angle_radians, axis)

    @staticmethod
    def MoveObject(position, velocity, delta_time):
        return position + velocity * delta_time

    @staticmethod
    def DecomposeMatrix(matrix):
        translation = glm.vec3(matrix[3])
        scale = glm.vec3(glm.length(matrix[0]), glm.length(matrix[1]), glm.length(matrix[2]))
        rotation_matrix = glm.mat3(matrix)
        rotation_matrix[0] /= scale.x
        rotation_matrix[1] /= scale.y
        rotation_matrix[2] /= scale.z
        rotation = glm.quat_cast(rotation_matrix)
        return translation, rotation, scale

    @staticmethod
    def InterpolateMatrices(matrix1, matrix2, alpha):
        t1, r1, s1 = AnimationUtils.DecomposeMatrix(matrix1)
        t2, r2, s2 = AnimationUtils.DecomposeMatrix(matrix2)
        interpolated_translation = glm.mix(t1, t2, alpha)
        interpolated_rotation = glm.slerp(r1, r2, alpha)
        interpolated_scale = glm.mix(s1, s2, alpha)
        translation_matrix = glm.translate(glm.mat4(1.0), interpolated_translation)
        rotation_matrix = glm.mat4_cast(interpolated_rotation)
        scale_matrix = glm.scale(glm.mat4(1.0), interpolated_scale)
        return translation_matrix * rotation_matrix * scale_matrix

    @staticmethod
    def AddObject(name, ps_type, position, velocity, rotation_axis, rotation_speed):
        AnimationUtils.objects[name] = {
            "name": name,
            "type": ps_type,
            "position": glm.vec3(position),
            "velocity": glm.vec3(velocity),
            "rotation_axis": glm.vec3(rotation_axis),
            "rotation_speed": rotation_speed,
            "rotation_matrix": glm.mat4(1.0),
        }

    @staticmethod
    def UpdateObjects(delta_time):
        for name, obj in AnimationUtils.objects.items():
            # Update position
            obj["position"] = AnimationUtils.MoveObject(
                obj["position"], obj["velocity"], delta_time
            )

            # Update rotation
            obj["rotation_matrix"] = AnimationUtils.RotateObject(
                obj["rotation_matrix"],
                obj["rotation_speed"] * delta_time,
                obj["rotation_axis"]
            )

            # Translation matrix
            translation_matrix = glm.translate(glm.mat4(1.0), obj["position"])
            transformation_matrix = translation_matrix * obj["rotation_matrix"]

            # If Polyscope supports dynamic updates, apply transformation_matrix here
            getattr(ps, f"get_{obj['type']}")(obj["name"]).set_transform(transformation_matrix)
