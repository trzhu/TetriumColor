import math
import glm
import tetrapolyscope as ps


class AnimationUtils:
    objects = {}

    @staticmethod
    def RotateQuaternion(current_quat, angle_degrees, axis):
        """
        Rotates a quaternion around a specified world-space axis.

        Parameters:
            current_quat (glm.quat): The current orientation of the object.
            angle_degrees (float): Rotation angle in degrees.
            axis (glm.vec3): World-space axis of rotation.

        Returns:
            glm.quat: The updated quaternion after applying rotation.
        """
        angle_radians = math.radians(angle_degrees)
        delta_quat = glm.angleAxis(angle_radians, glm.normalize(axis))
        return glm.quat(delta_quat) * glm.quat(current_quat)

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
    def AddObject(name, ps_type, position, velocity, rotation_axis, rotation_speed, rotation_matrix=None):
        """
        Adds an object with optional initial rotation matrix (glm.mat4).
        """
        initial_quat = (
            glm.quat_cast(rotation_matrix)
            if rotation_matrix is not None
            else glm.quat()
        )
        AnimationUtils.objects[name] = {
            "name": name,
            "type": ps_type,
            "position": glm.vec3(position),
            "velocity": glm.vec3(velocity),
            "rotation_axis": glm.vec3(rotation_axis),
            "rotation_speed": rotation_speed,
            "rotation_quat": initial_quat
        }

    @staticmethod
    def UpdateObjects(delta_time):
        for name, obj in AnimationUtils.objects.items():
            obj["position"] = AnimationUtils.MoveObject(
                obj["position"], obj["velocity"], delta_time
            )

            obj["rotation_quat"] = AnimationUtils.RotateQuaternion(
                obj["rotation_quat"],
                obj["rotation_speed"] * delta_time,
                obj["rotation_axis"]
            )

            translation_matrix = glm.translate(glm.mat4(1.0), obj["position"])
            rotation_matrix = glm.mat4_cast(obj["rotation_quat"])
            transformation_matrix = translation_matrix * rotation_matrix

            getattr(ps, f"get_{obj['type']}")(obj["name"]).set_transform(transformation_matrix)
