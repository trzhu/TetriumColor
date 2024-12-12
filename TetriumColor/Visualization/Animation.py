import glm
import math


class AnimationUtils:
    @staticmethod
    def RotateObject(rotation_matrix, angle_degrees, axis):
        """
        Rotates an object around a given axis.
        :param rotation_matrix: The initial rotation matrix (glm.mat4).
        :param angle_degrees: The angle of rotation in degrees.
        :param axis: The axis of rotation as a glm.vec3 (e.g., glm.vec3(1, 0, 0) for x-axis).
        :return: Updated rotation matrix (glm.mat4).
        """
        angle_radians = math.radians(angle_degrees)
        return rotation_matrix * glm.rotate(glm.mat4(1.0), angle_radians, axis)

    @staticmethod
    def MoveObject(position, velocity, delta_time):
        """
        Moves an object based on velocity and elapsed time.
        :param position: Current position as a glm.vec3.
        :param velocity: Velocity vector (glm.vec3).
        :param delta_time: Elapsed time since the last frame.
        :return: Updated position (glm.vec3).
        """
        return position + velocity * delta_time

    @staticmethod
    def DecomposeMatrix(matrix):
        """
        Decomposes a glm.mat4 into translation, rotation (quaternion), and scale.
        :param matrix: The input transformation matrix (glm.mat4).
        :return: (translation, rotation, scale)
        """
        translation = glm.vec3(matrix[3])
        scale = glm.vec3(glm.length(matrix[0]), glm.length(
            matrix[1]), glm.length(matrix[2]))
        rotation_matrix = glm.mat3(matrix)

        # Normalize the rotation matrix (remove scale influence)
        rotation_matrix[0] /= scale.x
        rotation_matrix[1] /= scale.y
        rotation_matrix[2] /= scale.z

        rotation = glm.quat_cast(rotation_matrix)
        return translation, rotation, scale

    @staticmethod
    def InterpolateMatrices(matrix1, matrix2, alpha):
        """
        Interpolates between two glm.mat4 transformation matrices.
        :param matrix1: The starting matrix (glm.mat4).
        :param matrix2: The ending matrix (glm.mat4).
        :param alpha: Interpolation factor (0.0 = matrix1, 1.0 = matrix2).
        :return: Interpolated matrix (glm.mat4).
        """
        # Decompose both matrices
        t1, r1, s1 = AnimationUtils.DecomposeMatrix(matrix1)
        t2, r2, s2 = AnimationUtils.DecomposeMatrix(matrix2)

        # Interpolate translation, rotation, and scale
        interpolated_translation = glm.mix(t1, t2, alpha)
        interpolated_rotation = glm.slerp(r1, r2, alpha)
        interpolated_scale = glm.mix(s1, s2, alpha)

        # Reconstruct the matrix
        translation_matrix = glm.translate(
            glm.mat4(1.0), interpolated_translation)
        rotation_matrix = glm.mat4_cast(interpolated_rotation)
        scale_matrix = glm.scale(glm.mat4(1.0), interpolated_scale)

        return translation_matrix * rotation_matrix * scale_matrix
