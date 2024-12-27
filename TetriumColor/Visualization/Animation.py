import math
import glm
import numpy as np

import tetrapolyscope as ps


class AnimationUtils:
    # Centralized store for objects being animated, where each object is identified by a unique name
    objects = {}

    @staticmethod
    def RotateObject(rotation_matrix, angle_degrees, axis):
        """
        Rotates an object around a specified axis by a given angle.

        Parameters:
            rotation_matrix (glm.mat4): The current rotation matrix of the object.
            angle_degrees (float): The angle of rotation in degrees.
            axis (glm.vec3): The axis of rotation (e.g., x, y, or z).

        Returns:
            glm.mat4: The updated rotation matrix after applying the rotation.
        """
        angle_radians = math.radians(angle_degrees)  # Convert angle to radians
        return rotation_matrix * glm.rotate(glm.mat4(1.0), angle_radians, axis)

    @staticmethod
    def MoveObject(position, velocity, delta_time):
        """
        Updates the position of an object based on its velocity and elapsed time.

        Parameters:
            position (glm.vec3): The current position of the object.
            velocity (glm.vec3): The velocity vector of the object.
            delta_time (float): The time step for the update.

        Returns:
            glm.vec3: The updated position of the object.
        """
        return position + velocity * delta_time

    @staticmethod
    def DecomposeMatrix(matrix):
        """
        Decomposes a transformation matrix into its translation, rotation, and scale components.

        Parameters:
            matrix (glm.mat4): The transformation matrix to decompose.

        Returns:
            tuple: A tuple containing the translation (glm.vec3), rotation (glm.quat), 
                   and scale (glm.vec3) components.
        """
        translation = glm.vec3(matrix[3])  # Extract the translation vector
        scale = glm.vec3(glm.length(matrix[0]), glm.length(matrix[1]), glm.length(matrix[2]))  # Compute scale factors
        rotation_matrix = glm.mat3(matrix)  # Extract rotation matrix by removing translation
        # Normalize rotation matrix by dividing each axis vector by its scale factor
        rotation_matrix[0] /= scale.x
        rotation_matrix[1] /= scale.y
        rotation_matrix[2] /= scale.z
        rotation = glm.quat_cast(rotation_matrix)  # Convert rotation matrix to quaternion
        return translation, rotation, scale

    @staticmethod
    def InterpolateMatrices(matrix1, matrix2, alpha):
        """
        Interpolates between two transformation matrices.

        Parameters:
            matrix1 (glm.mat4): The starting transformation matrix.
            matrix2 (glm.mat4): The target transformation matrix.
            alpha (float): The interpolation factor (0.0 to 1.0).

        Returns:
            glm.mat4: The interpolated transformation matrix.
        """
        # Decompose both matrices into translation, rotation, and scale
        t1, r1, s1 = AnimationUtils.DecomposeMatrix(matrix1)
        t2, r2, s2 = AnimationUtils.DecomposeMatrix(matrix2)

        # Interpolate each component
        interpolated_translation = glm.mix(t1, t2, alpha)
        interpolated_rotation = glm.slerp(r1, r2, alpha)
        interpolated_scale = glm.mix(s1, s2, alpha)

        # Reconstruct the transformation matrix
        translation_matrix = glm.translate(glm.mat4(1.0), interpolated_translation)
        rotation_matrix = glm.mat4_cast(interpolated_rotation)
        scale_matrix = glm.scale(glm.mat4(1.0), interpolated_scale)
        return translation_matrix * rotation_matrix * scale_matrix

    @staticmethod
    def AddObject(name, ps_type, position, velocity, rotation_axis, rotation_speed):
        """
        Adds an object to the centralized object store with initial properties.

        Parameters:
            name (str): Unique name for the object.
            ps_type (str): Polyscope type (e.g., "sphere", "arrow").
            position (list or glm.vec3): Initial position of the object.
            velocity (list or glm.vec3): Initial velocity of the object.
            rotation_axis (list or glm.vec3): Axis of rotation.
            rotation_speed (float): Speed of rotation in degrees per second.
        """
        AnimationUtils.objects[name] = {
            "name": name,
            "type": ps_type,
            "position": glm.vec3(position),
            "velocity": glm.vec3(velocity),
            "rotation_axis": glm.vec3(rotation_axis),
            "rotation_speed": rotation_speed,
            "rotation_matrix": glm.mat4(1.0),  # Initial rotation matrix (identity matrix)
        }

    @staticmethod
    def UpdateObjects(delta_time):
        """
        Updates the position and rotation of all objects in the centralized store.

        Parameters:
            delta_time (float): The time step for the update.
        """
        for name, obj in AnimationUtils.objects.items():
            # Update position based on velocity
            obj["position"] = AnimationUtils.MoveObject(
                obj["position"], obj["velocity"], delta_time
            )

            # Update rotation based on rotation speed and axis
            obj["rotation_matrix"] = AnimationUtils.RotateObject(
                obj["rotation_matrix"],
                obj["rotation_speed"] * delta_time,
                obj["rotation_axis"]
            )

            # Create a transformation matrix combining translation and rotation
            translation_matrix = glm.translate(glm.mat4(1.0), obj["position"])
            transformation_matrix = translation_matrix * obj["rotation_matrix"]

            # Apply the transformation to the Polyscope object (if supported)
            getattr(ps, f"get_{obj['type']}")(obj["name"]).set_transform(transformation_matrix)
