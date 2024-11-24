import numpy as np
import numpy.typing as npt
import math


def SampleAnglesEqually(samples, dim) -> npt.NDArray:
    """
    For a given dimension, sample the sphere equally
    """
    if dim == 2:
        return SampleCircle(samples)
    elif dim == 3:
        return SampleFibonacciSphere(samples)
    else:
        raise NotImplementedError("Only 2D and 3D Spheres are supported")


def SampleCircle(samples=1000) -> npt.NDArray:
    return np.array([[2 * math.pi * (i / float(samples)) for i in range(samples)]]).T


def SampleFibonacciSphere(samples=1000) -> npt.NDArray:
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # stupid but i do not care right now
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arccos(z / r)
        theta = np.arctan2(y, x)
        points.append((theta, phi))

    return np.array(points)
