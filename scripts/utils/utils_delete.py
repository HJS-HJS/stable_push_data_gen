import numpy as np
from typing import List

def fibonacci_sphere(samples: int=2000) -> List[float, float, float]:
    """generate ICR in fibonacci sphere

    Args:
        samples (int, optional): Number to make circle velocity samples. Defaults to 2000.

    Returns:
        List[float, float, float]: Velocities list with [x, y, z]
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples//2):
        x = 1 - (i / float(samples - 1)) * 2  # x goes from 1 to -1
        radius = np.sqrt(1 - x * x)  # radius at x

        theta = phi * i  # golden angle increment

        y = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    return np.array(points)

def linear_velocities(samples: int=200) -> List[float, float, float]:
    """_summary_

    Args:
        samples (int, optional): _description_. Defaults to 2000.

    Returns:
        List[float, float, float]: _description_
    """

    log_radius = np.linspace(np.log10(1e-4), np.log10(100), 200)
    radius_positive = np.power(10, log_radius)
    radius = np.concatenate((np.flip(-radius_positive), radius_positive))
    print(radius)
    icrs = np.vstack((np.zeros_like(radius), radius)).T
    print(icrs)
    return icrs



def velocity2icr(velocity: List[float, float, float]) -> List[float, float]:
    """Calculate ICR (Instantaneous Center of Rotation) for each velocity.

    Args:
        velocity (List[float, float, float]): Velocities list with [x, y, z]

    Returns:
        List[float, float]: List of ICR [x/z, y/z]
    """
    vx, vy, w = velocity[:,0], velocity[:,1], velocity[:,2]
    ICRs = []
    for i in range(len(vx)):
        if w[i] == 0:
            w[i] = 1e-6
        icr= np.array([-vy[i] / w[i], vx[i] / w[i]])
        ICRs.append(icr)
        
    return np.array(ICRs)
    vx, vy, w = velocity[:,0], velocity[:,1], velocity[:,2]
    ICRs = []
    for i in range(len(vx)):
        if w[i] == 0:
            w[i] = 1e-6
        icr= np.array([-vy[i] / w[i], vx[i] / w[i]])
        ICRs.append(icr)
        
    return np.array(ICRs)