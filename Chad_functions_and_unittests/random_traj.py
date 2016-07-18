import numpy as np
import os
import csv
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt


def randaniso(b, s, f, p, xs, ys, zs):
    """
    Builds a single random anisotropic trajectory without using spherical
    coordinates, as randtraj does. Anisotropic behavior is determined by the
    "stretching" coefficients xs ys zs.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    xs: x stretching coefficient
    ys: y stretching coefficient
    zs: z stretching coefficient

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = (ttraject[num-1, 5] + ttraject[num, 2])*xs
        ttraject[num, 6] = (ttraject[num-1, 6] + ttraject[num, 19])*ys
        ttraject[num, 7] = (ttraject[num-1, 7] + ttraject[num, 20])*zs

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                                    (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = np.sqrt((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def randconv(b, s, f, p, xs, ys, zs):
    """
    Builds a single random trajectory with a convection term. Magnitude of the
    convection term is determined by xc, yc, and zc.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    xc: x convection
    yc: y convection
    zc: z convection

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2] + xc
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19] + yc
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20] + zc

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                                    (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = np.sqrt((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def multrandaniso(b, s, f, p, xs, ys, zs):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randaniso(b, s, f, 1, xs, ys, zs)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randaniso(b, s, f, counter, xs, ys, zs), axis=0)

    return one


def multrandconv(b, s, f, p, xc, yc, zc):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randconv(b, s, f, 1, xc, yc, zc)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randconv(b, s, f, counter, xc, yc, zc), axis=0)

    return one
