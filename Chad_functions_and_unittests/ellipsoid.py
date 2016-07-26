from __future__ import division
import numpy as np
import numpy.linalg as la


def mvee(points, tol=0.001):
    N, d = points.shape

    Q = np.zeros([N, d+1])
    Q[:, 0:d] = points[0:N, 0:d]
    Q[:, d] = np.ones([1, N])

    Q = np.transpose(Q)
    points = np.transpose(points)
    count = 1
    err = 1
    u = (1/N) * np.ones(shape=(N,))

    while err > tol:

        X = np.dot(np.dot(Q, np.diag(u)), np.transpose(Q))
        M = np.diag(np.dot(np.dot(np.transpose(Q), la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1)/((d+1)*(M[jdx] - 1))
        new_u = (1 - step_size)*u
        new_u[jdx] = new_u[jdx] + step_size
        count = count + 1
        err = la.norm(new_u - u)
        u = new_u

    U = np.diag(u)
    c = np.dot(points, u)
    A = (1/d) * la.inv(np.dot(np.dot(points, U), np.transpose(points)) - np.dot(c, np.transpose(c)))
    return A, np.transpose(c)


def ellipse(u, v):
    x = rx*cos(u)*cos(v)
    y = ry*sin(u)*cos(v)
    z = rz*sin(v)
    return x, y, z


def extrema(traj, n1, n2, frames):
    """
    n1: column containing particle data
    n2: start column of trajectory data
    frames: number of frames per trajectory

    extrema: array containing xyz data of start and end points of trajectories
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+3]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

    pathmax = np.zeros((total-1, 3))
    pathmin = np.zeros((total-1, 3))
    # maxi = dict()

    for num in range(1, total):

        # maxi[num] = path[num].argmax(axis=0)
        pathmax[num-1, :] = path[num][0, :]
        pathmin[num-1, :] = path[num][frames-2, :]

    extrema = np.append(pathmax, pathmin, axis=0)
    return extrema


def enclosed_MSD(traj, n1, n2, n3, frames):
    """
    Creates a set of 6 points that will be used to form a diffusion ellipse
    based on MSD data. DEFUNCT

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    raw2DMSDs = traj[:, n3:n3+4]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw2DMSDs[min1:max1, :])
        time[num] = (rawtime[min1:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    MMSD = MMSD[frames - 2, :]
    disp = np.sqrt(MMSD)

    pts = np.array([[disp[1],  disp[1],  0],
                    [-disp[1], -disp[1], 0],
                    [disp[1],  -disp[1], 0],
                    [-disp[1], disp[1],  0],
                    [disp[2],  0,        disp[2]],
                    [-disp[2], 0,        -disp[2]],
                    [-disp[2], 0,        disp[2]],
                    [disp[2],  0,        -disp[2]],
                    [0,       disp[3],   disp[3]],
                    [0,       -disp[3],  -disp[3]],
                    [0,       -disp[3],  disp[3]],
                    [0,       disp[3],   -disp[3]]])

    return pts


def maxtraj(traj, n1, n2):
    """
    Creates a 3-column matrix of xyz data of xyzmaxes and xyzmins from traj
    ectory dataset.

    traj: original trajectory dataset
    n1: particle numbers (0)
    n2: xyz data (8)
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+3]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

        pathmax = np.zeros((total-1, 3))
        pathmin = np.zeros((total-1, 3))

        maxi = dict()
        mini = dict()
        maxes = np.zeros((6*(total-1), 4))

    for num in range(1, total):

        maxi[num] = path[num].argmax(axis=0)
        mini[num] = path[num].argmax(axis=0)
        maxes[num-1, 0:3] = path[num][maxi[num][0], :]
        maxes[(total-1) + num-1, 0:3] = path[num][mini[num][0], :]
        maxes[2*(total-1) + num-1, 0:3] = path[num][maxi[num][1], :]
        maxes[3*(total-1) + num-1, 0:3] = path[num][mini[num][1], :]
        maxes[4*(total-1) + num-1, 0:3] = path[num][maxi[num][2], :]
        maxes[5*(total-1) + num-1, 0:3] = path[num][mini[num][2], :]

    for num in range(1, 6*(total-1)):

        maxes[num-1, 3] = (maxes[num, 0])**2 + (maxes[num, 1])**2 + (maxes[num, 2])**2

    maxes = sorted(maxes, lambda x:x[3])

    return maxes
