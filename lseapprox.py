"""
The function lowerapprox(r) finds the optimal solution of
    the best r-term piecewise linear convex 'lower'
    approximation problem described in the paper
    'Tractable Approximate Robust Geometric Programming'
    by Kan-Lin Hsiung, Seung-Jean Kim and Stephen Boyd.

The function upperapprox(r) finds the best r-term piecewise
    linear convex 'upper' approximation based on the
    solution of lowerApprox by adding the error to b.

This Python implementation is a direct translation of their
    Matlab function, which can be found at
    http://web.stanford.edu/~boyd/papers/rgp.html

Dependencies: Python 3.6 and Numpy.

Usage:
    call upperapprox(r) or lowerapprox(r).

Input:
    r, integer, assumed to be >= 2

Output:
    A, Numpy array, describing a r x 2 matrix
    b, Numpy array, describing a r vector
    apx_err, float, the approximation error

Notes:
    1) The original Matlab function was designed for r <= 100,
    the reason for this is not stated, so results for r > 100
    may not be accurate.
    2) results rounded up to 8 decimals.

Author: Marnix Suilen, February 2018
"""

import numpy as np
import math

def upperapprox(r):
    """r-term piecewise linear convex upper approximation to the
    lse-function.

    Args:
        r (int): the degree of approximation. (Assumed to be >= 2)

    Returns:
        A (Numpy array), describing a r x 2 matrix.
        b (Numpy array), describing a r vector.
        apx_err (float), the approximation error.
    """
    A, b, err = lowerapprox(r)
    b = np.add(err,b)

    return A, b, err


def lowerapprox(r):
    """r-term piecewise linear convex lower approximation to the
    lse-function.

    Args:
        r (int): the degree of approximation. (Assumed to be >= 2)

    Returns:
        A (Numpy array), describing a r x 2 matrix.
        b (Numpy array), describing a r vector.
        apx_err (float), the approximation error.
    """

    tol = 1e-14
    uu = math.log(2)
    ll = 0

    if r == 2:
        A = np.array([[0, 1], [1, 0]])
        b = np.array([[0], [0]])
    else:
        while (uu - ll) > tol:
            mid = (uu + ll) / 2
            A, b = invapx(mid)
            if A.shape[0] > r:
                ll = mid
            else:
                uu = mid
        A, b = invapx(uu)
    apx_err = uu

    return A, b, apx_err


def invapx(epsilon):
    """Finds the optimal solution of the 'inverse' best PWL convex lower
    approximation problem defined in the robust GP paper.
    Used as subroutine in approxlse(r).

    Args:
        epsilon (float), given error of lower approximation.

    Returns:
        F, g (Numpy arrays), describes the optimal solution.
    """

    F = np.array([1, 0])
    g = np.array([0])
    xys = np.array([0, -math.inf])
    vert = np.array([0, math.log(math.exp(epsilon) - 1)])

    tol = 1e-12
    leftest_x = math.log(math.exp(epsilon) - 1)
    M = abs(1e2 * leftest_x)
    while True:

        ll = -M
        if vert.ndim == 1:
            uu = vert[0]
            vx = vert[0]
            vy = vert[1]
        else:
            uu = vert[0, 0]
            vx = vert[0, 0]
            vy = vert[0, 1]

        while (uu - ll) > tol:
            midx = (ll + uu) / 2
            midy = math.log(1 - math.exp(midx))
            Fi, gi = addshp(midx, midy)

            if (np.dot(Fi, np.array([[vx], [vy]])) + gi) >= 0:
                uu = midx
            if (np.dot(Fi, np.array([[vx], [vy]])) + gi) <= 0:
                ll = midx

        xys = np.vstack((np.array([midx, midy]), xys))

        F = np.vstack((Fi, F))
        g = np.vstack((gi, g))

        ll = -M
        uu = xys[0, 0]
        while (uu - ll) > tol:
            mid = (ll + uu) / 2
            midexp = math.exp(mid)
            val = (-gi - Fi[0, 0] * mid) / Fi[0, 1]
            try:
                z = math.exp(val)
            except OverflowError:
                z = math.inf
            midval = math.log(midexp + z)

            if midval <= epsilon:
                uu = mid
            if midval >= epsilon:
                ll = mid

        vx = ll
        vy = (-gi - Fi[0, 0] * vx) / Fi[0, 1]

        if vx > leftest_x:
            vert = np.vstack((np.array([vx, vy]), vert))
        else:
            vert = np.vstack((np.array([-gi / Fi[0], 0]), vert))
            F = np.vstack((np.array([0, 1]), F))
            g = np.vstack((np.array([0]), g))
            xys = np.vstack((np.array([-math.inf, 0]), xys))
            break

    F = np.round(F, 8)
    g = np.round(g, 8)
    return F, g


def addshp(x, y):
    """Finds the tangent line of lse(x,y) = 0 at a given point (x,y).

    Args:
        x, y (floats).

    Returns:
        Fi, gi (Numpy arrays), describing the tangent line.
    """

    fgrad = 1/(math.exp(x) + math.exp(y)) * np.array([[math.exp(x)], [math.exp(y)]])
    Fi = fgrad.transpose()
    fgradt = fgrad.transpose()
    xyvect = np.array([[x], [y]])
    gi = np.dot((-1 * fgradt), xyvect)

    return Fi, gi