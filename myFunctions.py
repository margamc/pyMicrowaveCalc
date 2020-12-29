# -----------------------------------------------------------
# Functions for Microwave Network Analysis
#
# 2020 - Margarita Martínez-Coves
# Master Thesis: "Estudio de estructuras con retardo de grupo
# negativo en tecnología microstrip"
# Universitat d'Alacant (Spain)
# email martinezcoves.m@gmail.com
# -----------------------------------------------------------

import numpy as np
import warnings
from scipy.signal import find_peaks

# The following three functions: fCalcSPMod1, fCalcSPMod2 and fCalcSPMod3 are particular for each Model of this project.
# They all work in a similar way, but for different number or configuration of variables. In fact, fCalcSPMod3 and
# fCalcSPMod2 could work for Model 1 if no-used variables are defined with a fixed value. I had implemented particular
# functions for each Model in order to speed up the process.

# Only fCalcMod3 is explained by comments, 1 and 2 are equivalent.
def fCalcSPMod3(t2, r1, z1, r2, r3, z2, r4, z3, r5, z0, fn):
    # Calculation of equivalent ABCD matrix for each branch (parallel). Each branch is composed by Resistors and
    # Transmission Lines in Serie.
    # _____________ R1 _ Z1 _ R2 _____________  [PATH 1]
    #   |                                  |
    #   |_ R3 _ Z2(t2) _ R4 _ Z3(t2) _ R5 _|    [PATH 2]
    warnings.filterwarnings("ignore")
    # Admittance for each TL
    y1 = 1 / z1 if (z1 != 0) else 0
    y2 = 1 / z2 if (z2 != 0) else 0
    y3 = 1 / z3 if (z3 != 0) else 0
    # Admittance for ports
    y0 = 1 / z0 if (z0 != 0) else 0

    # THETA is the normalized electrical length (theta = beta * l) for each normalized frequency and t2, which defines
    # the relation between TL lengths of the same branch.
    lenfn = fn.size
    THETA1A = np.pi * fn                    # THETA1A is for Z1 (always length = pi)
    THETA2A = t2 * fn                       # THETA2A is for Z2 (variable length = t2)
    THETA2B = ((2 * np.pi) - t2) * fn       # THETA2B is for Z3 (variable length = 2pi - t2)

    # Following lines are necessary to create Resistance equivalent ABCD matrix ( [1, r ; 0, 1]), but multidimensional
    # in order to be able to compute all frequency values through a matrix multiplication. So each R matrix is an array
    # with the same 2x2 ABCD matrix for each frequency value (lenfn) --> ( lenfn x 2 x 2 ). lenfn must be the
    # 1st dimension in Python.
    R1 = np.array([[1, r1], [0, 1]])
    R2 = np.array([[1, r2], [0, 1]])
    R3 = np.array([[1, r3], [0, 1]])
    R4 = np.array([[1, r4], [0, 1]])
    R5 = np.array([[1, r5], [0, 1]])
    R1 = np.tile(R1, (lenfn, 1, 1))
    R2 = np.tile(R2, (lenfn, 1, 1))
    R3 = np.tile(R3, (lenfn, 1, 1))
    R4 = np.tile(R4, (lenfn, 1, 1))
    R5 = np.tile(R5, (lenfn, 1, 1))
    # Creation of TL equivalent ABCD matrix ( [cos(bl), j*Z*sin(bl) ; j*Y*sin(bl), cos(bl)] ) for each frequency value.
    # Size of each TL matrix is lenfn x 2 x 2. lenfn must be the 1st dimension too.
    TL1 = np.stack([np.stack([np.cos(THETA1A), 1j * z1 * np.sin(THETA1A)], axis=-1),
                    np.stack([1j * y1 * np.sin(THETA1A), np.cos(THETA1A)], axis=-1)], axis=1)
    TL2 = np.stack([np.stack([np.cos(THETA2A), 1j * z2 * np.sin(THETA2A)], axis=-1),
                    np.stack([1j * y2 * np.sin(THETA2A), np.cos(THETA2A)], axis=-1)], axis=1)
    TL3 = np.stack([np.stack([np.cos(THETA2B), 1j * z3 * np.sin(THETA2B)], axis=-1),
                    np.stack([1j * y3 * np.sin(THETA2B), np.cos(THETA2B)], axis=-1)], axis=1)
    # Matrix multiplication for each path. Python is able to compute [2x2] * [2x2] for each first dimension (lenfn).
    # Size of each PATH (ABCD matrix) is lenfn x 2 x 2. So, it is the ABCD matrix equivalent for each frequency value.
    PATH1 = R1 @ TL1 @ R2
    PATH2 = R3 @ TL2 @ R4 @ TL3 @ R5
    warnings.filterwarnings("default")
    return fPath2SP(PATH1, PATH2, y0)


def fCalcSPMod2(t1, r1, z1, r2, z2, r3, r4, z3, r5, z0, fn):
    # ____ R1 _ Z1(t1) _ R2 _ Z2(t1) _ R3 ____  [PATH 1]
    #   |                                  |
    #   |__________ R4 _ Z3 _ R5 __________|    [PATH 2]
    warnings.filterwarnings("ignore")
    y1 = 1 / z1 if (z1 != 0) else 0
    y2 = 1 / z2 if (z2 != 0) else 0
    y3 = 1 / z3 if (z3 != 0) else 0

    y0 = 1 / z0 if (z0 != 0) else 0

    lenfn = fn.size
    THETA1A = t1 * fn                       # THETA1A is for Z1
    THETA1B = (np.pi - t1) * fn             # THETA1B is for Z2
    THETA2A = 2 * np.pi * fn                # THETA2A is for Z3
    R1 = np.array([[1, r1], [0, 1]])
    R2 = np.array([[1, r2], [0, 1]])
    R3 = np.array([[1, r3], [0, 1]])
    R4 = np.array([[1, r4], [0, 1]])
    R5 = np.array([[1, r5], [0, 1]])
    R1 = np.tile(R1, (lenfn, 1, 1))
    R2 = np.tile(R2, (lenfn, 1, 1))
    R3 = np.tile(R3, (lenfn, 1, 1))
    R4 = np.tile(R4, (lenfn, 1, 1))
    R5 = np.tile(R5, (lenfn, 1, 1))
    TL1 = np.stack([np.stack([np.cos(THETA1A), 1j * z1 * np.sin(THETA1A)], axis=-1),
                    np.stack([1j * y1 * np.sin(THETA1A), np.cos(THETA1A)], axis=-1)], axis=1)
    TL2 = np.stack([np.stack([np.cos(THETA1B), 1j * z2 * np.sin(THETA1B)], axis=-1),
                    np.stack([1j * y2 * np.sin(THETA1B), np.cos(THETA1B)], axis=-1)], axis=1)
    TL3 = np.stack([np.stack([np.cos(THETA2A), 1j * z3 * np.sin(THETA2A)], axis=-1),
                    np.stack([1j * y3 * np.sin(THETA2A), np.cos(THETA2A)], axis=-1)], axis=1)
    PATH1 = R1 @ TL1 @ R2 @ TL2 @ R3
    PATH2 = R4 @ TL3 @ R5
    warnings.filterwarnings("default")
    return fPath2SP(PATH1, PATH2, y0)


def fCalcSPMod1(r1, z1, r2, r3, z2, r4, z0, fn):
    # _____________ R1 _ Z1 _ R2 _____________  [PATH 1]
    #   |                                  |
    #   |__________ R3 _ Z2 _ R4 __________|    [PATH 2]
    warnings.filterwarnings("ignore")
    y1 = 1 / z1 if (z1 != 0) else 0
    y2 = 1 / z2 if (z2 != 0) else 0

    y0 = 1 / z0 if (z0 != 0) else 0

    lenfn = fn.size
    THETA1A = np.pi * fn                    # THETA1A is for Z1
    THETA2A = 2 * np.pi * fn                # THETA2A is for Z2
    R1 = np.array([[1, r1], [0, 1]])
    R2 = np.array([[1, r2], [0, 1]])
    R3 = np.array([[1, r3], [0, 1]])
    R4 = np.array([[1, r4], [0, 1]])
    R1 = np.tile(R1, (lenfn, 1, 1))
    R2 = np.tile(R2, (lenfn, 1, 1))
    R3 = np.tile(R3, (lenfn, 1, 1))
    R4 = np.tile(R4, (lenfn, 1, 1))
    TL1 = np.stack([np.stack([np.cos(THETA1A), 1j * z1 * np.sin(THETA1A)], axis=-1),
                    np.stack([1j * y1 * np.sin(THETA1A), np.cos(THETA1A)], axis=-1)], axis=1)
    TL2 = np.stack([np.stack([np.cos(THETA2A), 1j * z2 * np.sin(THETA2A)], axis=-1),
                    np.stack([1j * y2 * np.sin(THETA2A), np.cos(THETA2A)], axis=-1)], axis=1)
    PATH1 = R1 @ TL1 @ R2
    PATH2 = R3 @ TL2 @ R4
    warnings.filterwarnings("default")
    return fPath2SP(PATH1, PATH2, y0)


def fPath2SP(PATH1, PATH2, y0):
    # This function converts ABCD parameters to Y-parameters, SUM both paths and convert the result to S-Parameters
    warnings.filterwarnings("ignore")
    # ABCD matrix -> Y matrix (for both paths)
    A1 = PATH1[:, 0, 0]
    B1 = PATH1[:, 0, 1]
    C1 = PATH1[:, 1, 0]
    D1 = PATH1[:, 1, 1]
    A2 = PATH2[:, 0, 0]
    B2 = PATH2[:, 0, 1]
    C2 = PATH2[:, 1, 0]
    D2 = PATH2[:, 1, 1]
    Y_path1 = np.stack(
        [np.stack([(D1 / B1), ((B1 * C1 - A1 * D1) / (B1))], axis=-1), np.stack([(-1 / B1), (A1 / B1)], axis=-1)],
        axis=1)
    Y_path2 = np.stack(
        [np.stack([(D2 / B2), ((B2 * C2 - A2 * D2) / (B2))], axis=-1), np.stack([(-1 / B2), (A2 / B2)], axis=-1)],
        axis=1)
    # SUM paths, parallel equivalent
    Y = Y_path1 + Y_path2
    # Y matrix -> SP matrix in separated arrays for each component (S11, S12, S21, S22)
    Y11 = Y[:, 0, 0]
    Y12 = Y[:, 0, 1]
    Y21 = Y[:, 1, 0]
    Y22 = Y[:, 1, 1]
    deltaY = (Y11 + y0) * (Y22 + y0) - Y12 * Y21
    S11 = ((y0 - Y11) * (y0 + Y22) + Y12 * Y21) / (deltaY)
    S12 = (-2 * Y12 * y0) / (deltaY)
    S21 = (-2 * Y21 * y0) / (deltaY)
    S22 = ((y0 + Y11) * (y0 - Y22) + Y12 * Y21) / (deltaY)
    warnings.filterwarnings("default")
    return S11, S12, S21, S22


def unwrap(q, phase):
    # THIS FUNCTION IS BASED ON MATLAB UNWRAP IMPLEMENTATION.
    # Adaptation of unwrap function to avoid calculating with Not A Number values.
    # Output contains NaN values where there was one, but the rest of the values are well calculated.
    # Find NaN's and Inf's
    p = q
    indf = np.isfinite(p)
    # Unwrap finite data (skip non finite entries)
    q[indf] = np.unwrap(p[indf], phase)
    return q


def getLocalZero(delay):
    # Return the local minimum position avoiding widths < 1 (maths indeterminacy) and positive delays
    # If out is positive -> Unique or widest and minimum point
    # If out is negative -> Minimum point but not the widest
    # If out is NaN -> No Local Minimum found
    localZero = np.NaN
    peaks, prop = find_peaks(-delay, width=(None, None))
    wdt = prop["widths"]
    erridx = (wdt >= 2) & (delay[peaks] < 0)
    peaks = peaks[erridx]
    if peaks.size == 1:
        localZero = peaks[0]
    elif peaks.size > 1:
        idxW = np.argmax(wdt)
        idxP = np.argmin(delay[peaks])
        if idxW == idxP:
            localZero = peaks[idxP]
        else:
            localZero = -peaks[idxP]
    return localZero


def getNZeros(nZeros, delay):
    # Call getLocalZeros() for different number of desired Zeros.
    # Returns iDel array, which contains the positions of Zeros (Negative Delays)
    iDel = np.repeat(np.NaN, nZeros)
    # Always look for left zero (freq < 1)
    iDel[0] = getLocalZero(delay[0:999])
    # Look for central zero (useful for symmetrical designs with 3 zeros)
    if nZeros >= 2:
        idel = getLocalZero(delay[900:1200])
        # Adapt relative to absolute position if a valid zero is found (idel > 0) [see getLocalZero() comments]
        iDel[1] = idel + 900 if idel >= 0 else idel - 900
    # Look for right zero (freq > 1) (useful for non-symmetrical designs)
    if nZeros == 3:
        idel = getLocalZero(delay[1001:-1])
        iDel[2] = idel + 1001 if idel >= 0 else idel - 1001
    return iDel


def getMeasures(nZeros, delay, SS21, SS11, SS22):
    warnings.filterwarnings("ignore", message="invalid value encountered in ")
    warnings.filterwarnings("ignore", message="divide by zero encountered in ")
    # Get NGD indexes
    iDel = getNZeros(nZeros, delay)
    # NGD array for NGD Values
    Del = np.repeat(np.NaN, nZeros)
    # Desired S-Parameters
    S21 = np.repeat(np.NaN, nZeros)
    S11 = np.repeat(np.NaN, nZeros)
    S22 = np.repeat(np.NaN, nZeros)
    # 3 desired bandwidths
    iBW = np.repeat(np.NaN, 2 * nZeros)
    iBW1dB = np.repeat(np.NaN, 2 * nZeros)
    iBW3dB = np.repeat(np.NaN, 2 * nZeros)
    S21dB = 20 * np.log10(abs(SS21))
    # Auxiliar for errors
    err = np.repeat(0, nZeros)
    for nn, idel in enumerate(iDel):
        if not np.isnan(idel):
            if idel >= 0:
                Del[nn] = delay[int(idel)]
                S21[nn] = np.abs(SS21[int(idel)])
                S11[nn] = np.abs(SS11[int(idel)])
                S22[nn] = np.abs(SS22[int(idel)])
            else:
                iDel[nn] = np.abs(idel)
                Del[nn] = delay[int(iDel[nn])]
                S21[nn] = np.abs(SS21[int(iDel[nn])])
                S11[nn] = np.abs(SS11[int(iDel[nn])])
                S22[nn] = np.abs(SS22[int(iDel[nn])])
                err[nn] = -1

            # NGD BANDWIDTH -> iBW
            # Find positive delays at the left side of NGD point
            aux, = np.nonzero(delay[0:int(iDel[nn])] >= 0)
            if aux.size > 0:
                # Save last positive delay
                iBW[2 * nn] = aux[-1]
            # Find positive delay at the right side of NGD point
            aux, = np.nonzero(delay[int(iDel[nn]):] >= 0)
            if aux.size > 0:
                # Save first positive delay
                iBW[2 * nn + 1] = aux[0] + int(iDel[nn])

            # 1DB BANDWIDTH -> iBW1dB
            aux, = np.nonzero(
                (S21dB[0:int(iDel[nn])] >= S21dB[int(iDel[nn])] + 1) | (
                        S21dB[0:int(iDel[nn])] <= S21dB[int(iDel[nn])] - 1))
            if aux.size > 0:
                iBW1dB[2 * nn] = aux[-1]
            aux, = np.nonzero(
                (S21dB[int(iDel[nn]):-1] >= S21dB[int(iDel[nn])] + 1) | (
                        S21dB[int(iDel[nn]):-1] <= S21dB[int(iDel[nn])] - 1))
            if aux.size > 0:
                iBW1dB[2 * nn + 1] = aux[0] + int(iDel[nn])

            # 3DB BANDWIDTH -> iBW3dB
            aux, = np.nonzero(
                (S21dB[0:int(iDel[nn])] >= S21dB[int(iDel[nn])] + 3) | (
                        S21dB[0:int(iDel[nn])] <= S21dB[int(iDel[nn])] - 3))
            if aux.size > 0:
                iBW3dB[2 * nn] = aux[-1]
            aux, = np.nonzero(
                (S21dB[int(iDel[nn]):-1] >= S21dB[int(iDel[nn])] + 3) | (
                        S21dB[int(iDel[nn]):-1] <= S21dB[int(iDel[nn])] - 3))
            if aux.size > 0:
                iBW3dB[2 * nn + 1] = aux[0] + int(iDel[nn])
        else:
            err[nn] = 1
    warnings.filterwarnings("default")
    return iDel, Del, S21, S11, S22, iBW, iBW1dB, iBW3dB, err
