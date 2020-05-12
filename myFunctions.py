import numpy as np
import warnings
from scipy.signal import find_peaks


def fCalcSPMod2(t1, r1, z1, r2, z2, r3, r4, z3, r5, z0, fn):
    warnings.filterwarnings("ignore")
    y1 = 1 / z1 if (z1 != 0) else 0
    y2 = 1 / z2 if (z2 != 0) else 0
    y3 = 1 / z3 if (z3 != 0) else 0

    y0 = 1 / z0 if (z0 != 0) else 0
    lenfn = fn.size

    THETA1A = t1 * fn
    THETA1B = (np.pi - t1) * fn
    THETA2A = 2 * np.pi * fn
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
    warnings.filterwarnings("ignore")
    y1 = 1 / z1 if (z1 != 0) else 0
    y2 = 1 / z2 if (z2 != 0) else 0

    y0 = 1 / z0 if (z0 != 0) else 0
    lenfn = fn.size

    THETA1A = np.pi * fn
    THETA2A = 2 * np.pi * fn
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
    warnings.filterwarnings("ignore")
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
    Y = Y_path1 + Y_path2
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
    # Find NaN's and Inf's
    p = q
    indf = np.isfinite(p)
    # Unwrap finite data (skip non finite entries)
    q[indf] = np.unwrap(p[indf], phase)
    return q


def getLocalZero(delay):
    # Return the local minimum avoiding widths < 1 (maths indeterminacy) and positive delays
    # If out is positive -> Unique or widest and minimum point
    # If out is negative -> Minimum point but not the widest
    # If out is NaN -> No Local Minimum found
    localZero = np.NaN
    peaks, prop = find_peaks(-delay, width=(None, None))
    wdt = prop["widths"]
    erridx = (wdt > 1) & (delay[peaks] < 0)
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


def get3Zeros_deprecated(delay):
    # Return the index of Local minimuns
    mIDX = np.array([np.nan, np.nan, np.nan])
    # Split into 3 freq ranges
    TF1, d1 = find_peaks(-delay[519:899], prominence=(None, None));
    TFc, dc = find_peaks(-delay[899:1099], prominence=(None, None));
    TF2, d2 = find_peaks(-delay[1099:1481], prominence=(None, None));
    if TF1.size == 1:
        mIDX[0] = TF1[0] + 519;
    elif TF1.size > 1:
        prom1 = np.where(d1["prominences"] == d1["prominences"].max())[0]
        mIDX[0] = TF1[prom1[0]] + 519

    if TFc.size == 1:
        mIDX[1] = TFc[0] + 899;
    elif TFc.size > 1:
        promc = np.where(dc["prominences"] == dc["prominences"].max())[0]
        mIDX[1] = TFc[promc[0]] + 899

    if TF2.size == 1:
        mIDX[2] = TF2[0] + 1099;
    elif TF2.size > 1:
        prom2 = np.where(d2["prominences"] == d2["prominences"].max())[0]
        mIDX[2] = TF2[prom2[0]] + 1099
    return mIDX


def get_BW(delay, mdelIDX, mdel):
    warnings.filterwarnings("ignore")
    iBW = np.empty((6))
    iBW[:] = np.NaN
    for i, _ in enumerate(mdelIDX):
        if ((not np.isnan(mdelIDX[i])) and (mdel[i] < 0)):
            aux, = np.nonzero(delay[0:mdelIDX[i]] >= 0)
            if aux.size > 0:
                iBW[2 * i] = aux[-1]
            aux, = np.nonzero(delay[mdelIDX[i]:-1] >= 0)
            if aux.size > 0:
                iBW[2 * i + 1] = aux[0] + mdelIDX[i]
    warnings.filterwarnings("default")
    return iBW


def get_BW1dB(S21, mdelIDX, mdel):
    warnings.filterwarnings("ignore")
    iBW1dB = np.empty((6))
    iBW1dB[:] = np.NaN
    S21dB = 20 * np.log10(abs(S21))
    for i, _ in enumerate(mdelIDX):
        if ((not np.isnan(mdelIDX[i])) and (mdel[i] < 0)):
            aux, = np.nonzero(S21dB[0:mdelIDX[i]] >= S21dB[mdelIDX[i]] + 1)
            if aux.size > 0:
                iBW1dB[2 * i] = aux[-1]
            aux, = np.nonzero(S21dB[mdelIDX[i]:-1] >= S21dB[mdelIDX[i]] + 1)
            if aux.size > 0:
                iBW1dB[2 * i + 1] = aux[0] + mdelIDX[i]
    warnings.filterwarnings("default")
    return iBW1dB


def get_BW3dB(S21, mdelIDX, mdel):
    warnings.filterwarnings("ignore")
    iBW3dB = np.empty((6))
    iBW3dB[:] = np.NaN
    S21dB = 20 * np.log10(abs(S21))
    for i, _ in enumerate(mdelIDX):
        if ((not np.isnan(mdelIDX[i])) and (mdel[i] < 0)):
            aux, = np.nonzero(S21dB[0:mdelIDX[i]] >= S21dB[mdelIDX[i]] + 3)
            if aux.size > 0:
                iBW3dB[2 * i] = aux[-1]
            aux, = np.nonzero(S21dB[mdelIDX[i]:-1] >= S21dB[mdelIDX[i]] + 3)
            if aux.size > 0:
                iBW3dB[2 * i + 1] = aux[0] + mdelIDX[i]
    warnings.filterwarnings("default")
    return iBW3dB
