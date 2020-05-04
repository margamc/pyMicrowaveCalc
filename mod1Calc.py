import numpy as np
import myFunctions as my
from datetime import datetime
import os
import time
import itertools as it
import multiprocessing
import scipy.io as sc

foldername = "Mod1"
lenRZ = 21
lenfn = 2001

R1 = np.linspace(0, 100, num=lenRZ)
R2 = np.linspace(0, 100, num=lenRZ)
R3 = np.linspace(0, 100, num=lenRZ)
R4 = np.linspace(0, 100, num=lenRZ)

Z1 = np.linspace(25, 125, num=lenRZ)
Z2 = np.linspace(25, 125, num=lenRZ)

fn = np.linspace(0, 2, num=lenfn)

z0 = 50
f0 = 1e9

f = f0 * fn

xRange = range(lenRZ)


def calcular(counter):
    DATA = list()
    m = np.array(counter)[0]
    n = np.array(counter)[1]
    # print("Nucleo " + multiprocessing.current_process().name[-2:] +
    #       " iniciado con R1=" + str(int(R1[m])) + " Z1=" + str(int(Z1[n])) +
    #       " [" + datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")
    # for m in x21:  # R1
    # for n in x21:  # Z1
    for o in xRange:  # R2
        for p in xRange:  # R3
            for q in xRange:  # Z2
                for r in xRange:  # R4
                    SS11, SS12, SS21, SS22 = my.fCalcSPMod1(R1[m], Z1[n], R2[o], R3[p], Z2[q], R4[r], z0, fn)
                    group_delay = -(np.diff(my.unwrap(np.angle(SS21), np.pi)) / np.diff(f)) / (2 * np.pi)
                    iDel = my.get3Zeros(group_delay)
                    Del = np.array([np.NaN, np.NaN, np.NaN])
                    S21 = np.array([np.NaN, np.NaN, np.NaN])
                    S11 = np.array([np.NaN, np.NaN, np.NaN])
                    S22 = np.array([np.NaN, np.NaN, np.NaN])
                    for nn, idel in enumerate(iDel):
                        if not np.isnan(idel):
                            Del[nn] = group_delay[int(idel)]
                            S21[nn] = np.abs(SS21[int(idel)])
                            S11[nn] = np.abs(SS11[int(idel)])
                            S22[nn] = np.abs(SS22[int(idel)])
                        # end
                    # end
                    iBW = my.get_BW(group_delay, iDel.astype(int), Del)
                    iBW1dB = my.get_BW1dB(SS21, iDel.astype(int), Del)
                    iBW3dB = my.get_BW3dB(SS21, iDel.astype(int), Del)

                    datos = np.array([
                        R1[m], Z1[n], R2[o], R3[p], Z2[q], R4[r],
                        Del[0], Del[1], Del[2],
                        S21[0], S21[1], S21[2],
                        S11[0], S11[1], S11[2],
                        S22[0], S22[1], S22[2],
                        iDel[0], iDel[1], iDel[2],
                        iBW3dB[0], iBW3dB[1], iBW3dB[2], iBW3dB[3], iBW3dB[4], iBW3dB[5],
                        iBW1dB[0], iBW1dB[1], iBW1dB[2], iBW1dB[3], iBW1dB[4], iBW1dB[5],
                        iBW[0], iBW[1], iBW[2], iBW[3], iBW[4], iBW[5]])
                    DATA.append(datos)
                # end r -> R4
            # end q -> Z2
        # end p -> R3
    # end o -> R2
    # end n -> Z1
    # end m -> R1

    filename = str(int(R1[m])).zfill(3) + "R1" + str(
        int(Z1[n])).zfill(3) + "Z1.mat"
    sc.savemat(foldername + "/" + filename, {'data': DATA}, do_compression=True)
    print("Archivo " + filename + " creado." +
          " [" + datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")


if __name__ == "__main__":
    print("Comienzo de ejecución. [" +
          datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")
    aux = list()
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        files = os.listdir(foldername)
        for file in files:
            r1 = int(file[0:3])
            z1 = int(file[5:8])
            aux.append((int(np.where(R1 == r1)[0]), int(np.where(Z1 == z1)[0])))
    t = time.time()
    M = list(xRange)
    N = list(xRange)
    contador = list(it.product(M, N))
    for elem in aux:
        contador.remove(aux)
    # end
    print(str(np.shape(contador)[0]) + " casos a computar con " + str(multiprocessing.cpu_count()) + " núcleos.")
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.map(calcular, contador)
    p.close()
    p.join()

    elapsed = time.time() - t
    print("Proceso completo! [" +
          datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")
    units = " (s)"
    if elapsed > 60:
        elapsed = elapsed / 60
        units = " (min)"
    if elapsed > 60:
        elapsed = elapsed / 60
        units = " (horas)"
    print("Duración total: " + str(elapsed) + units)
