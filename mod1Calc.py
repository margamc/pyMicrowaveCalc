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
nZeros = 2

R1 = np.linspace(0, 100, num=lenRZ)
R2 = np.linspace(0, 100, num=lenRZ)
R3 = np.linspace(0, 100, num=lenRZ)
R4 = np.linspace(0, 100, num=lenRZ)

Z1 = np.linspace(25, 125, num=lenRZ)
Z2 = np.linspace(25, 125, num=lenRZ)

lenfn = 2001  # Not change. myFunctions.getNZeros() depends on it
fn = np.linspace(0, 2, num=lenfn)
if nZeros == 1:
    fn = fn[0:1000]
    lenfn = fn.size
elif nZeros == 2:
    fn = fn[0:1201]
    lenfn = fn.size
# 0 - 999 --> 1 zero
# 900 - 1200 --> 2 zeros
# 1001 - end --> 3 zeros

z0 = 50
f0 = 1e9

f = f0 * fn

xRange = range(lenRZ)


def calcular(counter):
    t = time.time()
    DATA = list()
    ERRORES = list()
    m = np.array(counter)[0]
    n = np.array(counter)[1]
    # for m in x21:  # R1
    # for n in x21:  # Z1
    for o in xRange:  # R2
        for p in xRange:  # R3
            for q in xRange:  # Z2
                for r in xRange:  # R4
                    SS11, SS12, SS21, SS22 = my.fCalcSPMod1(R1[m], Z1[n], R2[o], R3[p], Z2[q], R4[r], z0, fn)
                    group_delay = -(np.diff(my.unwrap(np.angle(SS21), np.pi)) / np.diff(f)) / (2 * np.pi)
                    iDel, Del, S21, S11, S22, iBW, iBW1dB, iBW3dB, err = my.getMeasures(nZeros, group_delay, SS21, SS11,
                                                                                        SS22)
                    if any(err):
                        errores = np.concatenate(([R1[m]], [Z1[n]], [R2[o]], [R3[p]], [Z2[q]], [R4[r]], err))
                        ERRORES.append(errores)

                    datos = np.concatenate(([R1[m]], [Z1[n]], [R2[o]], [R3[p]], [Z2[q]], [R4[r]],
                                            iDel, Del, S21, S11, S22, iBW3dB, iBW1dB, iBW))
                    DATA.append(datos)
                # end r -> R4
            # end q -> Z2
        # end p -> R3
    # end o -> R2
    # end n -> Z1
    # end m -> R1
    elapsed = time.time() - t
    t2 = time.time()
    filename = str(int(R1[m])).zfill(3) + "R1" + str(
        int(Z1[n])).zfill(3) + "Z1.mat"
    sc.savemat(foldername + "/" + filename, {'data': DATA}, do_compression=True)
    sc.savemat(foldername + "_errors" + "/" + "errors_" + filename, {'errors': ERRORES}, do_compression=True)
    elapsed2 = time.time() - t2
    elapsed3 = time.time() - t
    print("Archivo " + filename + " creado." +
          " [" + datetime.now().strftime('%H:%M:%S.%f')[:-4] + "] | Bucle = " + str(elapsed) + " Guardado = " + str(
        elapsed2) + " Total = " + str(elapsed3))
    return


if __name__ == "__main__":

    print("Comienzo de ejecución. [" +
          datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")
    aux = list()
    if not os.path.exists(foldername + "_errors"):
        os.makedirs(foldername + "_errors")
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
        contador.remove(elem)
    # end
    print(str(np.shape(contador)[0]) + " casos a computar con " + str(multiprocessing.cpu_count()) + " núcleos.")
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    # p = multiprocessing.Pool(6)
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
