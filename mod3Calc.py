# -----------------------------------------------------------
# Main script for Model 3
#
# 2020 - Margarita Martínez-Coves
# Master Thesis: "Estudio de estructuras con retardo de grupo
# negativo en tecnología microstrip"
# Universitat d'Alacant (Spain)
# email martinezcoves.m@gmail.com
# -----------------------------------------------------------

import numpy as np
import myFunctions as my
from datetime import datetime
import os
import time
import itertools as it
import multiprocessing
import scipy.io as sc

j = 8               # Is the T value
foldername = "080T1"
lenRZ = 11  # 11 21 # 11 for increments of 10, 21 for increments of 21.
lenT = 37  # 37 73  # 37 for increments of 10, 73 for increments of 5.
nZeros = 3

R = np.linspace(0, 100, num=lenRZ)

Z = np.linspace(25, 125, num=lenRZ)

T2 = np.linspace(0, 2 * np.pi, num=lenT)

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
tRange = range(lenT)


def calcular(counter):
    DATA = list()
    ERRORES = list()
    k = np.array(counter)[0]
    l = np.array(counter)[1]
    m = np.array(counter)[2]

    # for j in tRange:  # T2
    # for k in xRange:  # R1
    # for l in xRange:  # Z1
    # for m in xRange:  # R2
    for n in xRange:  # R3
        for o in xRange:  # Z2
            for p in xRange:  # R4
                for q in xRange:  # Z3
                    for r in xRange:  # R5
                        SS11, SS12, SS21, SS22 = my.fCalcSPMod3(T2[j], R[k], Z[l], R[m], R[n], Z[o], R[p], Z[q],
                                                                R[r], z0, fn)
                        group_delay = -(np.diff(my.unwrap(np.angle(SS21), np.pi)) / np.diff(f)) / (2 * np.pi)
                        iDel, Del, S21, S11, S22, iBW, iBW1dB, iBW3dB, err = my.getMeasures(nZeros, group_delay, SS21,
                                                                                            SS11, SS22)
                        if any(err):
                            errores = np.concatenate(([Z[n]], [R[o]], [R[p]], [Z[q]], [R[r]], err))
                            ERRORES.append(errores)

                        datos = np.concatenate(([Z[n]], [R[o]], [R[p]], [Z[q]], [R[r]],
                                                iDel, Del, S21, S11, S22, iBW3dB, iBW1dB, iBW))
                        DATA.append(datos)
                    # end r -> R5
                # end q -> Z3
            # end p -> R4
        # end o -> Z2
    # end n -> R3
    # end m -> R2
    # end l -> Z1
    # end k -> R1
    # end j -> T2

    filename = str(int(round(T2[j] * 180 / np.pi))).zfill(3) + "T2" + str(int(R[k])).zfill(3) + "R1" + str(
        int(Z[l])).zfill(3) + "Z1" + str(int(R[m])).zfill(3) + "R2.mat"

    sc.savemat(foldername + "/" + filename, {'data': DATA}, do_compression=True)
    sc.savemat(foldername + "_errors" + "/" + "errors_" + filename, {'errors': ERRORES}, do_compression=True)
    print("Archivo " + filename + " creado." +
          " [" + datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")


if __name__ == "__main__":
    print("Comienzo de ejecución. [" +
          datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")
    # Comprueba si ya hay una carpeta con archivos creados para no repetir calculos
    aux = list()
    if not os.path.exists(foldername + "_errors"):
        os.makedirs(foldername + "_errors")
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        files = os.listdir(foldername)
        for file in files:
            r1 = int(file[5:8])
            z1 = int(file[10:13])
            r2 = int(file[15:18])
            aux.append((int(np.where(R == r1)[0]), int(np.where(Z == z1)[0]), int(np.where(R == r2)[0])))
    t = time.time()
    # J = list(tRange)
    K = list(xRange)
    L = list(xRange)
    M = list(xRange)
    contador = list(it.product(K, L, M))
    for elem in aux:
        contador.remove(elem)
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
