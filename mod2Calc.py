import numpy as np
import myFunctions as my
from datetime import datetime
import os
import time
import itertools as it
import multiprocessing
import scipy.io as sc

j = 1
foldername = "Mod2"
lenRZ = 11  # 11 21
lenT = 19  # 19 37
nZeros = 3

R1 = np.linspace(0, 100, num=lenRZ)
R2 = np.linspace(0, 100, num=lenRZ)
R3 = np.linspace(0, 100, num=lenRZ)
R4 = np.linspace(0, 100, num=lenRZ)
R5 = np.linspace(0, 100, num=lenRZ)

Z1 = np.linspace(25, 125, num=lenRZ)
Z2 = np.linspace(25, 125, num=lenRZ)
Z3 = np.linspace(25, 125, num=lenRZ)

T1 = np.linspace(0, np.pi, num=lenT)

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

    # for j in tRange:  # T1
    # for k in xRange:  # R1
    # for l in xRange:  # Z1
    # for m in xRange:  # R2
    for n in xRange:  # Z2
        for o in xRange:  # R3
            for p in xRange:  # R4
                for q in xRange:  # Z3
                    for r in xRange:  # R5
                        SS11, SS12, SS21, SS22 = my.fCalcSPMod2(T1[j], R1[k], Z1[l], R2[m], Z2[n], R3[o], R4[p], Z3[q],
                                                                R5[r], z0, fn)
                        group_delay = -(np.diff(my.unwrap(np.angle(SS21), np.pi)) / np.diff(f)) / (2 * np.pi)
                        iDel, Del, S21, S11, S22, iBW, iBW1dB, iBW3dB, err = my.getMeasures(nZeros, group_delay, SS21,
                                                                                            SS11, SS22)
                        # iDel = my.get3Zeros(group_delay)
                        # Del = np.array([np.NaN, np.NaN, np.NaN])
                        # S21 = np.array([np.NaN, np.NaN, np.NaN])
                        # S11 = np.array([np.NaN, np.NaN, np.NaN])
                        # S22 = np.array([np.NaN, np.NaN, np.NaN])
                        # for nn, idel in enumerate(iDel):
                        #     if not np.isnan(idel):
                        #         Del[nn] = group_delay[int(idel)]
                        #         S21[nn] = np.abs(SS21[int(idel)])
                        #         S11[nn] = np.abs(SS11[int(idel)])
                        #         S22[nn] = np.abs(SS22[int(idel)])
                        #     # end
                        # # end
                        # iBW = my.get_BW(group_delay, iDel.astype(int), Del)
                        # iBW1dB = my.get_BW1dB(SS21, iDel.astype(int), Del)
                        # iBW3dB = my.get_BW3dB(SS21, iDel.astype(int), Del)
                        #
                        # datos = np.array([
                        #     T1[j], R1[k], Z1[l], R2[m], Z2[n], R3[o], R4[p], Z3[q], R5[r],
                        #     Del[0], Del[1], Del[2],
                        #     S21[0], S21[1], S21[2],
                        #     S11[0], S11[1], S11[2],
                        #     S22[0], S22[1], S22[2],
                        #     iDel[0], iDel[1], iDel[2],
                        #     iBW3dB[0], iBW3dB[1], iBW3dB[2], iBW3dB[3], iBW3dB[4], iBW3dB[5],
                        #     iBW1dB[0], iBW1dB[1], iBW1dB[2], iBW1dB[3], iBW1dB[4], iBW1dB[5],
                        #     iBW[0], iBW[1], iBW[2], iBW[3], iBW[4], iBW[5]])
                        if any(err):
                            errores = np.concatenate(([R1[m]], [Z1[n]], [R2[o]], [R3[p]], [Z2[q]], [R4[r]], err))
                            ERRORES.append(errores)

                        datos = np.concatenate(([R1[m]], [Z1[n]], [R2[o]], [R3[p]], [Z2[q]], [R4[r]],
                                                iDel, Del, S21, S11, S22, iBW3dB, iBW1dB, iBW))
                        DATA.append(datos)
                    # end r -> R5
                # end q -> Z3
            # end p -> R4
        # end o -> R3
    # end n -> Z2
    # end m -> R2
    # end l -> Z1
    # end k -> R1
    # end j -> T1

    filename = str(int(T1[j])).zfill(3) + "T1" + str(int(R1[k])).zfill(3) + "R1" + str(
        int(Z1[l])).zfill(3) + "Z1" + str(int(R2[m])).zfill(3) + "R2.mat"
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
            aux.append((int(np.where(R1 == r1)[0]), int(np.where(Z1 == z1)[0]), int(np.where(R2 == r2)[0])))
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
