# -----------------------------------------------------------
# Main script for Model 1
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

# INPUT PARAMETERS
foldername = "Mod1"  # Folder where data will be saved
lenRZ = 21  # Resistance and Impedance precision. 21 for 5 ohms increment
nZeros = 2  # How many zeros to look for.

# Arrays of different Resistances and Impedances values.
R1 = np.linspace(0, 100, num=lenRZ)
R2 = np.linspace(0, 100, num=lenRZ)
R3 = np.linspace(0, 100, num=lenRZ)
R4 = np.linspace(0, 100, num=lenRZ)
Z1 = np.linspace(25, 125, num=lenRZ)
Z2 = np.linspace(25, 125, num=lenRZ)

# TODO: Adapt myFunctions.getNZeros() to work with different lenfn and limit values.
lenfn = 2001  # DO NOT CHANGE. myFunctions.getNZeros() depends on it
fn = np.linspace(0, 2, num=lenfn)  # Frequency between 0 and 2 (normalized)
# Speed up by reducing fn size. Comment for complete computation.
if nZeros == 1:
    fn = fn[0:1000]
    lenfn = fn.size
elif nZeros == 2:
    fn = fn[0:1201]
    lenfn = fn.size
# 0 - 999 --> 1 zero
# 900 - 1200 --> 2 zeros
# 1001 - end --> 3 zeros

z0 = 50  # Ports impedance
f0 = 1e9  # Central frequency

f = f0 * fn  # Denormalized frequency array

xRange = range(lenRZ)  # Range create vector from 0 to lenRZ-1. Index for loops.


# Function for multiprocessing. Each processor runs this function with different values and writes results into a file.
def calcular(counter):
    t = time.time()
    DATA = list()
    ERRORES = list()
    m = np.array(counter)[0]  # m and n values comes from multiprocessor call in main. This avoid 2 loops.
    n = np.array(counter)[1]
    # for m in x21:  # R1
    # for n in x21:  # Z1
    for o in xRange:  # R2
        for p in xRange:  # R3
            for q in xRange:  # Z2
                for r in xRange:  # R4
                    # Calculate S-Parameters
                    SS11, SS12, SS21, SS22 = my.fCalcSPMod1(R1[m], Z1[n], R2[o], R3[p], Z2[q], R4[r], z0, fn)
                    # Calculate Group Delay
                    group_delay = -(np.diff(my.unwrap(np.angle(SS21), np.pi)) / np.diff(f)) / (2 * np.pi)
                    # Get desired points (gd and index, s21, s11...)
                    iDel, Del, S21, S11, S22, iBW, iBW1dB, iBW3dB, err = my.getMeasures(nZeros, group_delay, SS21, SS11,
                                                                                        SS22)
                    # Errors are also saved for verification purposes.
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
    # Time consumption and write to file.
    elapsed = time.time() - t
    t2 = time.time()
    filename = str(int(R1[m])).zfill(3) + "R1" + str(int(Z1[n])).zfill(3) + "Z1.mat"
    sc.savemat(foldername + "/" + filename, {'data': DATA}, do_compression=True)
    sc.savemat(foldername + "_errors" + "/" + "errors_" + filename, {'errors': ERRORES}, do_compression=True)
    elapsed2 = time.time() - t2
    elapsed3 = time.time() - t
    print("Archivo " + filename + " creado." + " [" + datetime.now().strftime('%H:%M:%S.%f')[:-4] +
          "] | Bucle = " + str(elapsed) + " Guardado = " + str(elapsed2) + " Total = " + str(elapsed3))
    return


if __name__ == "__main__":
    print("Comienzo de ejecución. [" +
          datetime.now().strftime('%H:%M:%S.%f')[:-4] + "]")
    aux = list()
    # Folder creation if it doesn't exist.
    if not os.path.exists(foldername + "_errors"):
        os.makedirs(foldername + "_errors")
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        # Check if there are files in folder to avoid repeat same combinations.
        files = os.listdir(foldername)
        for file in files:
            r1 = int(file[0:3])
            z1 = int(file[5:8])
            aux.append((int(np.where(R1 == r1)[0]), int(np.where(Z1 == z1)[0])))
    t = time.time()
    M = list(xRange)        # M and N are index for multiprocessing.
    N = list(xRange)
    contador = list(it.product(M, N))   # Create a list with all possible combinations
    # Delete combinations which are already saved in folder.
    for elem in aux:
        contador.remove(elem)
    # end
    print(str(np.shape(contador)[0]) + " casos a computar con " + str(multiprocessing.cpu_count()) + " núcleos.")
    # Multiprocessing computation with all available cpu
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.map(calcular, contador)       # Run function (calcular) for all desired combinations (contador) in parallel.
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
