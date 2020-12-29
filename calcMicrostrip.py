# -----------------------------------------------------------
# Function-script for width and length TL calculation
#
# 2020 - Margarita Martínez-Coves
# Master Thesis: "Estudio de estructuras con retardo de grupo
# negativo en tecnología microstrip"
# Universitat d'Alacant (Spain)
# email martinezcoves.m@gmail.com
# -----------------------------------------------------------

import numpy as np
import sys

# Call by command line: python calcMicrostrip.py W1 L1 W2 L2
# e.g: python calcMicrostrip.py 0.5 95.4 1.6 185.2
if __name__ == "__main__":
	W1 = float(sys.argv[1])
	L1 = float(sys.argv[2])
	W2 = float(sys.argv[3])
	L2 = float(sys.argv[4])

	nR = 2
	r_space = 0.5

	L2_h = L1 + nR*r_space
	print("L2_h = " + str(L2_h))
	L2_v = (L2 - L2_h - 2*W2)/2
	print("L2_v = " + str(L2_v))