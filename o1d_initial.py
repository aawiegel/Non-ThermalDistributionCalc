import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy import constants

initial_O1D_file = 'O1d_energy_255nm.csv' #File that contains initial O1D energy data (Reference: Dylweski 2001)
T = 300. #Temperature in Kelvin
R = constants.value('molar gas constant') #Molar gas constant
eV = constants.value('electron volt')
Na = constants.value('Avogadro constant')
length = 1001 #Number of data points

#Read energy and probability data from file 
E_eV = np.genfromtxt(initial_O1D_file, dtype = float, delimiter = ",", usecols = 0)
Prob = np.genfromtxt(initial_O1D_file, dtype = float, delimiter = ",", usecols = 1)

#Convert to reduced energy units
x = E_eV*eV*Na/(R*T)

#Convert from O(1D)+O2(1D) C.O.M. energy to lab energy of O1D
x_O = x*32./48.
x_P = x*32./49.
x_Q = x*32./50.


#Create x space
x_L = np.linspace(0,x_Q[x_Q.size-1],length)


#Normalize probability using trapezoid integration and interpolate
Prob_O_norm = Prob/integrate.trapz(Prob,x_O)
Prob_O_L = interpolate.interp1d(x_O, Prob_O_norm)

Prob_P_norm = Prob/integrate.trapz(Prob,x_P)
Prob_P_L = interpolate.interp1d(x_P, Prob_P_norm)

Prob_Q_norm = Prob/integrate.trapz(Prob,x_Q)
Prob_Q_L = interpolate.interp1d(x_Q, Prob_Q_norm)

#Generate array using x space and normalized probability; switch axes to put into columns
output = np.array((x_L,Prob_O_L(x_L)), dtype=float)
output = np.swapaxes(output,0,1)

#Save text to file for 16O
np.savetxt('norm_16O1D.csv', output, delimiter=",")

#Generate array using x space and normalized probability; switch axes to put into columns
output = np.array((x_L,Prob_P_L(x_L)), dtype=float)
output = np.swapaxes(output,0,1)

#Save text to file for 17O
np.savetxt('norm_17O1D.csv', output, delimiter=",")

#Generate array using x space and normalized probability; switch axes to put into columns
output = np.array((x_L,Prob_Q_L(x_L)), dtype=float)
output = np.swapaxes(output,0,1)

#Save text to file for 18O
np.savetxt('norm_18O1D.csv', output, delimiter=",")

#Plot
plt.plot(x_O,Prob,'o',x_L,Prob_O_L(x_L),'-',x_L,Prob_P_L(x_L),'--',x_L,Prob_Q_L(x_L),'--')
plt.legend(['data', '16O','17O','18O'], loc='best')
plt.show()
