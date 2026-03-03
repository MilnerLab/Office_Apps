import numpy as np
import matplotlib.pyplot as plt
from base_core.quantities.models import Frequency 
from base_core.quantities.enums import Prefix
from scipy.optimize import curve_fit
import math

h = 6.626*10**-34
k = 1.381*10**-23
def anisotropy(alpha,B,I):
    
    return np.sqrt(alpha*I/(4*B*h)) #Note that the formula from the paper omits the h because it uses energy units cm^-1. E = hf = hc/lambda

def c2t_3d_lowtemp(E_0,B,alpha):
    return 1 - 2/E_0*math.sqrt(h*B/alpha)

def c2t_3d_hightemp(E_0,B,alpha,T=0.6):
    return 1 - (1/E_0)*math.sqrt(math.pi*k*T/alpha)*(3/4*h*B/(k*T)+1)

def cgstosi(alpha):
    return alpha*10**-24/(8.998*10**15)#from angstrom cubed to F*m^2

def high_pot_limit(alpha,E_0,B): #high potential well approximation which treats the laser field as CW instead of pulse
    return alpha*(E_0)**2/(4*B*h) #E_0 is converted to V/cm. Should be >> 1 for approximation to hold.

def temp_limit(T,B):
    return k*T/B/h

epsilon_0 = 8.854*10**-12 #in F/m
c = 2.998*10**8
I_0 = 10*10**10/(0.01**2) #W/m^2
E_0 = math.sqrt(2*I_0/(epsilon_0*c)) #V/m
print(f"{E_0:.3e}")



I = np.arange(50*10**10,1.5*10**12,1*10**10) # in W/cm^2
E = np.sqrt(2*I/(0.01)**2/(epsilon_0*c)) #in V/m
alpha_ocs = 3.7 #in angstrom cubed. The polarizability anisotropy
B_ocs_drop = Frequency(2.18,Prefix.GIGA)
B_ocs_jet = Frequency(6,Prefix.GIGA)
alpha_ocs = cgstosi(alpha_ocs)
c2t_ocs = c2t_3d_hightemp(E_0=E,B=B_ocs_drop.value(),alpha=alpha_ocs)

alpha_cs2 = 4.9 #\alpha_|| - \alpha_\perp
B_cs2_drop = Frequency(730,Prefix.MEGA)
B_cs2_jet = Frequency(3.3,Prefix.GIGA)
alpha_cs2 = cgstosi(alpha_cs2)
align_cs2 = anisotropy(alpha_cs2,B_cs2_drop.value(),I)
c2t_cs2 = c2t_3d_hightemp(E_0=E,B=B_cs2_drop.value(),alpha=alpha_cs2)

alpha_cl2 = 3.3 #gas phase
#B_cl2 = 

T_drop = 0.6 #Kelvin
T_jet = 12
potlimit_ocs = high_pot_limit(alpha_ocs,E_0=E_0,B=B_ocs_drop.value())
#potlimit_cs2 = high_pot_limit(alpha_cs2,E_0=E_0,B=B_cs2.value())

print("OCS: "+ f"{potlimit_ocs:.3e}")
#print("CS2: " + f"{potlimit_cs2:.3e}")

temp_limit_ocs_drop = temp_limit(T=T_drop,B=B_ocs_drop.value())
temp_limit_cs2_drop = temp_limit(T=T_drop,B=B_cs2_drop.value())

temp_limit_ocs_jet = temp_limit(T=T_jet,B=B_ocs_jet.value())
temp_limit_cs2_jet = temp_limit(T=T_jet,B=B_cs2_jet.value())
#temp_limit_ocs_jet = temp_limit(T=T_jet,B=B)

print("OCS Droplet temp limit: "+ f"{temp_limit_ocs_drop:.3e}")
print("OCS jet temp limit: "+ f"{temp_limit_ocs_jet:.3e}")
print("CS2 Droplet temp limit: "+ f"{temp_limit_cs2_drop:.3e}")
print("CS2 jet temp limit: " + f"{temp_limit_cs2_jet:.3e}")



fig, ax = plt.subplots()

ax.plot(I,c2t_ocs,'r',label=r"OCS: $\Delta \alpha = 3.7 \mathring{A}^3$, Renormalized in Droplets B = 2.18 GHz")
ax.plot(I,c2t_cs2,'b',label=r"CS2: $\Delta \alpha = 4.9 \mathring{A}^3$, Renormalized in Droplets B = 730 MHz")
ax.legend(loc='best')
ax.set_xlabel(r"Intensity ($W/cm^2$)")
ax.set_ylabel(r"$<cos^2\theta_{3D}>$ (isotropic gives 1/3 instead of 1/2)")
ax.set_title("The hightemp limit kT/B >> 1 formula from Seideman (we only have ratios less than 100)")
plt.show()

