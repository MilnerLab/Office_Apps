from matplotlib import pyplot as plt
from base_core.math.functions import erfc 
from scipy.optimize import curve_fit 
import numpy as np

folder = r"C:\\Users\\camp06\\OneDrive - UBC\\Documents\\PhDVault\\BeamProfileData\\Knife_edge\\"
file_old = folder + "20260721_ga.csv"
file_new = folder + "20260911GAKnifeEdge.csv"

data_old = np.loadtxt(file_old,delimiter=',',skiprows=1)
data_new = np.loadtxt(file_new,delimiter=',',skiprows=1)

inch_tomm = 2.54
x_old = data_old[:,0]
y_old = data_old[:,1]
x_new = data_new[:,0]
y_new = data_new[:,1]

fit_result_old, error_old = curve_fit(erfc,x_old,y_old,maxfev=5000)
fit_result_new, error_new = curve_fit(erfc,x_new,y_new)

shift = fit_result_old[1]/2
x_old_shift = x_old - shift #shift origin to make axis the slit width
x_n = np.linspace(np.min(x_new), np.max(x_new),100)
x_o = np.linspace(np.min(x_old),np.max(x_old),100)

y_max_n = np.max(y_new)
y_min_n = np.min(y_new)
y_max_o = np.max(y_old)
y_min_o = np.min(y_old)

y_old = (y_old - y_min_o) / y_max_o
y_new = (y_new - y_min_n) / y_max_n
fit_old = (erfc(x_o,*fit_result_old) - y_min_o) / y_max_o
fit_new = (erfc(x_n,*fit_result_new) - y_min_n) / y_max_n

#Multiply the sigma by 4! 2 for the the fullwidth (instead of half) and 2 for the power being proportional to the square of the gaussian.
diameter_old = 2*2*fit_result_old[2]
diameter_new = 2*2*fit_result_new[2]

print("Old diameter = ",diameter_old,"\nNew diameter = ",diameter_new)

plt.plot(x_o - shift,fit_old,label=r'2026/07/21 GA, D = ' + f"{diameter_old:.1f} mm",color='g')
plt.plot(x_old_shift,y_old,color='g',marker='d',linestyle='')
plt.plot(x_n,fit_new,label=r'2026/09/11 GA, D = ' + f"{diameter_new:.1f} mm",color='b')
plt.plot(x_new,y_new,color='b',marker='d',linestyle='')
plt.text(0.1,0.8,'')
plt.legend()
plt.show()

