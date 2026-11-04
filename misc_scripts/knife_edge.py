from altair import Data
from matplotlib import pyplot as plt
from base_core.math.functions import erfc 
from scipy.optimize import curve_fit 
import numpy as np

def rescale(y):
    return (y - np.min(y)) / np.max(y)


folder = r"C:\\Users\\camp06\\OneDrive - UBC\\Documents\\PhDVault\\Equipment and Instrumentation\\BeamProfileData\\Knife_edge\\"

date = "20261009"

plot_title = date + " knife edge"

files = {
    "GA": [folder + r"20261009153_GA_Horizontal.csv", folder + r"20261009318_GA_Vertical.csv"],
    "DA": [folder + r"20261009158_DA_Horizontal.csv", folder + r"20261009307_DA_Vertical.csv"],
    "Probe": [folder + r"20261009211_Probe_Horizontal.csv", folder + r"20261009252_Probe_Vertical_3.csv"]
    }

# files = {
#     "GA_ian": [folder + r"20261009153_GA_Horizontal.csv"],
#     "GA_cam": [folder + r"20260721_ga.txt"],

#     }

#scan axes
dimensions = [0,1]



inch_tomm = 2.54
fit_results = {}

style = {"GA": ["+","g"], "DA": ["d","b"], "Probe": [".","r"]} # ordering: [marker,color]

fig, axes = plt.subplots(1,2,figsize=(8,6))
for key in list(files.keys()):
    for i in dimensions:
        arr = np.loadtxt(files[key][i],delimiter='\t')
        x = arr[:,0]
        y = arr[:,1]
        y_scaled = rescale(y)
        x_mean = np.mean(x)
        if y[0] > y[-1]:
            x = - (x - x_mean)
        else:
            x = x - x_mean
        # initial guess from the data: [A, x0, sigma, offset]; x0 is where y crosses its midpoint
        y_mid = (np.max(y) + np.min(y)) / 2
        p0 = [np.max(y) - np.min(y), x[np.argmin(np.abs(y - y_mid))], (np.max(x) - np.min(x)) / 10, y_mid]
        fit, error = curve_fit(erfc,x,y,p0=p0,maxfev=5000)
        x_fit = np.linspace(np.min(x),np.max(x),100) 
        y_fit_scaled = rescale(erfc(x_fit,*fit))
        D = fit[2] * 4 #Multiply the sigma by 4! 2 for the the fullwidth (instead of half) and 2 for the power being proportional to the square of the gaussian.
        print(error*4)
        delta_D = round(float(error[2,2] * 4),3)
        axes[i].plot(x,y_scaled,marker = style[key][0],color=style[key][1],linestyle='')
        axes[i].plot(x_fit,y_fit_scaled,label=key + f" D = {D:.1f} mm",color=style[key][1])
    
     

axes[0].set_title("Horizontal")
axes[1].set_title("Vertical")
axes[0].legend()
axes[1].legend()
fig.suptitle(plot_title)
plt.show()

#shift = fit_result_old[1]/2
#x_old_shift = x_old - shift #shift origin to make axis the slit width
#x_n = np.linspace(np.min(x), np.max(x),100)



# y_old = (y_old - y_min_o) / y_max_o
# y_new = (y_new - y_min_n) / y_max_n
# fit_old = (erfc(x_o,*fit_result_old) - y_min_o) / y_max_o
# fit_new = (erfc(x_n,*fit_result_new) - y_min_n) / y_max_n





# plt.plot(x_o - shift,fit_old,label=r'2026/07/21 GA, D = ' + f"{diameter_old:.1f} mm",color='g')
# plt.plot(x_old_shift,y_old,color='g',marker='d',linestyle='')
# plt.plot(x_n,fit_new,label=r'2026/09/11 GA, D = ' + f"{diameter_new:.1f} mm",color='b')
# plt.plot(x_new,y_new,color='b',marker='d',linestyle='')
# plt.text(0.1,0.8,'')


