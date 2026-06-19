from encodings.punycode import T

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from base_core.math.functions import gaussian

mpl.rcParams.update({
    #"figure.figsize": (6.299,3.543), #16:9 beamer frame
    "figure.figsize": (5,2), #90% of beamer frame width
    
    "axes.titlesize": "medium",
    "axes.labelsize": 10,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.7,
    
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 8,
    
    "lines.linewidth": 0.7,
    
    "text.usetex": True,      
    "font.family": "serif",
    "font.serif": ["cmr10"]})

def truncate_cmap(cmap, minval=0.0, maxval=0.5):
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated', cmap(np.linspace(minval, maxval, 256)))
    return new_cmap

def rotate_about_center(x, y, cx, cy, angle_deg):
    angle = np.radians(angle_deg)
    x_c = x - cx
    y_c = y - cy
    x_rot = x_c * np.cos(angle) - y_c * np.sin(angle)
    y_rot = x_c * np.sin(angle) + y_c * np.cos(angle)
    return x_rot + cx, y_rot + cy

def fill_gradient(ax, t, envelope, cx, cy, angle, cmap='rainbow_r',vmin=0.4,vmax=1):
    """Fill under a Gaussian envelope with a red→violet horizontal gradient."""
    
    # Create a 2D image: x = time, y = amplitude
    n_y = 10000
    y_min, y_max = envelope.min(), envelope.max()
   
    
    
    y_vals = np.linspace(-y_max,y_max,n_y)
        
    norm_t = (t - t.min()) / (t.max() - t.min())  # 0→1 across time
    
    colors = plt.get_cmap(cmap)  # shape (len(t), 4)
    cmap_trunc = truncate_cmap(colors,minval=vmin,maxval=vmax)
    
    # Build a 2D grid
    t_grid, y_grid = np.meshgrid(t, y_vals)

    # Rotate the grid
    t_rot, y_rot = rotate_about_center(t_grid, y_grid, cx, cy, angle)

    # Build a color array (same shape as grid)
    color_grid = np.tile(norm_t, (len(y_vals), 1))  # gradient along t ax1is

    # Mask outside envelope
    mask_2d = np.abs(y_grid) > envelope[np.newaxis, :]
    color_grid[mask_2d] = np.nan

    ax.pcolormesh(t_rot, y_rot, color_grid, cmap=cmap_trunc, shading='auto')
    
#-----------------------------------------------------------------------------------
fig, (ax1,ax2) = plt.subplots(1,2,gridspec_kw={"left":0.1,"right":0.98,"wspace":0.2})

figfolder = r"C:\Users\camp06\OneDrive - UBC\Documents\Presentations"
figfilename = figfolder + r"\\slowCFG_arms.png"

t = np.linspace(0,10,1000)
a = 0.5
x0 = 5
sigma = 2
offset = 0
gaussian_envelope = gaussian(t,a,x0,sigma,offset)


t_line = np.array([0,10])
freq_line = np.array([gaussian_envelope[0],gaussian_envelope[-1]])
center_x = 0
center_y = 0
angle1 = 20
angle2 = -20
angle3 = 40

t_line1, freq_line1 = rotate_about_center(t_line,freq_line,center_x,center_y,angle1)
t_line2, freq_line2 = rotate_about_center(t_line,freq_line,center_x,center_y,angle2)
t_line3, freq_line3 = rotate_about_center(t_line,freq_line,center_x,center_y,angle3)

#rotated_envelope = rotate_about_center(t,gaussian_envelope,x0,0,30)

fill_gradient(ax1,t,gaussian_envelope,center_x,center_y,angle=angle1,cmap="rainbow_r",vmin=0.5,vmax=1)
fill_gradient(ax1,t,gaussian_envelope,center_x,center_y,angle=angle2,cmap="rainbow",vmin=0.5,vmax=1)
fill_gradient(ax2,t,gaussian_envelope,center_x,center_y,angle=angle3,cmap="seismic_r",vmin=0,vmax=0.25)

ax1.plot(t_line1,freq_line1,"k--",zorder=100)
ax1.plot(t_line2,freq_line2,"k--",zorder=100)
#ax1.set_aspect('equal')
#ax1.set_ylim(-1,1)
#ax1.tick_params(left=False,bottom=False)
ax1.set_yticks([0])
ax1.set_yticklabels([r'$\omega_0$'])
ax1.set_xticks([])
ax1.set_xticklabels([])


ax1.text(0.25,0.85,r"$\omega_+(t) = \omega_0 + \beta t$",transform=ax1.transAxes,fontsize=6)
ax1.text(0.25,0.15,r"$\omega_-(t) = \omega_0 -\beta t$",transform=ax1.transAxes,fontsize=6)
ax1.set_xlabel(r'Time')
ax1.set_ylabel(r'Optical frequency')
ax1.set_xlim([0,10])
ax1.set_ylim([-6,6])


ax2.plot(t_line3,freq_line3,"k--",zorder=100)
ax2.set_xlabel(r'Time')
ax2.set_ylabel(r'Centrifuge frequency')
ax2.set_xlim([0,8])
ax2.set_ylim([0,8])
ax2.set_yticks([0])
ax2.set_yticklabels([r'0'])
ax2.set_xticks([])
ax2.set_xticklabels([])

ax2.text(0.25,0.65,r"$\Omega_\mathrm{CFG}(t) = 2\beta t$",transform=ax2.transAxes,fontsize=6)

print('xlim = ',ax1.get_xlim())
print('ylim = ',ax1.get_ylim())

fig.savefig(figfilename,dpi=300,format='png',bbox_inches='tight')
plt.show()
