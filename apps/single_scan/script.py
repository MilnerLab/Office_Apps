from pathlib import Path
from matplotlib.widgets import Button
from matplotlib import pyplot as plt

from Lab_apps._io.dat_finder import DatFinder
from Lab_apps._io.dat_loader import load_time_scan
from Lab_apps.single_scan.domain.plotting import plot_single_scan

'''
file_path = find_most_recent_scanfile()

scan_data = load_time_scan(file_path)

fig, ax = plt.subplots(figsize=(8, 4))
plot_single_scan(ax, scan_data)

ax.legend(loc="upper right")
fig.tight_layout()
plt.show()
'''

def main() -> None:
    file_path = DatFinder().find_most_recent_scanfile()

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.2)  
    button_ax = fig.add_axes((0.8, 0.05, 0.15, 0.075))
    refresh_button = Button(button_ax, "Refresh")

    def on_refresh(event):
        scan_data = load_time_scan(file_path)

        ax.clear()
        plot_single_scan(ax, scan_data)
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    refresh_button.on_clicked(on_refresh)

    plt.show()


if __name__ == "__main__":
    main()
