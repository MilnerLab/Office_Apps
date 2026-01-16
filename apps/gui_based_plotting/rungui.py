from pathlib import Path
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from numpy import test

#Import by module...
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scan, load_time_scans 

from apps.single_scan.domain.plotting import plot_single_scan

from apps.scan_averaging.domain.plotting import plot_averaged_scan 
from apps.scan_averaging.domain.averaging import average_scans 
from apps.scan_averaging.domain.models import AveragedScansData


def main() -> None:
    file_path = DatFinder().find_most_recent_scanfile()
    
    folder_path = Path(r"Z:\Droplets\20251215\DScan4+5") #in the loop because we will have a selector in the gui
    file_paths = DatFinder(folder_path).find_scanfiles()

    averagedScanData = average_scans(load_time_scans(file_paths))

    #file_path = Path("Z:\\Droplets\\20251210132842_ScanFile.dat")

    fig =  plt.figure(figsize=(8, 4)) #we are going to replace with pyqtgraph...
    
    button_size = (0.6, 0.05, 0.3, 0.075)
    button_text = "Load Updated Data"
    button_ax = fig.add_axes(button_size)
    refresh_button = Button(button_ax, button_text)

    def on_refresh(event):
        
        #When there's a GUI for loading datasets, this stuff will definitely end up in some function/method...
        scan_data = load_time_scan(file_path)
        averagedScanData = average_scans(load_time_scans(file_paths))
  
        #preparing the figure for drawing will be in a function too
        allax = fig.axes 
        if len(allax)>1: #delete all axes except the button axis
            #should be written properly instead of one at a time :)
            #Expect this will change a lot with pyqtgraph 
            fig.delaxes(allax[1])
            fig.delaxes(allax[2])
     
        axs = fig.subplots(2,sharex=True) #number of subpots should be picked by number of checkboxes of what we want plotted
        plt.subplots_adjust(hspace = 0,bottom=button_size[2]) #make space for button
        

        '''
        need to programmatically pick what to plot based on checkboxes
        additional plot options:
        signal vs time (c2t, averaged c2t, ions, averaged ions etc...)
        signal vs frequency (regular FT/ power spectrum)
        f vs t using STFT or hilbert etc...
        
        Two options would be nice: 
        "all from the same measurement on the same plot, but different plots for each measurement"
        and 
        "each metric gets its own plot, with different measurements in different colours to compare
        
        
        There will then also need to be inputs like: 
        moving average, STFT window size, window type, frequency units
        
        and ones that aren't about calculations like axis limits, window aspect ratio, error bars vs shading
        
        and it should also have an export figure option (~1080p .png or PRL/PRA-formatted PDF),
        but also an "export figure generation" option that saves a proprietary config type thing.
        
        The file browser thing should have a "pick folder" button to make it pop up, 
        and then from there when you select a folder it gets added as a new "measurement" with a new colour,
        but then the GUI should have a simple way to change the colour of each measurement, 
        and measurements with the same colour chosen should get combined when plotted as if they're 1 measurement.
        
        This seems like an easy UI option for an arbitrary number of measurements. 
        
        There should also be a really easy way to unload datasets, and to unload *all* datasets.
        
        
        '''
        plot_single_scan(axs[0], scan_data)
        plot_averaged_scan(axs[1], averagedScanData)
        fig.canvas.draw_idle()

        

    refresh_button.on_clicked(on_refresh)

    plt.show()


if __name__ == "__main__":
    main()
