import pathlib


foldername = "/mnt/valeryshare/Droplets/"
dwqd = pathlib.Path(foldername).glob('*ScanFile.dat')
print(list(dwqd))