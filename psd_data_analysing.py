import numpy as np
import matplotlib.pyplot as plt
import re

PSDCYL_COMPONENT = "psdcyl_monitor_out"
PSD_COMPONENT = "psd_monitor"

PSDCYL_XVAR = "radial"
PSDCYL_YVAR = "yheight"
PSD_XVAR = "X"
PSD_YVAR = "Y"

COMMENT_SYMBOL = "#"
DELIMITER = ":"
KEY_INSTR_PARAM = "Param"  # the string at the begging of the information of instrument parameters
KEY_SIZE = "type"
KEY_XVAR = "xvar"
KEY_YVAR = "yvar"
KEY_XLABEL = "xlabel"
KEY_YLABEL = "ylabel"
KEY_XYLIMITS = "xylimits"
REVEVANT_KEYS = [KEY_INSTR_PARAM, KEY_SIZE, KEY_XVAR, KEY_YVAR, KEY_XLABEL, KEY_YLABEL, KEY_XYLIMITS]

WAVENUMBER_INCOMING = "ki"
PATTERN_SIZE = re.compile(r"([0-9]*),\s?([0-9]*)")
PATTERN_XYLIMITS = re.compile(
    r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")


class PsdInformation:
    def __init__(self, filename):  # psd_type, psd_name,
        f = open(file=filename).readlines()
        keys = []
        contents = []
        for line in f:
            if line.startswith(COMMENT_SYMBOL):
                line = line[2:]
                key, content = line.split(DELIMITER, maxsplit=1)
                keys.append(key)
                contents.append(content)

        self.metadata_dict = dict(zip(keys, contents))
        self.xsize, self.ysize = list(
            map(int, re.search(pattern=PATTERN_SIZE, string=self.metadata_dict[KEY_SIZE]).groups()))
        self.xx, self.yy = self._xy_axes()
        self.intensities = np.loadtxt(fname=filename, comments=COMMENT_SYMBOL, max_rows=self.ysize)

    def _xy_axes(self):
        xmin, xmax, ymin, ymax = list(
            map(float, re.search(pattern=PATTERN_XYLIMITS, string=self.metadata_dict[KEY_XYLIMITS]).groups()))
        xaxis = np.linspace(start=xmin, stop=xmax, num=self.xsize)
        yaxis = np.linspace(start=ymin, stop=ymax, num=self.ysize)
        return np.meshgrid(xaxis, yaxis)


def folder_name(instrument, day, month, year, hms):
    if isinstance(instrument, str):
        if isinstance(day, int):
            if isinstance(month, int):
                if isinstance(year, int):
                    if isinstance(hms, int):
                        pass
                    else:
                        raise RuntimeError("Invalid type of time of day.")
                else:
                    raise RuntimeError("Invalid type of year.")
            else:
                raise RuntimeError("Invalid type of month.")
        else:
            raise RuntimeError("Invalid type of day.")
    else:
        raise RuntimeError("Invalid type of instrument name.")
    day = str(day)
    month = str(month)
    year = str(year)
    hms = str(hms)

    if len(day) == 1:
        day = "0" + day
    if len(month) == 1:
        month = "0" + month
    if len(day) == len(month) == 2 and len(year) == 4 and len(hms) == 6:
        date = "".join([year, month, day])
        return "_".join([instrument, date, hms])
    else:
        raise RuntimeError("Invalid length of the date parameters.")


instrument_name = "Mushroom_MM"
folder_day = 25
folder_month = 6
folder_year = 2020
folder_hms = 162941
folder = folder_name(instrument=instrument_name, day=folder_day, month=folder_month, year=folder_year, hms=folder_hms)

file_psd_cyl = "psd_cyl.dat"
file_psd_flat = "psd_flat.dat"

cyl_centre = -1.05  # vertical position of the centre of the cylindrical PSD
psdcyl = PsdInformation(filename="/".join([folder, file_psd_cyl]))

plt.figure(1)
plt.contourf(psdcyl.xx, psdcyl.yy, psdcyl.intensities)
plt.colorbar()

psdflat = PsdInformation(filename="/".join([folder, file_psd_flat]))

plt.figure(2)
plt.contourf(psdflat.xx, psdflat.yy, psdflat.intensities)
plt.colorbar()

plt.show()
