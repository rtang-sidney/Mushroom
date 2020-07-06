import numpy as np
import matplotlib.pyplot as plt
import re
from geometry_context import GeometryContext
from helper import InstrumentContext, distance_point2line, points_to_line
from secondary_spectrometer import wavenumber_bragg

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
KEY_POSITION = "position"
REVEVANT_KEYS = [KEY_INSTR_PARAM, KEY_SIZE, KEY_XVAR, KEY_YVAR, KEY_XLABEL, KEY_YLABEL, KEY_XYLIMITS, KEY_POSITION]

WAVENUMBER_INCOMING = "ki"
PATTERN_SIZE = re.compile(r"([0-9]*),\s?([0-9]*)")
PATTERN_XYLIMITS = re.compile(
    r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_POSITION = re.compile(r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")


class PsdInformation():
    def __init__(self, filename, geometryctx: GeometryContext, instrumentctx=InstrumentContext):  # psd_type, psd_name,
        f = open(file=filename).readlines()
        keys = []
        contents = []
        for line in f:
            if line.startswith(COMMENT_SYMBOL):
                line = line[2:]
                key, content = line.split(DELIMITER, maxsplit=1)
                key = key.strip()
                content = content.strip()
                keys.append(key)
                contents.append(content)

        self.metadata_dict = dict(zip(keys, contents))
        self.xsize, self.ysize = list(
            map(int, re.search(pattern=PATTERN_SIZE, string=self.metadata_dict[KEY_SIZE]).groups()))
        self.xaxis, self.yaxis = self._xy_axes()
        self.xx, self.yy = np.meshgrid(self.xaxis, self.yaxis)
        self.intensities = np.loadtxt(fname=filename, comments=COMMENT_SYMBOL, max_rows=self.ysize)
        self.azimuthal_angles2d, self.wavenumbers2d = np.empty_like(self.intensities)
        self.wavenumber_f = np.empty_like(self.intensities)
        self._position2kf(geo_ctx=geometryctx, instrument=instrumentctx)

    def _xy_axes(self):
        xmin, xmax, ymin, ymax = list(
            map(float, re.search(pattern=PATTERN_XYLIMITS, string=self.metadata_dict[KEY_XYLIMITS]).groups()))
        xaxis = np.linspace(start=xmin, stop=xmax, num=self.xsize)
        yaxis = np.linspace(start=ymin, stop=ymax, num=self.ysize)
        return xaxis, yaxis

    def _get_psdcyl_middle(self):
        return re.search(pattern=PATTERN_POSITION, string=self.metadata_dict[KEY_POSITION]).group(2)

    def _position2kf(self, geo_ctx: GeometryContext, instrument: InstrumentContext):

        value_xvar = self.metadata_dict[KEY_XVAR]
        if self.metadata_dict[KEY_XVAR] == PSDCYL_XVAR:
            azimuthal_angles = np.deg2rad(self.xaxis)

            psdcyl_real_y = float(self._get_psdcyl_middle()) - self.yaxis
            # the psdcyl in McStas is defined by the middle point and the height, whereas the position with respect to
            # the sample is needed in our program
            psdcyl_xpos = abs(geo_ctx.detector_line_vert[2])
            wavenumbers = []
            for ypos in psdcyl_real_y:
                point_detector = (psdcyl_xpos, ypos)
                line_detetor_focus = points_to_line(point1=point_detector, point2=geo_ctx.focus_point)
                distances_an_line = distance_point2line(point=geo_ctx.analyser_points, line=line_detetor_focus)
                index_shortest = np.argmin(distances_an_line)
                wavenumber = wavenumber_bragg(geo_ctx=geo_ctx, instrument=instrument, analyser_point=(
                    geo_ctx.analyser_points[0][index_shortest], geo_ctx.analyser_points[1][index_shortest]))
                wavenumbers.append(wavenumber)
            wavenumbers = np.array(wavenumbers)
            return np.meshgrid(azimuthal_angles,wavenumbers)
        elif self.metadata_dict[KEY_XVAR] == PSD_XVAR:
            pass
        else:
            raise RuntimeError("Cannot match the x variable {}.".format(self.metadata_dict[KEY_XVAR]))


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
