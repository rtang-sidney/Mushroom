import numpy as np
import matplotlib.pyplot as plt
import re
from geometry_context import GeometryContext
from helper import InstrumentContext, distance_point2line, points_to_line, MASS_NEUTRON, PLANCKS_CONSTANT, \
    CONVERSION_JOULE_TO_EV
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
PATTERN_PARAM = re.compile(r"\s*(\S*_?\S*)=([-+]?[0-9]*\.?[0-9]*)")


class PsdInformation:
    def __init__(self, filename):  # psd_type, psd_name,
        geometryctx = GeometryContext(side="same")
        instrumentctx = InstrumentContext()
        f = open(file=filename).readlines()
        keys = []
        contents = []
        for line in f:
            if line.startswith(COMMENT_SYMBOL):
                line = line[2:]
                key, content = line.split(DELIMITER, maxsplit=1)
                key = key.strip()
                content = content.strip()
                if key == KEY_INSTR_PARAM:
                    key, content = re.search(pattern=PATTERN_PARAM, string=content).groups()
                keys.append(key)
                contents.append(content)

        self.metadata_dict = dict(zip(keys, contents))
        self.wavenumber_incoming = float(self.metadata_dict[WAVENUMBER_INCOMING])
        self.xsize, self.ysize = list(
            map(int, re.search(pattern=PATTERN_SIZE, string=self.metadata_dict[KEY_SIZE]).groups()))
        self.xaxis, self.yaxis = self._xy_axes()
        self.xx, self.yy = np.meshgrid(self.xaxis, self.yaxis)
        self.intensities = np.loadtxt(fname=filename, comments=COMMENT_SYMBOL, max_rows=self.ysize)
        # self.azimuthal_angles2d = np.empty_like(self.intensities)
        # self.wavenumbers2d = np.empty_like(self.intensities)
        self.wavevector_transfer, self.energy_transfer = self._position2dispersion(geo_ctx=geometryctx,
                                                                                   instrument=instrumentctx)

        self._position2dispersion(geo_ctx=geometryctx, instrument=instrumentctx)

    def _xy_axes(self):
        xmin, xmax, ymin, ymax = list(
            map(float, re.search(pattern=PATTERN_XYLIMITS, string=self.metadata_dict[KEY_XYLIMITS]).groups()))
        xaxis = np.linspace(start=xmin, stop=xmax, num=self.xsize)
        yaxis = np.linspace(start=ymin, stop=ymax, num=self.ysize)
        return xaxis, yaxis

    def _get_psdcyl_middle(self):
        return float(re.search(pattern=PATTERN_POSITION, string=self.metadata_dict[KEY_POSITION]).group(2))

    def _position2dispersion(self, geo_ctx: GeometryContext, instrument: InstrumentContext):
        value_xvar = self.metadata_dict[KEY_XVAR]
        if value_xvar == PSDCYL_XVAR:
            azimuthal_angles = np.deg2rad(self.xaxis)
            psdcyl_real_y = self._get_psdcyl_middle() - self.yaxis
            # the psdcyl in McStas is defined by the middle point and the height, whereas the position with respect to
            # the sample is needed in our program
            psdcyl_xpos = abs(geo_ctx.detector_line_vert[2])
            wavenumbers = []
            polar_angles = []
            for ypos in psdcyl_real_y:
                point_detector = (psdcyl_xpos, ypos)
                line_detetor_focus = points_to_line(point1=point_detector, point2=geo_ctx.focus_point)
                distances_an_line = distance_point2line(point=geo_ctx.analyser_points, line=line_detetor_focus)
                index_shortest = np.argmin(distances_an_line)
                wavenumber = wavenumber_bragg(geo_ctx=geo_ctx, instrument=instrument, analyser_point=(
                    geo_ctx.analyser_points[0][index_shortest], geo_ctx.analyser_points[1][index_shortest]))
                polar_angle = geo_ctx.polar_angles[index_shortest]
                wavenumbers.append(wavenumber)
                polar_angles.append(polar_angle)
            wavenumbers = np.array(wavenumbers)
            polar_angles = np.array(polar_angles)
            azi2d, kf2d = np.meshgrid(azimuthal_angles, wavenumbers)
            azi2d2, pol2d = np.meshgrid(azimuthal_angles, polar_angles)
            kfx2d = kf2d * np.cos(pol2d) * np.cos(pol2d)
            kfy2d = kf2d * np.cos(pol2d) * np.sin(pol2d)
            kfz2d = kf2d * np.sin(pol2d)
            ki2d = np.full_like(a=kf2d, fill_value=self.wavenumber_incoming)
            q2d = np.linalg.norm([kfx2d - ki2d, kfy2d, kfz2d], axis=0)
            omega2d = PLANCKS_CONSTANT ** 2 / (2.0 * MASS_NEUTRON) * (kf2d ** 2 - ki2d ** 2) / CONVERSION_JOULE_TO_EV
            return q2d, omega2d
        elif self.metadata_dict[KEY_XVAR] == PSD_XVAR:
            pass
        else:
            raise RuntimeError("Cannot match the x variable {}.".format(self.metadata_dict[KEY_XVAR]))

        pass
        # return wavenumbers, polar_angles, azimuthal_angles


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
plt.contourf(psdcyl.wavevector_transfer, psdcyl.energy_transfer * 1e3, psdcyl.intensities)
plt.xlabel(r"Wavevector transfer $Q=k_{i}-k_{f}$ (Angstrom$^{-1}$)")
plt.ylabel(r"Energy transfer $\Delta E = E_{i}-E_{f}$ (meV)")
plt.colorbar()

# psdflat = PsdInformation(filename="/".join([folder, file_psd_flat]))
#
# plt.figure(2)
# plt.contourf(psdflat.xx, psdflat.yy, psdflat.intensities)
# plt.colorbar()

plt.show()
