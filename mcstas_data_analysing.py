import numpy as np
import matplotlib.pyplot as plt
import re
from geometry_context import GeometryContext
from helper import InstrumentContext, distance_point2line, points_to_line, MASS_NEUTRON, PLANCKS_CONSTANT, \
    CONVERSION_JOULE_TO_EV
import sys

PSDCYL_COMPONENT = "psdcyl_monitor_out"
PSD_COMPONENT = "psd_monitor"
PSDVERT_COMPONENT = "psd_monitor_vertical"
LMON_COMP = "l_monitor"

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
KEY_COMPONENT = "component"
KEY_XUNIT = "xunit"
KEY_YUNIT = "Yunit"
REVEVANT_KEYS = [KEY_INSTR_PARAM, KEY_SIZE, KEY_XVAR, KEY_YVAR, KEY_XLABEL, KEY_YLABEL, KEY_XYLIMITS, KEY_POSITION,
                 KEY_COMPONENT]

UNIT_CENTIMETRE = "cm"
UNIT_METRE = "m"

WAVENUMBER_INCOMING = "ki"
PATTERN_SIZE = re.compile(r"([0-9]*),\s?([0-9]*)")
PATTERN_XYLIMITS = re.compile(
    r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_POSITION = re.compile(r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_PARAM = re.compile(r"\s*(\S*_?\S*)=([-+]?[0-9]*\.?[0-9]*)")
PATTERN_LABEL = re.compile(r"\s*(\S*\s*\S*)\s*\[(\S*)\]")


class PsdInformation:
    def __init__(self, angle, file_psd):  # psd_type, psd_name,
        self.scan_angle = angle
        folder = self.angle_to_folder(angle=self.scan_angle)
        filename = "/".join([folder, file_psd])
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
                elif key == KEY_XLABEL:
                    label, unit = re.search(pattern=PATTERN_LABEL, string=content).groups()
                    key = [KEY_XLABEL, KEY_XUNIT]
                    content = [label, unit]
                elif key == KEY_YLABEL:
                    label, unit = re.search(pattern=PATTERN_LABEL, string=content).groups()
                    key = [KEY_YLABEL, KEY_YUNIT]
                    content = [label, unit]
                if type(key) == list:
                    keys.extend(key)
                    contents.extend(content)
                else:
                    keys.append(key)
                    contents.append(content)

        self.metadata_dict = dict(zip(keys, contents))
        # print(self.metadata_dict)
        # self.wavenumber_incoming = float(self.metadata_dict[WAVENUMBER_INCOMING]) * 1e10
        self.wavenumber_incoming = np.pi / (instrumentctx.lattice_distance_pg002 * np.sin(np.deg2rad(67.8819704289675)))
        self.xsize, self.ysize = list(
            map(int, re.search(pattern=PATTERN_SIZE, string=self.metadata_dict[KEY_SIZE]).groups()))
        x_1d, y_1d = self._xy_axes()
        # self.xx, self.yy = np.meshgrid(self.xaxis, self.yaxis)
        intensities = np.loadtxt(fname=filename, comments=COMMENT_SYMBOL, max_rows=self.ysize)
        self.x_1d, self.y_1d, self.intensities = self._psd_signal_adjust(x_1d, y_1d, intensities)

        # self.azimuthal_angles2d = np.empty_like(self.intensities)
        # self.wavenumbers2d = np.empty_like(self.intensities)
        # self.wavevector_transfer_x, self.wavevector_transfer_y, self.wavevector_transfer_z, self.energy_transfer \
        # , self.intensities = self._position2dispersion(intensities, geo_ctx=geometryctx, instrument=instrumentctx)
        # self.wavevector_transfer, self.energy_transfer, self.intensities = self.position2dispersion(intensities,
        #                                                                                             geo_ctx=geometryctx,
        #                                                                                             instrument=instrumentctx)

        # self._position2dispersion(geo_ctx=geometryctx, instrument=instrumentctx)

    def angle_to_folder(self, angle):
        print(angle)
        scanname = "".join([SCAN_FOLDER_PREFIX, str(int(angle))])
        folder = "/".join([FOLDER_DATE, scanname])
        return folder

    def rotation2d(self, x, y, angle):
        new_x = x * np.cos(angle) - y * np.sin(angle)
        new_y = x * np.sin(angle) + y * np.cos(angle)
        return new_x, new_y

    def _xy_axes(self):
        xmin, xmax, ymin, ymax = list(
            map(float, re.search(pattern=PATTERN_XYLIMITS, string=self.metadata_dict[KEY_XYLIMITS]).groups()))
        xaxis = np.linspace(start=xmin, stop=xmax, num=self.xsize)
        yaxis = np.linspace(start=ymin, stop=ymax, num=self.ysize)
        if self.metadata_dict[KEY_XUNIT] == UNIT_CENTIMETRE:
            xaxis *= 1e-2
        if self.metadata_dict[KEY_YUNIT] == UNIT_CENTIMETRE:
            yaxis *= 1e-2
        return xaxis, yaxis

    def _get_psdcyl_middle(self):
        return float(re.search(pattern=PATTERN_POSITION, string=self.metadata_dict[KEY_POSITION]).group(2))

    def _psd_signal_adjust(self, x_1d, y_1d, intensity_2d):
        component = self.metadata_dict[KEY_COMPONENT]
        if component == PSDCYL_COMPONENT:
            x_1d = np.deg2rad(x_1d)
            y_1d = self._get_psdcyl_middle() + y_1d
        elif component == PSD_COMPONENT:
            # x and y axes are exchanged in McStas
            # x_1d, y_1d = y_1d * 1e-2, x_1d * 1e-2
            # intensity_2d = np.transpose(intensity_2d)
            x_2d, y_2d = np.meshgrid(x_1d, y_1d)
            intensity_2d = np.where(np.linalg.norm([x_2d, y_2d], axis=0) > 0.4, intensity_2d, 0)
        elif component == PSDVERT_COMPONENT:
            x_1d = np.append(np.linspace(start=-170, stop=-5, num=166), np.linspace(start=5, stop=170, num=166))
            y_1d = self._get_psdcyl_middle() + y_1d
            intensity_1d = np.mean(intensity_2d, axis=1)
            azi_index = np.searchsorted(x_1d, self.scan_angle)
            # print(azi_index)
            intensity_2d = np.zeros((y_1d.shape[0], x_1d.shape[0]))
            # print(type(x_1d), "\n", type(y_1d))
            # print(np.meshgrid(x_1d, y_1d))
            # print(x_1d.shape, y_1d.shape, intensity_2d.shape)
            intensity_2d[:, azi_index] = intensity_1d
            x_1d = np.deg2rad(x_1d)
        else:
            raise RuntimeError("Cannot match the x variable {}.".format(self.metadata_dict[KEY_XVAR]))
        return x_1d, y_1d, intensity_2d


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


def position2dispersion(x_1d, y_1d, intensity_2d, psd_comp, ki, geo_ctx: GeometryContext):
    azi_1d = np.linspace(-np.pi, np.pi, num=361)
    azi_2d, pol_2d = np.meshgrid(azi_1d, geo_ctx.polar_angles)
    inten_new_2d = np.zeros_like(azi_2d)
    if psd_comp == PSDCYL_COMPONENT:
        azi_1d_indices = np.searchsorted(azi_1d, x_1d)
        y_1d_indices = np.searchsorted(geo_ctx.dete_vert_y, y_1d) + geo_ctx.dete_hori_x.shape[0]
        azi_indices_2d, y_indices_2d = np.meshgrid(azi_1d_indices, y_1d_indices)
        np.add.at(inten_new_2d, (y_indices_2d, azi_indices_2d), intensity_2d)
    elif psd_comp == PSD_COMPONENT:
        x_2d, y_2d = np.meshgrid(x_1d, y_1d)
        radii_2d = np.linalg.norm([x_2d, y_2d], axis=0)
        intensity_2d = np.where(radii_2d > abs(geo_ctx.detector_line_vert[-1]), intensity_2d, 0)
        radii_indices = geo_ctx.dete_hori_x.shape[0] - np.searchsorted(geo_ctx.dete_hori_x[::-1], radii_2d)

        azi_indices = np.searchsorted(azi_1d, np.arctan2(y_2d, x_2d))
        azi_indices = np.where(azi_indices < azi_1d.shape[0], azi_indices, azi_1d.shape[0] - 1)
        np.add.at(inten_new_2d, (radii_indices, azi_indices), intensity_2d)
    elif psd_comp == PSDVERT_COMPONENT:
        x_1d_indices = np.arange(x_1d.shape[0])
        y_1d_indices = np.searchsorted(geo_ctx.dete_vert_y, y_1d) + geo_ctx.dete_hori_x.shape[0]
        x_indices_2d, y_indices_2d = np.meshgrid(x_1d_indices, y_1d_indices)
        np.add.at(inten_new_2d, (y_indices_2d, x_1d_indices), intensity_2d)
    else:
        raise RuntimeError("Cannot match the component {:s}.".format(psd_comp))
    kf_2d = np.transpose(np.full_like(np.transpose(azi_2d), fill_value=geo_ctx.wavenumbers))
    kfx_2d = kf_2d * np.cos(pol_2d) * np.cos(azi_2d)
    kfy_2d = kf_2d * np.cos(pol_2d) * np.sin(azi_2d)
    kfz_2d = kf_2d * np.sin(pol_2d)
    qx_2d = ki - kfx_2d
    qy_2d = -kfy_2d
    qz_2d = -kfz_2d
    q_2d = np.linalg.norm([qx_2d, qy_2d, qz_2d], axis=0)
    omega_ev_2d = PLANCKS_CONSTANT ** 2 / (2.0 * MASS_NEUTRON) * (ki ** 2 - kf_2d ** 2) / CONVERSION_JOULE_TO_EV
    return kf_2d, q_2d, omega_ev_2d, inten_new_2d


def intensity_update(angle, intensities_flat, intensities_vertical):  # intensities_cyl
    # intensities_cyl += PsdInformation(angle, FILE_PSD_CYL).intensities
    intensities_flat += PsdInformation(angle, FILE_PSD_FLAT).intensities
    intensities_vertical += PsdInformation(angle, FILE_PSD_VERTICAL).intensities
    # return intensities_cyl, intensities_flat, intensities_vertical
    return intensities_flat, intensities_vertical


def plot_ax_2d(fig, ax, x_2d, y_2d, z_2d, psd_comp, subplot_index=None):
    if np.count_nonzero(z_2d) > 0:
        cnt = ax.contour(x_2d, y_2d, z_2d, levels=np.linspace(np.max(z_2d) / 10.0, np.max(z_2d), num=10),
                         cmap="coolwarm")
        fig.colorbar(cnt, ax=ax)
    if psd_comp == PSDCYL_COMPONENT or psd_comp == PSDVERT_COMPONENT:
        xlabel = r"Azimuthal angle $\theta$ (deg)"
        ylabel = "Vertical position (m)"
    elif psd_comp == PSD_COMPONENT:
        if subplot_index == 0:
            xlabel = "x position (m)"
            ylabel = "y position (m)"
            ax.set_title("Cartesian")
        elif subplot_index == 1:
            xlabel = "Angular position (deg)"
            ylabel = "Radial position (m)"
            ax.set_title("Radial")
        else:
            raise RuntimeError("Wrong index or no index for flat psd plot given.")
    else:
        raise RuntimeError("Invalid type of PSD given.")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()


np.set_printoptions(threshold=sys.maxsize, precision=2)

FILE_PSD_CYL = "psd_cyl.dat"
FILE_PSD_FLAT = "psd_flat.dat"
FILE_PSD_VERTICAL = "psd_vertical.dat"
FILE_L_FLAT = "l_flat.dat"
FILE_L_VERTICAL = "l_vertical.dat"

# FOLDER_DATE = "090720_Overnight"
# FOLDER_DATE = "130720_Overnight"
# FOLDER_DATE = "160720_Overnight"
# FOLDER_DATE = "230720_Day"
# FOLDER_DATE = "230720_Graphite"
# FOLDER_DATE = "230720_PGOvernight"
# FOLDER_DATE = "240720_PGDay"
# FOLDER_DATE = "240720_VDay"
FOLDER_DATE = "240720_PGNight"
# FOLDER_DATE = "240720_VNight"

SCAN_FOLDER_PREFIX = "Angle"
azimuthal_angles = np.linspace(start=5, stop=170, num=166)

angle = azimuthal_angles[0]
# psd_cyl = PsdInformation(angle, FILE_PSD_CYL)
# azi_angles_cyl, y_positions_cyl, intensities_cyl, ki = psd_cyl.x_1d, psd_cyl.y_1d, psd_cyl.intensities, psd_cyl.wavenumber_incoming
psd_flat = PsdInformation(angle, FILE_PSD_FLAT)
x_pos_flat, y_pos_flat, intensities_flat, ki = psd_flat.x_1d, psd_flat.y_1d, psd_flat.intensities, psd_flat.wavenumber_incoming
psd_vertical = PsdInformation(angle, FILE_PSD_VERTICAL)
x_pos_vertical, y_pos_vertical, intensities_vertical = psd_vertical.x_1d, psd_vertical.y_1d, psd_vertical.intensities

intensities_flat, intensities_vertical = intensity_update(angle=-angle,
                                                          intensities_flat=intensities_flat,
                                                          intensities_vertical=intensities_vertical)

for angle in azimuthal_angles[1:]:
    intensities_flat, intensities_vertical = intensity_update(angle=angle, intensities_flat=intensities_flat,
                                                              intensities_vertical=intensities_vertical)
    intensities_flat, intensities_vertical = intensity_update(angle=-angle, intensities_flat=intensities_flat,
                                                              intensities_vertical=intensities_vertical)

print("ki={}".format(ki * 1e-10))
# azi_cyl_2d, y_cyl_2d = np.meshgrid(azi_angles_cyl, y_positions_cyl)
# fig, ax = plt.subplots()
# plot_ax_2d(fig=fig, ax=ax, x_2d=np.rad2deg(azi_cyl_2d), y_2d=y_cyl_2d, z_2d=intensities_cyl, psd_comp=PSDCYL_COMPONENT)
# plt.title("Total intensities on cyl. PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Data_collection_cyl.pdf"]))
# plt.close(fig)

# azi_vert_2d, y_vert_2d = np.meshgrid(x_pos_vertical, y_pos_vertical)
# fig, ax = plt.subplots()
# plot_ax_2d(fig=fig, ax=ax, x_2d=np.rad2deg(azi_vert_2d), y_2d=y_vert_2d, z_2d=intensities_vertical,
#            psd_comp=PSDVERT_COMPONENT)
# plt.title("Total intensities on vert. PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Data_collection_vert.pdf"]))
# plt.close(fig)
#
# x_flat_2d, y_flat_2d = np.meshgrid(x_pos_flat, y_pos_flat)
# azi_flat_2d = np.rad2deg(np.arctan2(y_flat_2d, x_flat_2d))
# radi_flat_2d = np.linalg.norm([x_flat_2d, y_flat_2d], axis=0)
# xplot = [x_flat_2d, azi_flat_2d]
# yplot = [y_flat_2d, radi_flat_2d]
# fig, axs = plt.subplots(2, 1)
# for i in range(axs.ravel().shape[0]):
#     plot_ax_2d(fig=fig, ax=axs[i], x_2d=xplot[i], y_2d=yplot[i], z_2d=intensities_flat, psd_comp=PSD_COMPONENT,
#                subplot_index=i)
# plt.suptitle("Total intensities on flat PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("_".join([FOLDER_DATE, "Data_collection_flat.pdf"]))
# plt.close(fig)

geometryctx = GeometryContext(side="same")
# kf_cyl_2d, wavevector_transfer_cyl, energy_transfer_cyl, intensities_new_cyl = position2dispersion(x_1d=azi_angles_cyl,
#                                                                                                    y_1d=y_positions_cyl,
#                                                                                                    intensity_2d=intensities_cyl,
#                                                                                                    psd_comp=PSDCYL_COMPONENT,
#                                                                                                    ki=ki,
#                                                                                                    geo_ctx=geometryctx)
#
# fig, ax = plt.subplots()
# cnt = ax.contour(wavevector_transfer_cyl * 1e-10, energy_transfer_cyl * 1e3, intensities_new_cyl,
#                  levels=np.linspace(np.max(intensities_new_cyl) / 10.0, np.max(intensities_new_cyl), num=10),
#                  cmap="coolwarm")
# ax.set_xlabel(r"$Q = k_i - k_f$ ($\AA^{-1}$)")
# if np.count_nonzero(intensities_new_cyl) > 0:
#     fig.colorbar(cnt, ax=ax)
# ax.set_ylabel(r"Energy transfer $\Delta E = E_{i}-E_{f}$ (meV)")
# plt.title("Total intensities on cyl. PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.grid()
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Dispersion_cyl.pdf"]))
# plt.close(fig)

kf_vert_2d, wavevector_transfer_vert, energy_transfer_vert, intensities_new_vert = position2dispersion(
    x_1d=x_pos_vertical,
    y_1d=y_pos_vertical,
    intensity_2d=intensities_vertical,
    psd_comp=PSDVERT_COMPONENT,
    ki=ki,
    geo_ctx=geometryctx)


# fig, ax = plt.subplots()
# if np.count_nonzero(intensities_new_vert) > 0:
#     cnt = ax.contour(wavevector_transfer_vert * 1e-10, energy_transfer_vert * 1e3, intensities_new_vert,
#                      levels=np.linspace(np.max(intensities_new_vert) / 10.0, np.max(intensities_new_vert), num=10),
#                      cmap="coolwarm")
#     fig.colorbar(cnt, ax=ax)
# ax.set_xlabel(r"$Q = k_i - k_f$ ($\AA^{-1}$)")
# ax.set_ylabel(r"Energy transfer $\Delta E = E_{i}-E_{f}$ (meV)")
# plt.title("Dispersion on vert. PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.grid()
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Dispersion_vert.pdf"]))
# plt.close(fig)
#
# kf_flat_2d, wavevector_transfer_flat, energy_transfer_flat, intensities_new_flat = position2dispersion(
#     x_1d=x_pos_flat,
#     y_1d=y_pos_flat,
#     intensity_2d=intensities_flat,
#     psd_comp=PSD_COMPONENT,
#     ki=ki,
#     geo_ctx=geometryctx)
#
# fig, ax = plt.subplots()
# if np.count_nonzero(intensities_new_flat) > 0:
#     cnt = ax.contour(wavevector_transfer_flat * 1e-10, energy_transfer_flat * 1e3, intensities_new_flat,
#                      levels=np.linspace(np.max(intensities_new_flat) / 10.0, np.max(intensities_new_flat), num=10),
#                      cmap="coolwarm")
#     fig.colorbar(cnt, ax=ax)
# ax.set_xlabel(r"$Q = k_i - k_f$ ($\AA^{-1}$)")
# ax.set_ylabel(r"Energy transfer $\Delta E = E_{i}-E_{f}$ (meV)")
# plt.title("Dispersion on flat PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.grid()
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Dispersion_flat.pdf"]))
# plt.close(fig)


def get_wavelength_signals(scan_angles):
    angle = scan_angles[0]
    scanname = "".join([SCAN_FOLDER_PREFIX, str(int(angle))])
    folder = "/".join([FOLDER_DATE, scanname])
    # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
    signals = np.loadtxt(fname="/".join([folder, FILE_L_VERTICAL]), comments=COMMENT_SYMBOL)
    wavelengths = signals[:, 0] * 1e-10
    intentisities = signals[:, 1]

    scanname = "".join([SCAN_FOLDER_PREFIX, str(int(-angle))])
    folder = "/".join([FOLDER_DATE, scanname])
    # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
    signals = np.loadtxt(fname="/".join([folder, FILE_L_VERTICAL]), comments=COMMENT_SYMBOL)
    intentisities += signals[:, 1]

    for angle in scan_angles[1:]:
        scanname = "".join([SCAN_FOLDER_PREFIX, str(int(angle))])
        folder = "/".join([FOLDER_DATE, scanname])
        # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
        signals = np.loadtxt(fname="/".join([folder, FILE_L_VERTICAL]), comments=COMMENT_SYMBOL)
        intentisities += signals[:, 1]
        scanname = "".join([SCAN_FOLDER_PREFIX, str(int(-angle))])
        folder = "/".join([FOLDER_DATE, scanname])
        # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
        signals = np.loadtxt(fname="/".join([folder, FILE_L_VERTICAL]), comments=COMMENT_SYMBOL)
        intentisities += signals[:, 1]
    return wavelengths, intentisities


def plot_kf_monitor(ax, kf, intensity, title):
    # kf_sort_index = np.argsort(kf)
    # ax.plot(kf[kf_sort_index] * 1e-10, intensity[kf_sort_index])
    ax.hist(kf * 1e-10, bins=100, range=(0.8, 1.3), weights=intensity)
    ax.grid()
    ax.set_title(title)
    # ax.set_xlim(0.8, 1.3)
    ax.set_ylabel("Intensity")


wavelength_lmonitor, intensity_lmonitor = get_wavelength_signals(scan_angles=azimuthal_angles)
wavenumber_lmonitor = 2.0 * np.pi / wavelength_lmonitor
# fig, axs = plt.subplots(3, 1, sharex="all")
# titles = ["lambda monitor", "vert. psd", "flat psd"]
# wavenumbers = [wavenumber_lmonitor, kf_vert_2d.flatten(), kf_flat_2d.flatten()]
# intensities = [intensity_lmonitor, intensities_new_vert.flatten(), intensities_new_flat.flatten()]
# for i in range(axs.ravel().shape[0]):
#     plot_kf_monitor(axs[i], wavenumbers[i], intensities[i], titles[i])
# axs[-1].set_xlabel(r"$k_f$ ($\AA^{-1}$)")
# fig.suptitle("Wavenumber from different monitors at ki={:5.3f}".format(ki * 1e-10))
# plt.grid()
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("_".join([FOLDER_DATE, "Wavenumber distribution.pdf"]))
# plt.close(fig)

fig, axs = plt.subplots(2, 1, sharex="all")
titles = ["lambda monitor", "vert. psd"]
wavenumbers = [wavenumber_lmonitor, kf_vert_2d.flatten()]
intensities = [intensity_lmonitor, intensities_new_vert.flatten()]
for i in range(axs.ravel().shape[0]):
    plot_kf_monitor(axs[i], wavenumbers[i], intensities[i], titles[i])
axs[-1].set_xlabel(r"$k_f$ ($\AA^{-1}$)")
fig.suptitle("Wavenumber from different monitors at ki={:5.3f}".format(ki * 1e-10))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("_".join([FOLDER_DATE, "Wavenumber_distribution.pdf"]))
# plt.show()
plt.close(fig)
print("{} is finished.".format(FOLDER_DATE))
