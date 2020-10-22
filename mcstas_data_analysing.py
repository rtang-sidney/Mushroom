import numpy as np
import matplotlib.pyplot as plt
import re
from geometry_context import GeometryContext
from helper import InstrumentContext, MASS_NEUTRON, PLANCKS_CONSTANT, \
    CONVERSION_JOULE_PER_EV, dispersion_signal
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
        geometryctx = GeometryContext()
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
        # self.wavenumber_incoming = np.pi / (
        #         instrumentctx.lattice_distance_pg002 * np.sin(np.deg2rad(31.675315000373345)))
        self.xsize, self.ysize = list(
            map(int, re.search(pattern=PATTERN_SIZE, string=self.metadata_dict[KEY_SIZE]).groups()))
        x_1d, y_1d = self._xy_axes()
        # self.xx, self.yy = np.meshgrid(self.xaxis, self.yaxis)
        intensities = np.loadtxt(fname=filename, comments=COMMENT_SYMBOL, max_rows=self.ysize)
        self.x_1d, self.y_1d, self.intensities = self._psd_signal_adjust(x_1d, y_1d, intensities, geo_ctx=geometryctx)

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

    def _psd_signal_adjust(self, x_1d, y_1d, intensity_2d, geo_ctx: GeometryContext):
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
            x_1d = np.arctan2(x_1d, abs(geo_ctx.detector_line_vert[-1])) + np.deg2rad(self.scan_angle)
            scanx_1d = np.deg2rad(np.linspace(-180, 180, num=361))
            y_1d = self._get_psdcyl_middle() + y_1d
            # intensity_1d = np.mean(intensity_2d, axis=1)
            azi_index = np.searchsorted(scanx_1d, x_1d)
            # print(azi_index)
            intensitynew_2d = np.zeros((y_1d.shape[0], scanx_1d.shape[0]))
            # print(type(x_1d), "\n", type(y_1d))
            # print(np.meshgrid(x_1d, y_1d))
            # print(x_1d.shape, y_1d.shape, intensity_2d.shape)
            index_2d = np.meshgrid(azi_index, np.arange(intensitynew_2d.shape[0]))
            # print(intensitynew_2d.shape)
            np.add.at(intensitynew_2d, (index_2d[1], index_2d[0]), intensity_2d)
            x_1d = scanx_1d
            intensity_2d = intensitynew_2d
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


def position2dispersion(x, y, psd_comp, geo_ctx: GeometryContext):
    ki = geo_ctx.wavenumber_in
    # azi_1d = np.linspace(-np.pi, np.pi, num=361)
    # azi_2d, pol_2d = np.meshgrid(azi_1d, geo_ctx.polar_angles)
    # inten_new_2d = np.zeros_like(azi_2d)
    # if psd_comp == PSDCYL_COMPONENT:
    #     azi_1d_indices = np.searchsorted(azi_1d, x)
    #     y_1d_indices = np.searchsorted(geo_ctx.dete_vert_y, y) + geo_ctx.dete_hori_x.shape[0]
    #     azi_indices_2d, y_indices_2d = np.meshgrid(azi_1d_indices, y_1d_indices)
    #     np.add.at(inten_new_2d, (y_indices_2d, azi_indices_2d), intensity_2d)
    # elif psd_comp == PSD_COMPONENT:
    #     x_2d, y_2d = np.meshgrid(x, y)
    #     radii_2d = np.linalg.norm([x_2d, y_2d], axis=0)
    #     intensity_2d = np.where(radii_2d > abs(geo_ctx.detector_line_vert[-1]), intensity_2d, 0)
    #     radii_indices = geo_ctx.dete_hori_x.shape[0] - np.searchsorted(geo_ctx.dete_hori_x[::-1], radii_2d)
    #
    #     azi_indices = np.searchsorted(azi_1d, np.arctan2(y_2d, x_2d))
    #     azi_indices = np.where(azi_indices < azi_1d.shape[0], azi_indices, azi_1d.shape[0] - 1)
    #     np.add.at(inten_new_2d, (radii_indices, azi_indices), intensity_2d)
    # elif psd_comp == PSDVERT_COMPONENT:
    #     x_1d_indices = np.arange(x.shape[0])
    #     y_1d_indices = np.searchsorted(geo_ctx.dete_vert_y, y) + geo_ctx.dete_hori_x.shape[0]
    #     y_1d_indices = np.where(y_1d_indices < geo_ctx.polar_angles.shape[0], y_1d_indices,
    #                             geo_ctx.polar_angles.shape[0] - 1)
    #     x_indices_2d, y_indices_2d = np.meshgrid(x_1d_indices, y_1d_indices)
    #     np.add.at(inten_new_2d, (y_indices_2d, x_indices_2d), intensity_2d)
    # else:
    #     raise RuntimeError("Cannot match the component {:s}.".format(psd_comp))
    if psd_comp == PSD_COMPONENT:
        radius = np.linalg.norm([x, y])
        azimuthal = np.arctan2(y, x)
        index = geo_ctx.dete_hori_x.shape[0] - np.searchsorted(geo_ctx.dete_hori_x[::-1], radius)
    elif psd_comp == PSDVERT_COMPONENT:
        azimuthal = x
        index = np.searchsorted(geo_ctx.dete_vert_y, y) + geo_ctx.dete_hori_x.shape[0]
    else:
        raise RuntimeError("Cannot match the component {:s}.".format(psd_comp))
    kf = geo_ctx.wavenumbers_out[index]
    polar = geo_ctx.polar_angles[index]
    kfx = kf * np.cos(polar) * np.cos(azimuthal)
    kfy = kf * np.cos(polar) * np.sin(azimuthal)
    kfz = kf * np.sin(polar)
    qx = ki - kfx
    qy = -kfy
    qz = -kfz
    # q_2d = np.linalg.norm([qx_2d, qy_2d, qz_2d], axis=0)
    omega_joule = PLANCKS_CONSTANT ** 2 / (2.0 * MASS_NEUTRON) * (ki ** 2 - kf ** 2)
    return kf, qx, qy, qz, omega_joule


def intensity_update(angle, intensities_flat, intensities_vertical):  # intensities_cyl
    # intensities_cyl += PsdInformation(angle, FILE_PSD_CYL).intensities
    intensities_flat += PsdInformation(angle, FILE_PSD_FLAT).intensities
    intensities_vertical += PsdInformation(angle, FILE_PSD_VERTICAL).intensities
    # return intensities_cyl, intensities_flat, intensities_vertical
    return intensities_flat, intensities_vertical


def plot_ax_2d(fig, ax, x_2d, y_2d, z_2d, psd_comp, subplot_index=None):
    if np.count_nonzero(z_2d) > 0:
        cnt = ax.contourf(x_2d, y_2d, z_2d, levels=np.linspace(np.max(z_2d) / 10.0, np.max(z_2d), num=10),
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

FOLDER_DATE = "190820_VNight"
ki = 1.2 * 1e10

SCAN_FOLDER_PREFIX = "Angle"
scan_angles = np.linspace(start=5, stop=170, num=166)
# scan_angles = np.linspace(start=5, stop=158, num=154)

angle = scan_angles[0]
# psd_cyl = PsdInformation(angle, FILE_PSD_CYL)
# azi_angles_cyl, y_positions_cyl, intensities_cyl, ki = psd_cyl.x_1d, psd_cyl.y_1d, psd_cyl.intensities, psd_cyl.wavenumber_incoming
psd_flat = PsdInformation(angle, FILE_PSD_FLAT)
x_pos_flat, y_pos_flat, intensities_flat = psd_flat.x_1d, psd_flat.y_1d, psd_flat.intensities
psd_vertical = PsdInformation(angle, FILE_PSD_VERTICAL)
x_pos_vertical, y_pos_vertical, intensities_vertical = psd_vertical.x_1d, psd_vertical.y_1d, psd_vertical.intensities

intensities_flat, intensities_vertical = intensity_update(angle=-angle,
                                                          intensities_flat=intensities_flat,
                                                          intensities_vertical=intensities_vertical)

for angle in scan_angles[1:]:
    intensities_flat, intensities_vertical = intensity_update(angle=angle, intensities_flat=intensities_flat,
                                                              intensities_vertical=intensities_vertical)
    intensities_flat, intensities_vertical = intensity_update(angle=-angle, intensities_flat=intensities_flat,
                                                              intensities_vertical=intensities_vertical)
# print("ki={}".format(ki * 1e-10))
# azi_cyl_2d, y_cyl_2d = np.meshgrid(azi_angles_cyl, y_positions_cyl)
# fig, ax = plt.subplots()
# plot_ax_2d(fig=fig, ax=ax, x_2d=np.rad2deg(azi_cyl_2d), y_2d=y_cyl_2d, z_2d=intensities_cyl, psd_comp=PSDCYL_COMPONENT)
# plt.title("Total intensities on cyl. PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Data_collection_cyl.pdf"]))
# plt.close(fig)

azi_vert_2d, y_vert_2d = np.meshgrid(x_pos_vertical, y_pos_vertical)
fig, ax = plt.subplots()
plot_ax_2d(fig=fig, ax=ax, x_2d=np.rad2deg(azi_vert_2d), y_2d=y_vert_2d, z_2d=intensities_vertical,
           psd_comp=PSDVERT_COMPONENT)
plt.title("Total intensities on vert. PSDs, ki={:5.3f}".format(ki * 1e-10))
plt.tight_layout()
plt.savefig("_".join([FOLDER_DATE, "Data_collection_vert.pdf"]))
plt.close(fig)

x_flat_2d, y_flat_2d = np.meshgrid(x_pos_flat, y_pos_flat)
azi_flat_2d = np.rad2deg(np.arctan2(y_flat_2d, x_flat_2d))
radi_flat_2d = np.linalg.norm([x_flat_2d, y_flat_2d], axis=0)
xplot = [x_flat_2d, azi_flat_2d]
yplot = [y_flat_2d, radi_flat_2d]
fig, axs = plt.subplots(2, 1)
for i in range(axs.ravel().shape[0]):
    plot_ax_2d(fig=fig, ax=axs[i], x_2d=xplot[i], y_2d=yplot[i], z_2d=intensities_flat, psd_comp=PSD_COMPONENT,
               subplot_index=i)
plt.suptitle("Total intensities on flat PSDs, ki={:5.3f}".format(ki * 1e-10))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("_".join([FOLDER_DATE, "Data_collection_flat.pdf"]))
plt.close(fig)

geometryctx = GeometryContext()

kf_vert, qx_vert, qy_vert, qz_vert, e_transfer_vert = np.array(list(map(
    lambda i: position2dispersion(x=x_pos_vertical[i], y=y_pos_vertical[i], psd_comp=PSDVERT_COMPONENT,
                                  geo_ctx=geometryctx), range(x_pos_vertical.shape[0]))))
kf_flat, qx_flat, qy_flat, qz_flat, e_transfer_flat = np.array(list(map(
    lambda i: position2dispersion(x=x_pos_flat[i], y=y_pos_flat[i], psd_comp=PSD_COMPONENT, geo_ctx=geometryctx),
    range(x_pos_flat.shape[0]))))

# range_kf = np.linspace(np.min(np.append(kf_vert, kf_flat)), np.max(np.append(kf_vert, kf_flat)), num=1000)
range_qx = np.linspace(np.min(np.append(qx_vert, qx_flat)), np.max(np.append(qx_vert, qx_flat)), num=1000)
range_qy = np.linspace(np.min(np.append(qy_vert, qy_flat)), np.max(np.append(qy_vert, qy_flat)), num=1000)
range_qz = np.linspace(np.min(np.append(qz_vert, qz_flat)), np.max(np.append(qz_vert, qz_flat)), num=1000)
range_de = np.linspace(np.min(np.append(e_transfer_vert, e_transfer_flat)),
                       np.max(np.append(e_transfer_vert, e_transfer_flat)), num=1000)
inten_qxde = dispersion_signal(range_x=range_qx, range_y=range_de, data_x=qx_vert, data_y=e_transfer_vert,
                               intensity=intensities_vertical) + dispersion_signal(range_x=range_qx,
                                                                                   range_y=range_de,
                                                                                   data_x=qx_flat,
                                                                                   data_y=e_transfer_flat,
                                                                                   intensity=intensities_flat)
inten_qyde = dispersion_signal(range_x=range_qy, range_y=range_de, data_x=qy_vert, data_y=e_transfer_vert,
                               intensity=intensities_vertical) + dispersion_signal(range_x=range_qy,
                                                                                   range_y=range_de,
                                                                                   data_x=qy_flat,
                                                                                   data_y=e_transfer_flat,
                                                                                   intensity=intensities_flat)
inten_qzde = dispersion_signal(range_x=range_qz, range_y=range_de, data_x=qz_vert, data_y=e_transfer_vert,
                               intensity=intensities_vertical) + dispersion_signal(range_x=range_qz,
                                                                                   range_y=range_de,
                                                                                   data_x=qz_flat,
                                                                                   data_y=e_transfer_flat,
                                                                                   intensity=intensities_flat)

# intensities_new_flat = \
#     position2dispersion(x=x_pos_flat, y=y_pos_flat, intensity_2d=intensities_flat, psd_comp=PSD_COMPONENT,
#                         geo_ctx=geometryctx)[-1]
# intensities_total = intensities_new_vert + intensities_new_flat


fig, axs = plt.subplots(1, 3, sharey="all")
plot_x = [range_qx, range_qy, range_qz]
plot_inte = [inten_qxde, inten_qyde, inten_qzde]
for i in range(axs.ravel().shape[0]):
    cnt = axs[i].contourf(plot_x[i] * 1e-10, range_de * 1e3 / CONVERSION_JOULE_PER_EV, plot_inte[i], cmap="coolwarm")
    # cnt = ax.contourf(wavevector_transfer * 1e-10, energy_transfer * 1e3, intensities_total, cmap="coolwarm")
    fig.colorbar(cnt, ax=axs[i])
# ax.set_xlabel(r"$Q = k_i - k_f$ ($\AA^{-1}$)")
# ax.set_ylabel(r"Energy transfer $\Delta E = E_{i}-E_{f}$ (meV)")
# plt.title("Dispersion on PSDs, ki={:5.3f}".format(ki * 1e-10))
# plt.grid()
# plt.tight_layout()
# plt.savefig("_".join([FOLDER_DATE, "Dispersion_total.pdf"]))
# plt.close(fig)
plt.show()


def get_wavelength_signals(scan_angles, file):
    if file in [FILE_L_VERTICAL, FILE_L_FLAT]:
        angle = scan_angles[0]
        scanname = "".join([SCAN_FOLDER_PREFIX, str(int(angle))])
        folder = "/".join([FOLDER_DATE, scanname])
        # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
        signals = np.loadtxt(fname="/".join([folder, file]), comments=COMMENT_SYMBOL)
        wavelengths = signals[:, 0] * 1e-10
        intentisities = signals[:, 1]

        scanname = "".join([SCAN_FOLDER_PREFIX, str(int(-angle))])
        folder = "/".join([FOLDER_DATE, scanname])
        # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
        signals = np.loadtxt(fname="/".join([folder, file]), comments=COMMENT_SYMBOL)
        intentisities += signals[:, 1]

        for angle in scan_angles[1:]:
            scanname = "".join([SCAN_FOLDER_PREFIX, str(int(angle))])
            folder = "/".join([FOLDER_DATE, scanname])
            # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
            signals = np.loadtxt(fname="/".join([folder, file]), comments=COMMENT_SYMBOL)
            intentisities += signals[:, 1]
            scanname = "".join([SCAN_FOLDER_PREFIX, str(int(-angle))])
            folder = "/".join([FOLDER_DATE, scanname])
            # signals = np.loadtxt(fname="/".join([folder, FILE_L_FLAT]), comments=COMMENT_SYMBOL)
            signals = np.loadtxt(fname="/".join([folder, file]), comments=COMMENT_SYMBOL)
            intentisities += signals[:, 1]
        return wavelengths, intentisities
    else:
        raise RuntimeError("Invalid file of the l-monitor")


def plot_kf_monitor(ax, kf, intensity, title):
    """
    plot the distribution of the kf as a histogram
    :param ax: the current axes in the plot figure
    :param kf: wave numbers as a 1D array in m-1, has to be trasfered to AA-1
    :param intensity: intensities of the respective wave numbers
    :param title: title of the sub-plot
    :return: nothing. the function plots the sub-figure
    """
    kf_plot = kf * 1e-10  # change the wave number unit from m-1 to AA-1
    kmax = 2.1  # largest velue of the wave number can be achieved by the instrument
    kmin = 0.9  # smallest velue of the wave number can be achieved by the instrument
    kres = 0.01  # a reasonable resolution of the wave numbers
    ax.hist(kf_plot, bins=int((kmax - kmin) / kres), range=(kmin, kmax), weights=intensity)
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel("Intensity")

# wavelength_l, intensity_lvertical = get_wavelength_signals(scan_angles=scan_angles, file=FILE_L_VERTICAL)
# intensity_lflat = get_wavelength_signals(scan_angles=scan_angles, file=FILE_L_FLAT)[-1]
# wavenumber_l = 2.0 * np.pi / wavelength_l
# intensity_l = intensity_lvertical + intensity_lflat
#
# fig, axs = plt.subplots(2, 1, sharex="all")
# titles = ["lambda monitor", "psd"]
# wavenumbers = [wavenumber_l, kf.flatten()]
# intensities = [intensity_l, intensities_total.flatten()]
# for i in range(axs.ravel().shape[0]):
#     plot_kf_monitor(axs[i], wavenumbers[i], intensities[i], titles[i])
# axs[-1].set_xlabel(r"$k_f$ ($\AA^{-1}$)")
# fig.suptitle("Wavenumber distribution at ki={:5.3f}".format(ki * 1e-10))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("_".join([FOLDER_DATE, "Wavenumber_distribution.pdf"]))
# # plt.show()
# plt.close(fig)
# print("{} is finished.".format(FOLDER_DATE))
