import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import neutron_context as neutron
from mushroom_context import MushroomContext

np.set_printoptions(threshold=sys.maxsize, precision=2)

FLAT = "flat"
VERTICAL = "vertical"
CYLINDER = "cylinder"

PSD_PREFIX = "psd"
LAMBDA_DETECTOR_PREFIX = "l"
DATAFILE_EXTENSION = "dat"

FOLDER_DATE = "190820_VNight"
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
PATTERN_LABEL = re.compile(r"\s*(\S*\s*\S*)\s*\[(\S*)]")

SCAN_FOLDER_PREFIX = "Angle"


class PsdInformation:
    def __init__(self, angle, psd_form):  # psd_type, psd_name,
        mushroom = MushroomContext()
        self.scan_angle = angle
        folder = angle_to_folder(angle=self.scan_angle)
        filepath = detector_filepath(detector_type=PSD_PREFIX, detector_form=psd_form, folder=folder)
        f = open(file=filepath).readlines()
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
        self.xsize, self.ysize = list(
            map(int, re.search(pattern=PATTERN_SIZE, string=self.metadata_dict[KEY_SIZE]).groups()))
        x_1d, y_1d = self._xy_axes()
        intensities = np.loadtxt(fname=filepath, comments=COMMENT_SYMBOL, max_rows=self.ysize)
        self.x_1d, self.y_1d, self.intensities = self._psd_signal_adjust(x_1d, y_1d, intensities, geo_ctx=mushroom)

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

    def _psd_signal_adjust(self, x, y, intensity, geo_ctx: MushroomContext):
        component = self.metadata_dict[KEY_COMPONENT]
        if component == PSDCYL_COMPONENT:
            x = np.deg2rad(x)
            y = self._get_psdcyl_middle() + y
            x, y = np.meshgrid(x, y)
        elif component == PSD_COMPONENT:
            x, y = np.meshgrid(x, y)
            intensity = np.where(np.linalg.norm([x, y], axis=0) > 0.4, intensity, 0)
        elif component == PSDVERT_COMPONENT:
            x = np.arctan2(x, abs(geo_ctx.detector_line_vert[-1])) + np.deg2rad(self.scan_angle)
            y = self._get_psdcyl_middle() + y
            x, y = np.meshgrid(x, y)
        else:
            raise RuntimeError("Cannot match the x variable {}.".format(self.metadata_dict[KEY_XVAR]))
        x, y = x.flatten(), y.flatten()
        intensity = intensity.flatten()
        return x, y, intensity


def angle_to_folder(angle):
    scanname = "".join([SCAN_FOLDER_PREFIX, str(int(round(angle)))])
    folder = "/".join([FOLDER_DATE, scanname])
    return folder


def detector_filepath(detector_type, detector_form, folder):
    filename = ".".join(["_".join([detector_type, detector_form]), DATAFILE_EXTENSION])
    filepath = "/".join([folder, filename])
    return filepath


def folder_name(instrument, day, month, year, hms):
    if isinstance(instrument, str) is False:
        raise RuntimeError("Invalid type of instrument name.")
    if isinstance(day, int) is False:
        raise RuntimeError("Invalid type of day.")
    if isinstance(month, int) is False:
        raise RuntimeError("Invalid type of month.")
    if isinstance(year, int) is False:
        raise RuntimeError("Invalid type of year.")
    if isinstance(hms, int) is False:
        raise RuntimeError("Invalid type of time of day.")

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


def position2dispersion(x, y, psd_type, geo_ctx: MushroomContext):
    ki = geo_ctx.wavenumber_in
    if psd_type == FLAT:
        radius = np.linalg.norm([x, y])
        azimuthal = np.arctan2(y, x)
        index = geo_ctx.dete_hori_x.shape[0] - np.searchsorted(geo_ctx.dete_hori_x[::-1], radius)
    elif psd_type == VERTICAL:
        azimuthal = x
        index = np.searchsorted(geo_ctx.dete_vert_y, y) + geo_ctx.dete_hori_x.shape[0]
    else:
        raise RuntimeError("Cannot match the component {:s}.".format(psd_type))
    kf = geo_ctx.wavenumbers_out[index]
    polar = geo_ctx.pol_angles[index]
    kfx = kf * np.cos(polar) * np.cos(azimuthal)
    kfy = kf * np.cos(polar) * np.sin(azimuthal)
    kfz = kf * np.sin(polar)
    qx = ki - kfx
    qy = -kfy
    qz = -kfz
    hw = neutron.planck_constant ** 2 / (2.0 * neutron.mass_neutron) * (ki ** 2 - kf ** 2)
    return kf, qx, qy, qz, hw


def intensity_merge(angle, intensities_flat, intensities_vertical):  # intensities_cyl
    # intensities_cyl += PsdInformation(angle, FILE_PSD_CYL).intensities
    intensities_flat += PsdInformation(angle, FLAT).intensities
    intensities_vertical += PsdInformation(angle, VERTICAL).intensities
    # return intensities_cyl, intensities_flat, intensities_vertical
    return intensities_flat, intensities_vertical


def scatter_plot_colour(fig, ax, data_x, data_y, data_z, psd_type):
    if np.count_nonzero(data_z) > 0:
        cnt = ax.scatter(data_x, data_y, c=data_z)
        fig.colorbar(cnt, ax=ax)
    if psd_type == PSDCYL_COMPONENT or psd_type == PSDVERT_COMPONENT:
        xlabel = r"Azimuthal angle $\theta$ (deg)"
        ylabel = "Vertical position (m)"
    elif psd_type == PSD_COMPONENT:
        xlabel = "x position (m)"
        ylabel = "y position (m)"
        ax.set_title("Cartesian")
    else:
        raise RuntimeError("Invalid type of PSD given.")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()


def get_wavelength_signals(angles, detector_form):
    angle = angles[0]
    filepath = detector_filepath(detector_type=LAMBDA_DETECTOR_PREFIX, detector_form=detector_form,
                                 folder=angle_to_folder(angle))
    signals = np.loadtxt(fname=filepath, comments=COMMENT_SYMBOL)
    wavelengths = signals[:, 0] * 1e-10
    intentisities = signals[:, 1]

    filepath = detector_filepath(detector_type=LAMBDA_DETECTOR_PREFIX, detector_form=detector_form,
                                 folder=angle_to_folder(-angle))
    signals = np.loadtxt(fname=filepath, comments=COMMENT_SYMBOL)
    intentisities += signals[:, 1]

    for angle in angles[1:]:
        filepath = detector_filepath(detector_type=LAMBDA_DETECTOR_PREFIX, detector_form=detector_form,
                                     folder=angle_to_folder(angle))
        signals = np.loadtxt(fname=filepath, comments=COMMENT_SYMBOL)
        intentisities += signals[:, 1]

        filepath = detector_filepath(detector_type=LAMBDA_DETECTOR_PREFIX, detector_form=detector_form,
                                     folder=angle_to_folder(-angle))
        signals = np.loadtxt(fname=filepath, comments=COMMENT_SYMBOL)
        intentisities += signals[:, 1]
    return wavelengths, intentisities


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


def psd_data(mushroom: MushroomContext, angles, psd_form):
    angle = angles[0]
    psd = PsdInformation(angle=angle, psd_form=psd_form)
    psd_x, psd_y, psd_intensity = psd.x_1d, psd.y_1d, psd.intensities
    for angle in angles[1:]:
        psd_intensity += PsdInformation(angle, psd_form).intensities

    fig, ax = plt.subplots()
    scatter_plot_colour(fig=fig, ax=ax, data_x=psd_x, data_y=psd_y, data_z=psd_intensity, psd_type=PSD_COMPONENT)
    plt.suptitle(r"Total intensity at {:s} PSD, ki={:5.3f} $\AA$".format(psd_form, mushroom.wavenumber_in * 1e-10))
    plot_file = "_".join([FOLDER_DATE, "PSDCollection_{:s}.pdf".format(psd_form)])
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: {:s}".format(plot_file))
    return psd_x, psd_y, psd_intensity


def psd2dispersion(mushroom: MushroomContext, psd_form, psd_x, psd_y, psd_intensity):
    kf, qx, qy, qz, e_transfer = np.array(list(map(
        lambda i: position2dispersion(x=psd_x[i], y=psd_y[i], psd_type=psd_form,
                                      geo_ctx=psd_intensity), range(psd_x.shape[0]))))

    fig, ax = plt.subplots()
    scatter_plot_colour(fig=fig, ax=ax, data_x=qx, data_y=qy, data_z=e_transfer, psd_type=PSD_COMPONENT)
    plt.suptitle(r"Dispersion from {:s} PSD, ki={:5.3f} $\AA$".format(psd_form, mushroom.wavenumber_in * 1e-10))
    plot_file = "_".join([FOLDER_DATE, "Dispersion_{:s}.pdf".format(psd_form)])
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: {:s}".format(plot_file))


def wavenumber_plot(mushroom: MushroomContext, angles, wavenumber_psd, intensity_psd):
    wavelength_l, intensity_lvertical = get_wavelength_signals(angles=angles, detector_form=VERTICAL)
    intensity_lflat = get_wavelength_signals(angles=angles, detector_form=VERTICAL)[-1]
    wavenumber_l = 2.0 * np.pi / wavelength_l
    intensity_l = intensity_lvertical + intensity_lflat

    fig, axs = plt.subplots(2, 1, sharex="all")
    titles = ["lambda monitor", "psd"]
    wavenumbers = [wavenumber_l, wavenumber_psd.flatten()]
    intensity_psd = [intensity_l, intensity_psd.flatten()]
    for i in range(axs.ravel().shape[0]):
        plot_kf_monitor(axs[i], wavenumbers[i], intensity_psd[i], titles[i])
    axs[-1].set_xlabel(r"$k_f$ ($\AA^{-1}$)")
    fig.suptitle("Wavenumber distribution at ki={:5.3f}".format(mushroom.wavenumber_in * 1e-10))
    plot_file = "_".join([FOLDER_DATE, "Wavenumber_distribution.pdf"])
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: {:s}".format(plot_file))


mushroomctx = MushroomContext()

scan_angles = np.linspace(start=5, stop=170, num=166)
