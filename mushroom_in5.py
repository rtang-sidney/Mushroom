import re

import matplotlib.pyplot as plt
import numpy as np

import geometry_calculation as geo
import neutron_context as nctx
from global_context import comment_symbol
from mushroom_context import MushroomContext

plt.rcParams.update({'font.size': 18})

FOLDER_MUSH = "McStas/Mushroom_SourceHZB_array_20220114_133040"
FOLDER_NEAT = "McStas/HZB_NEAT_20220118_185751"
FOLDER_MUSH_TOF_HZB = "McStas/Mushroom_TOFHZB_20220119_104234"

INSTR_MUSH_MONO = "Mushroom Mono"
INSTR_MUSH_TOF = "Mushroom TOF"
INSTR_NEAT = "NEAT"

MONITOR_PREFIX_PSD = "psd"
LD_PREFIX = "l"
DATA_EXTENSION = "dat"
FORM_FLAT = "flat"
FORM_NEAT = "y"
FORM_TOF = "tof"
PSD_XVAR = "X"
PSD_YVAR = "Y"
COMP_FLAT = "_".join([MONITOR_PREFIX_PSD, FORM_FLAT])

MONITOR_PSD = "PSD"
MONITOR_TOF = "TOF"
PSD_TOF_MONITOR = "psd_tof_monitor"  # shows PSD signal of each TOF bin
MONITOR_TYPES = [MONITOR_PSD, MONITOR_TOF]

MONITOR_PREFIX_PSD = "psd"
MONITOR_PREFIX_TOF = "tof"

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
KEY_YPOS = "pos_y"
KEY_ZPOS = "pos_z"
KEY_ERRORS = "Errors"
KEY_VALUES = "values"
REVEVANT_KEYS = [KEY_INSTR_PARAM, KEY_SIZE, KEY_XVAR, KEY_YVAR, KEY_XLABEL, KEY_YLABEL, KEY_XYLIMITS, KEY_POSITION,
                 KEY_COMPONENT]

UNIT_CENTIMETRE = "cm"
UNIT_METRE = "m"
UNIT_DEG = "deg"
UNIT_S = "s"

KI_NEAT = 1.1e10  # the incoming ki at NEAT is already selected by the chopper
NEAT_PRIMARY = 11.97 + 2.143 + 0.165  # flight path of NEAT primary spectrometer

TBIN_MIN = 23800  # mu-s, the lower bound of the time bins
TBIN_MAX = 24500  # mu-s, the upper bound of the time bins
TBIN_NUMBER = 70  # number of bins used
TBINS = np.linspace(TBIN_MIN, TBIN_MAX, TBIN_NUMBER, endpoint=False)
DISTANCE_CS = 14.278  # chopper-sample distance

WAVENUMBER_INCOMING = "ki"
PATTERN_SIZE = re.compile(r"([0-9]*),\s?([0-9]*)")
PATTERN_XYLIMITS = re.compile(
    r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_POSITION = re.compile(r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_PARAM = re.compile(r"\s*(\S*_?\S*)=([-+]?[0-9]*\.?[0-9]*)")
PATTERN_LABEL = re.compile(r"\s*(\S*\s*\S*)\s*\[(\S*)]")
PATTERN_VALUES = re.compile(r"#\s*values:\s*([0-9]*.[0-9]*)\s*([0-9]*.[0-9]*)\s*[0-9]*.[0-9]*")


class MonitorInformation:
    def __init__(self, filepath):
        f = open(file=filepath).readlines()
        keys = []
        contents = []
        error_row = 0
        for line in f:
            if line.startswith(comment_symbol):
                if KEY_ERRORS in line:
                    error_row = f.index(line)
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
        self.x_1d, self.y_1d = self._xy_axes()
        self.inten_2d = np.loadtxt(fname=filepath, comments=comment_symbol, max_rows=self.ysize)
        if error_row > 0:
            self.inten_errors = np.loadtxt(fname=filepath, skiprows=error_row + 1, max_rows=self.ysize)
        else:
            raise RuntimeError("Failed to load intensity errors.")

    def _xy_axes(self):
        xmin, xmax, ymin, ymax = list(
            map(float, re.search(pattern=PATTERN_XYLIMITS, string=self.metadata_dict[KEY_XYLIMITS]).groups()))
        xaxis = np.linspace(start=xmin, stop=xmax, num=self.xsize)
        yaxis = np.linspace(start=ymin, stop=ymax, num=self.ysize)
        if self.metadata_dict[KEY_XUNIT] == UNIT_CENTIMETRE:
            xaxis *= 1e-2
        elif self.metadata_dict[KEY_XUNIT] == UNIT_DEG:
            xaxis = np.deg2rad(xaxis)
        elif self.metadata_dict[KEY_XUNIT] in [UNIT_METRE, UNIT_S]:
            pass
        else:
            raise RuntimeError("Does not recognise the unit {}".format(self.metadata_dict[KEY_XUNIT]))
        if self.metadata_dict[KEY_YUNIT] == UNIT_CENTIMETRE:
            yaxis *= 1e-2
        elif self.metadata_dict[KEY_YUNIT] == UNIT_DEG:
            yaxis = np.deg2rad(yaxis)
        elif self.metadata_dict[KEY_YUNIT] in [UNIT_METRE, UNIT_S]:
            pass
        else:
            raise RuntimeError("Does not recognise the unit {}".format(self.metadata_dict[KEY_YUNIT]))
        return xaxis, yaxis


def neat_detector(scan_folder):
    """
    gives the information of the neutron beam in the given scan
    :param scan_folder: the directory of the whole scan
    :return: wavelengths (lambda_1d) of the neuron beam, intensities (inten_1d), and errors (error_1d)
    """
    tof_file = detector_filepath(detector_type=MONITOR_PREFIX_TOF, detector_form=FORM_NEAT, folder=scan_folder)
    tof_neat = MonitorInformation(filepath=tof_file)
    y_1d, tof_1d, inten_2d, error_2d = tof_neat.x_1d, tof_neat.y_1d, tof_neat.inten_2d, tof_neat.inten_errors
    y_2d, tof_2d = np.meshgrid(y_1d, tof_1d)
    # inten_1d = np.sum(inten_2d, axis=1)
    # error_1d = np.sqrt(np.sum(error_2d ** 2, axis=1))
    radius_detector = 2.5  # m, distance from the tof-detector to the chopper
    distance = np.linalg.norm([y_2d, radius_detector])
    time_cs = NEAT_PRIMARY / nctx.wavenumber2velocity(KI_NEAT)
    time_sd = tof_2d - time_cs
    v_2d = distance / time_sd
    # print(velocity)
    lambda_2d = nctx.velocity2wavelength(v_2d)
    # print(wavelength * 1e10)
    return lambda_2d.flatten(), inten_2d.flatten(), error_2d.flatten()


def psd_tof_monitor(mush_ctx: MushroomContext, scan_folder):
    def radius2wavelength(mush_ctx: MushroomContext, tof, radius_now, inten_now, err_now):
        psd_bank_index = np.argmin(abs(radius_now - mush_ctx.dete_points[0, :]))
        d_sd = distance_sd(mush_ctx, psd_bank_index)  # sample-detector distance
        d_cd = DISTANCE_CS + d_sd  # chopper-detector distance
        velocity_cd = d_cd / tof  # only in elastic case
        wavelength_cd = nctx.velocity2wavelength(velocity_cd)
        return wavelength_cd  # plot_intensity, plot_error

    def distance_sd(mush_ctx: MushroomContext, index):
        distance_sa = geo.points_distance(point1=mush_ctx.sa_point, point2=mush_ctx.an_points[:, index])
        distance_ad = geo.points_distance(point1=mush_ctx.an_points[:, index], point2=mush_ctx.dete_points[:, index])
        distance_sd = distance_sa + distance_ad
        return distance_sd

    # plot_wavelength = wavelengths_neat  # np.linspace(5.2, 6.0, 80)
    # plot_intensity = np.zeros_like(plot_wavelength)
    # plot_error = np.zeros_like(plot_wavelength)
    plot_wavelength = np.empty(0)
    plot_intensity = np.empty(0)
    plot_error = np.empty(0)

    for slice, tbin in enumerate(TBINS):
        tof = tbin * 1e-6
        filepath = detector_filepath(detector_type=MONITOR_PREFIX_PSD, detector_form="_".join([FORM_TOF, str(slice)]),
                                     folder=scan_folder)
        # print(filepath)
        psd_tof = MonitorInformation(filepath=filepath)
        x_1d, y_1d, inten_2d, error_2d = psd_tof.x_1d, psd_tof.y_1d, psd_tof.inten_2d, psd_tof.inten_errors
        x_2d, y_2d = np.meshgrid(x_1d, y_1d)
        r_2d = np.linalg.norm([x_2d, y_2d], axis=0)
        # print(np.max(inten_2d))
        inten_flatten = inten_2d.flatten()
        finite_inten = inten_flatten != 0  # an index array where the intensity is nonzero
        r_flatten, inten_flatten, err_flatten = r_2d.flatten()[finite_inten], inten_flatten[finite_inten], \
                                                error_2d.flatten()[finite_inten]
        # print(r_flatten)
        for counter, radius in enumerate(r_flatten):
            wavelength = radius2wavelength(mush_ctx, tof, radius, inten_flatten[counter],
                                           err_flatten[counter])
            plot_wavelength = np.append(plot_wavelength, wavelength)
            plot_intensity = np.append(plot_intensity, inten_flatten[counter])
            plot_error = np.append(plot_error, err_flatten[counter])
        # print(plot_intensity)
    return plot_wavelength, plot_intensity, plot_error


def detector_filepath(detector_type, detector_form, folder):
    filename = ".".join(["_".join([detector_type, detector_form]), DATA_EXTENSION])
    filepath = "/".join([folder, filename])
    return filepath


def neat_inten_adjsut(scan_folder):
    inten_psd, err_psd = neat_intensity(scan_folder, form=FORM_NEAT)
    lambda_tof, inten_tof, error_tof = neat_detector(scan_folder)
    total_inten_tof = np.sum(inten_tof)
    total_err_tof = np.sum(error_tof)
    inten_tof *= inten_psd / total_inten_tof
    error_tof *= err_psd / total_err_tof
    return lambda_tof, inten_tof, error_tof


def neat_intensity(scan_folder, form=FORM_NEAT):
    psd_name = detector_filepath(detector_type=MONITOR_PREFIX_TOF, detector_form=form, folder=scan_folder)
    lines = open(file=psd_name).readlines()
    inten, error = None, None
    for line in lines:
        if KEY_VALUES in line:
            # print(line)
            inten, error = re.match(PATTERN_VALUES, line).groups()
            return float(inten), float(error)
    if inten is None:
        raise RuntimeError("Failed to find the intensity")


def l_intensity(scan_folder, form=FORM_FLAT):
    l_file = detector_filepath(detector_type=LD_PREFIX, detector_form=form, folder=scan_folder)
    l_data = np.loadtxt(l_file).transpose()
    wavelengths = l_data[0, :] * 1e-10
    intensities = l_data[1, :]
    errors = l_data[2, :]
    return wavelengths, intensities, errors


# def f_gaussian(x, a, b, c):
#     return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

def expectation_weighted(x, intensity):
    return np.sum(x * intensity) / np.sum(intensity)


def std_weighted(x, intensity):
    mean = expectation_weighted(x, intensity)
    variance = np.sum((x - mean) ** 2 * intensity) / np.sum(intensity)
    return np.sqrt(variance)


def fwhm_gaussian(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def plot_instrument(instr_name, wavelengths, intens, errs):
    def integrate_inten(instr_name, wavelengths_eff, intens_eff):
        if instr_name == INSTR_MUSH_TOF:
            l_sa, inten_sa, err_sa = l_intensity(FOLDER_MUSH_TOF_HZB, form="sa")
            l_nonzero_inten = wavelengths_eff[np.nonzero(intens_eff)]
            measured_indices = np.where(np.logical_and(l_sa > np.min(l_nonzero_inten), l_sa < np.max(l_nonzero_inten)))[
                0]
            integrated_inten = np.sum(intens_eff) * np.sum(inten_sa) / np.sum(inten_sa[measured_indices])
            return integrated_inten
        else:
            raise ValueError(
                "This instrument simulation is irrelevant for integrating intensity: {:s}".format(instr_name))

    eff_indices = np.where(np.logical_and(wavelengths > PLOT_LAMBDA_MIN, wavelengths < PLOT_LAMBDA_MAX))[0]
    wavelengths_eff = wavelengths[eff_indices]
    intens_eff = intens[eff_indices]
    errs_eff = errs[eff_indices]
    std = std_weighted(wavelengths_eff, intens_eff)
    fwhm = fwhm_gaussian(std)
    stepsize = std
    plot_lambda_bins = int(round((PLOT_LAMBDA_MAX - PLOT_LAMBDA_MIN) / stepsize)) + 1
    if plot_lambda_bins % 2 == 0:
        plot_lambda_bins += 1
    else:
        pass
    plot_lambda = np.linspace(PLOT_LAMBDA_MIN, PLOT_LAMBDA_MAX, plot_lambda_bins)
    plot_inten = np.histogram(wavelengths_eff, bins=plot_lambda, weights=intens_eff)[0]
    plot_err = np.histogram(wavelengths_eff, bins=plot_lambda, weights=errs_eff)[0]
    ax.errorbar(plot_lambda[:-1] * 1e10, plot_inten / np.max(plot_inten), yerr=plot_err / np.max(plot_inten), fmt='o-',
                label="{:s}\n".format(instr_name) + r"$I=$" + "{:.1e}\n".format(
                    np.sum(intens_eff)) + r"$\mathrm{FWHM}=$" + "{:.3f}".format(fwhm * 1e10) + r"$\AA$")
    if instr_name == INSTR_MUSH_TOF:
        return integrate_inten(instr_name, wavelengths_eff, intens_eff)
    else:
        pass


mush_ctx = MushroomContext()
L_MIDDLE = mush_ctx.wavelength_f
L_MARGIN = 0.2e-10
PLOT_LAMBDA_MIN = L_MIDDLE - L_MARGIN
PLOT_LAMBDA_MAX = L_MIDDLE + L_MARGIN

fig, ax = plt.subplots()
path_mush = FOLDER_MUSH
wavelengths_mush, intensities_mush, errors_mush = l_intensity(path_mush)
intensities_mush *= 161 * 2
errors_mush *= np.sqrt(161 * 2)
plot_instrument(instr_name=INSTR_MUSH_MONO, wavelengths=wavelengths_mush, intens=intensities_mush, errs=errors_mush)

path_mush_tofhzb = FOLDER_MUSH_TOF_HZB
wavelengths_tofhzb, intensities_tofhzb, errors_tofhzb = psd_tof_monitor(mush_ctx=mush_ctx, scan_folder=path_mush_tofhzb)
intensities_tofhzb *= 161 * 2
errors_tofhzb *= np.sqrt(161 * 2)
integrated_inten = plot_instrument(instr_name=INSTR_MUSH_TOF, wavelengths=wavelengths_tofhzb, intens=intensities_tofhzb,
                                   errs=errors_tofhzb)
print(integrated_inten)

path_neat = FOLDER_NEAT
wavelengths_neat, intensities_neat, errors_neat = neat_inten_adjsut(scan_folder=path_neat)
plot_instrument(instr_name=INSTR_NEAT, wavelengths=wavelengths_neat, intens=intensities_neat, errs=errors_neat)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_title(r"$k_{i}=$" + "{:.3f}".format(
    mush_ctx.wavenumber_f * 1e-10) + r" $\AA^{-1}$,   " + r"$\lambda_{i}=$" + "{:.3f}".format(
    mush_ctx.wavelength_f * 1e10) + r" $\AA$")
ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel("Intensity (peak normalised to 1)")
ax.tick_params(axis="both", direction="in")
ax.set_xlim(PLOT_LAMBDA_MIN * 1e10, PLOT_LAMBDA_MAX * 1e10)
fig.savefig("Mushroom_CountRate.png", bbox_inches='tight')
plt.close(fig)
