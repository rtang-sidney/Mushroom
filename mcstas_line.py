import matplotlib.pyplot as plt
import numpy as np
import re
import os
import neutron_context as nctx
from global_context import comment_symbol
from mushroom_context import MushroomContext
import global_context as glb
import geometry_calculation as geo

plt.rcParams.update({'font.size': 18})

# Calculates the dispersion relation along one hkl-line. It consists a Python calculation based on the magnon theory
# as well as a McStas simulation

# FOLDER = "McStas/Line_ki1.6"
FOLDER = "McStas/LineNew_ki1.6"

PSD_PREFIX = "psd"
LD_PREFIX = "l"
DATA_EXTENSION = "dat"
FORM_FLAT = "flat"
PSD_XVAR = "X"
PSD_YVAR = "Y"
COMP_FLAT = "_".join([PSD_PREFIX, FORM_FLAT])

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
REVEVANT_KEYS = [KEY_INSTR_PARAM, KEY_SIZE, KEY_XVAR, KEY_YVAR, KEY_XLABEL, KEY_YLABEL, KEY_XYLIMITS, KEY_POSITION,
                 KEY_COMPONENT]

UNIT_CENTIMETRE = "cm"
UNIT_METRE = "m"
UNIT_DEG = "deg"

WAVENUMBER_INCOMING = "ki"
PATTERN_SIZE = re.compile(r"([0-9]*),\s?([0-9]*)")
PATTERN_XYLIMITS = re.compile(
    r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_POSITION = re.compile(r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_PARAM = re.compile(r"\s*(\S*_?\S*)=([-+]?[0-9]*\.?[0-9]*)")
PATTERN_LABEL = re.compile(r"\s*(\S*\s*\S*)\s*\[(\S*)]")
PATTERN_SCANFOLDER = re.compile(r"sa([-+]?[0-9]*)\_an([-+]?[0-9]*)")


class PsdInformation:
    def __init__(self, scan_folder):
        """
        initialise the object with the PSD information
        :param ki: incoming wavenumber
        :param sa_ang: current sample rotation
        :param an_deg: the scan angle that defines the folder storing the simulated data
        :param psd_form: denotes which PSD is to be handled
        """
        mush_ctx = MushroomContext()
        filepath = detector_filepath(detector_type=PSD_PREFIX, detector_form=FORM_FLAT, folder=scan_folder)
        f = open(file=filepath).readlines()
        keys = []
        contents = []
        error_line = 0
        for line in f:
            if line.startswith(comment_symbol):
                if KEY_ERRORS in line:
                    error_line = f.index(line)
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
        # print(x_1d)
        intensities = np.loadtxt(fname=filepath, comments=comment_symbol, max_rows=self.ysize)
        if error_line > 0:
            inten_errors = np.loadtxt(fname=filepath, skiprows=error_line + 1, max_rows=self.ysize)
        else:
            raise RuntimeError("Failed to load intensity errors.")
        self.x_1d, self.y_1d, self.intensities, self.inten_errors = self._psd_signal_adjust(x_1d, y_1d, intensities,
                                                                                            inten_errors,
                                                                                            geo_ctx=mush_ctx)

    def _xy_axes(self):
        xmin, xmax, ymin, ymax = list(
            map(float, re.search(pattern=PATTERN_XYLIMITS, string=self.metadata_dict[KEY_XYLIMITS]).groups()))
        xaxis = np.linspace(start=xmin, stop=xmax, num=self.xsize)
        yaxis = np.linspace(start=ymin, stop=ymax, num=self.ysize)
        if self.metadata_dict[KEY_XUNIT] == UNIT_CENTIMETRE:
            xaxis *= 1e-2
        elif self.metadata_dict[KEY_XUNIT] == UNIT_METRE or self.metadata_dict[KEY_XUNIT] == UNIT_DEG:
            pass
        else:
            raise RuntimeError("Does not recognise the unit {}".format(self.metadata_dict[KEY_XUNIT]))
        if self.metadata_dict[KEY_YUNIT] == UNIT_CENTIMETRE:
            yaxis *= 1e-2
        elif self.metadata_dict[KEY_YUNIT] == UNIT_METRE or self.metadata_dict[KEY_YUNIT] == UNIT_DEG:
            pass
        else:
            raise RuntimeError("Does not recognise the unit {}".format(self.metadata_dict[KEY_YUNIT]))
        return xaxis, yaxis

    def _psd_signal_adjust(self, x, y, intensity, inten_errors, geo_ctx: MushroomContext):
        component = self.metadata_dict[KEY_COMPONENT]
        if component == COMP_FLAT:
            x, y = np.meshgrid(x, y)
        else:
            raise RuntimeError("Cannot match the x variable {}.".format(self.metadata_dict[KEY_XVAR]))
        x, y = x.flatten(), y.flatten()
        intensity = intensity.flatten()
        inten_errors = inten_errors.flatten()
        ind_nonzero = np.nonzero(intensity)
        return x[ind_nonzero], y[ind_nonzero], intensity[ind_nonzero], inten_errors[ind_nonzero]


def detector_filepath(detector_type, detector_form, folder):
    filename = ".".join(["_".join([detector_type, detector_form]), DATA_EXTENSION])
    filepath = "/".join([folder, filename])
    return filepath


def intensity_adjsut(scan_folder, psd_inten, inten_error, kf):
    l_file = detector_filepath(detector_type=LD_PREFIX, detector_form=FORM_FLAT, folder=scan_folder)
    l_data = np.loadtxt(l_file).transpose()
    wavelengths = l_data[0, :]
    intensities = l_data[1, :]
    lf = nctx.wavenumber2wavelength(kf)
    inten_before = np.sum(intensities)
    l_inten_edit = np.where(abs(wavelengths - lf * 1e10) < 1, intensities, 0)
    l_inten_after = np.sum(l_inten_edit)
    psd_inten_after = psd_inten * l_inten_after / inten_before
    # print(inten_error.shape, psd_inten_after.shape, intensities.shape)
    error_after = inten_error * psd_inten_after / psd_inten
    # print(l_inten_after, inten_before, l_inten_after / inten_before)
    return psd_inten_after, error_after


def psd2scattering(psd_x, psd_y, ki, kf, sa_ang):
    def one_point(x, y, ki, kf, sa_ang):
        azi = np.arctan2(y, x)
        r = np.linalg.norm([x, y])
        index = np.argmin(abs(r - mush_ctx.dete_points[0, :]))
        pol_ang = mush_ctx.pol_angles[index]
        qx = ki - kf * np.cos(pol_ang) * np.cos(azi)
        qy = -kf * np.cos(pol_ang) * np.sin(azi)
        qz = -kf * np.sin(pol_ang)
        qx, qy = geo.rotation_around_z(rot_angle=-sa_ang, old_x=qx, old_y=qy)
        return qx, qy, qz

    if psd_x.shape[0] == 1:
        return one_point(psd_x, psd_y, ki, kf, sa_ang)
    else:
        qx_c = np.zeros(0)
        qy_c = np.zeros(0)
        qz_c = np.zeros(0)
        for i in range(psd_x.shape[0]):
            qx, qy, qz = one_point(psd_x[i], psd_y[i], ki, kf, sa_ang)
            qx_c = np.append(qx_c, qx)
            qy_c = np.append(qy_c, qy)
            qz_c = np.append(qz_c, qz)
        return qx_c, qy_c, qz_c


def an2scattering(an_x, an_y, ki, kf, sa_rot, an_azi):
    pol_ang = np.arctan2(an_y, an_x)
    qx = ki - kf * np.cos(pol_ang) * np.cos(an_azi)
    qy = -kf * np.cos(pol_ang) * np.sin(an_azi)
    qz = -kf * np.sin(pol_ang)
    qx, qy = geo.rotation_around_z(rot_angle=-sa_rot, old_x=qx, old_y=qy)
    return qx, qy, qz


mush_ctx = MushroomContext()
ki = 1.6e10
kf = 1.1e10
aa = 4.5e-10
k, l = -1.0, -0.12
hw = nctx.wavenumber2energy(ki) - nctx.wavenumber2energy(kf)

simu_file = "McStas/ki1.6_k-1.00_l-0.12.dat"
simu_data = np.loadtxt(simu_file, delimiter=",").transpose()
simu_inten, simu_h = simu_data[6, :], simu_data[8, :]
simu_inten = simu_inten / np.max(simu_inten)
simu_inten = np.where(simu_inten > 1e-6, simu_inten, 1e-6)
sort_indices = np.argsort(simu_h)
simu_h = simu_h[sort_indices]
simu_inten = simu_inten[sort_indices]

qx_collect = np.zeros(0)
qy_collect = np.zeros(0)
inten_collect = np.zeros(0)
error_collect = np.zeros(0)

for subfolder in os.listdir(FOLDER):
    path = "/".join([FOLDER, subfolder])
    psd_info = PsdInformation(path)
    if psd_info.intensities.shape[0] == 0:
        sa_deg, an_deg = re.match(PATTERN_SCANFOLDER, subfolder).groups()
        sa_deg, an_deg = float(sa_deg), float(an_deg)
        sa_rad = np.deg2rad(sa_deg)
        an_rad = np.deg2rad(an_deg)
        an_x, an_y = psd_info.metadata_dict[KEY_ZPOS], psd_info.metadata_dict[KEY_YPOS]
        an_x, an_y = float(an_x), float(an_y)
        qx, qy, qz = an2scattering(an_x, an_y, ki, kf, sa_rad, an_rad)
        inten = 0
        inten_error = 0
    else:
        sa_deg, an_deg = re.match(PATTERN_SCANFOLDER, subfolder).groups()
        sa_deg = float(sa_deg)
        sa_rad = np.deg2rad(sa_deg)
        qx, qy, qz = psd2scattering(psd_info.x_1d, psd_info.y_1d, ki, kf, sa_rad)
        inten, inten_error = intensity_adjsut(scan_folder=path, psd_inten=psd_info.intensities,
                                              inten_error=psd_info.inten_errors, kf=kf)
    qx = nctx.q2rlu(q_value=qx, l_const=aa)
    qy = nctx.q2rlu(q_value=qy, l_const=aa)
    qz = nctx.q2rlu(q_value=qz, l_const=aa)
    if isinstance(inten, int):
        print(subfolder, qx, qy, qz, inten)
        qx_collect = np.append(qx_collect, qx)
        qy_collect = np.append(qy_collect, qy)
        inten_collect = np.append(inten_collect, inten)
        error_collect = np.append(error_collect, inten_error)
    elif isinstance(inten, np.ndarray) and inten.shape[0] == 1:
        qx = qx[0]
        qy = qy[0]
        inten = inten[0]
        inten_error = inten_error[0]
        print(subfolder, qx, qy, qz, inten)
        qx_collect = np.append(qx_collect, qx)
        qy_collect = np.append(qy_collect, qy)
        inten_collect = np.append(inten_collect, inten)
        error_collect = np.append(error_collect, inten_error)
    else:
        # print(subfolder, qx.shape, qy.shape, qz.shape, inten.shape)
        # print(subfolder, qx.shape[0])
        # # if q_vector.shape[1] != inten.shape[0]:
        # #     raise RuntimeError("THe size of the position array is not consistent with the intensity")
        # for i in range(inten.shape[0]):
        #     print(subfolder, qx[i], qy[i], qz[i], inten[i])
        #     qx_collect = np.append(qx_collect, qx[i])
        #     qy_collect = np.append(qy_collect, qy[i])
        #     inten_collect = np.append(inten_collect, inten[i])
        #     error_collect = np.append(error_collect, inten_error[i])
        print(subfolder, np.mean(qx), np.mean(qy), np.mean(qz), np.sum(inten))
        qx_collect = np.append(qx_collect, np.mean(qx))
        qy_collect = np.append(qy_collect, np.mean(qy))
        inten_collect = np.append(inten_collect, np.sum(inten))
        error_collect = np.append(error_collect, np.sum(inten_error))

inten_collect = inten_collect / np.max(inten_collect)
inten_collect = np.where(inten_collect > 1e-6, inten_collect, 1e-6)
ind_collect = np.argsort(qx_collect)
fig, ax = plt.subplots()
ax.errorbar(qx_collect[ind_collect], inten_collect[ind_collect], yerr=error_collect[ind_collect], fmt='o-',
            label="McStas")
ax.plot(simu_h, simu_inten, "o-", label="Python")

ax.legend()
ax.set_xlabel("(h, {:.2f}, {:.2f}) (rlu)".format(k, l))
ax.set_ylabel("Intensity normalised to 1")
ax.set_title(r"$\hbar\omega$ = {:.2f} meV".format(nctx.joule2mev(hw)))
ax.tick_params(axis="both", direction="in")
ax.set_yscale("log")
fig.savefig("{}LineNew_ki{:.1f}_h_k{:.2f}_l{:.2f}.png".format(glb.path_mcstas, ki * 1e-10, k, l),
            bbox_inches='tight')
plt.close(fig)
