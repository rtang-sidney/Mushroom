import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import neutron_context as nctx
from global_context import comment_symbol
from mushroom_context import MushroomContext
import global_context as glb
import geometry_calculation as geo

# It is probably not in use any longer

# np.set_printoptions(threshold=sys.maxsize, precision=2)

FLAT = "flat"

PSD_PREFIX = "psd"
LD_PREFIX = "l"
DATAFILE_EXTENSION = "dat"

FOLDER = "250121"
PSDFLAT_COMPONENT = "psd_flat"
LD_COMP = "l_monitor"

PSDCYL_XVAR = "ra"
PSDCYL_YVAR = "y"
PSD_XVAR = "X"
PSD_YVAR = "Y"

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
UNIT_DEG = "deg"

WAVENUMBER_INCOMING = "ki"
PATTERN_SIZE = re.compile(r"([0-9]*),\s?([0-9]*)")
PATTERN_XYLIMITS = re.compile(
    r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_POSITION = re.compile(r"\s*([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)\s([-+]?[0-9]*\.?[0-9]*)")
PATTERN_PARAM = re.compile(r"\s*(\S*_?\S*)=([-+]?[0-9]*\.?[0-9]*)")
PATTERN_LABEL = re.compile(r"\s*(\S*\s*\S*)\s*\[(\S*)]")

SCAN_FOLDER_PREFIX = "Angle"


class PsdInformation:
    def __init__(self, ki, sa_ang, an_deg, psd_form):
        """
        initialise the object with the PSD information
        :param ki: incoming wavenumber
        :param sa_ang: current sample rotation
        :param an_deg: the scan angle that defines the folder storing the simulated data
        :param psd_form: denotes which PSD is to be handled
        """
        mush_ctx = MushroomContext()
        subfold = scan2folder(ki, sa_ang, an_deg)
        filepath = detector_filepath(detector_type=PSD_PREFIX, detector_form=psd_form, folder=subfold)
        f = open(file=filepath).readlines()
        keys = []
        contents = []
        for line in f:
            if line.startswith(comment_symbol):
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
        intensities = np.loadtxt(fname=filepath, comments=comment_symbol, max_rows=self.ysize)
        self.x_1d, self.y_1d, self.intensities = self._psd_signal_adjust(x_1d, y_1d, intensities, geo_ctx=mush_ctx)

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

    def _psd_signal_adjust(self, x, y, intensity, geo_ctx: MushroomContext):
        component = self.metadata_dict[KEY_COMPONENT]
        if component == PSDFLAT_COMPONENT:
            x, y = np.meshgrid(x, y)
        else:
            raise RuntimeError("Cannot match the x variable {}.".format(self.metadata_dict[KEY_XVAR]))
        x, y = x.flatten(), y.flatten()
        intensity = intensity.flatten()
        return x, y, intensity


def ki2folder(ki):
    return 'ki{:.1f}'.format(ki * 1e-10)


def scan2folder(ki, sa_deg, an_deg):
    folder = ki2folder(ki)
    subfold = '{:s}{}/sa{:d}_an{:d}'.format(glb.path_mcstas, folder, int(sa_deg), int(an_deg))
    return subfold


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


def position2dispersion(x, y, ki, psd_type, mush_ctx: MushroomContext):
    """
    calculate the scattering variables from the PSD information
    :param x: x-axis of the psd file
    :param y: y-axis of the psd file
    :param ki: incoming wavenumber
    :param psd_type: type of the psd monitor, determining the interpretation of the x- & y-axes
    :param mush_ctx: Mushroom context
    :return: outgoing wavenumber kf (for comparing the wavelengths with the lambda-monitors), Q-vectors and 
    energy transfer hbar * omega
    """
    if psd_type == FLAT:
        rad_1d = np.linalg.norm([x, y], axis=0)
        azi_1d = np.arctan2(y, x)
        indices = mush_ctx.dete_points.shape[0] - np.searchsorted(mush_ctx.dete_points[::-1, 0], rad_1d)
    else:
        raise RuntimeError("Cannot match the component {:s}.".format(psd_type))
    kf = mush_ctx.wavenumber_f
    pol_1d = mush_ctx.pol_angles[indices]
    kfx = kf * np.cos(pol_1d) * np.cos(azi_1d)
    kfy = kf * np.cos(pol_1d) * np.sin(azi_1d)
    kfz = kf * np.sin(pol_1d)
    qx = ki - kfx
    qy = -kfy
    qz = -kfz
    return qx, qy, qz


def merge_analysers(ki, sam_rot, psd_form):
    # merge the PSD data for different analyser arrays at the same sample rotation
    psd_x, psd_y, psd_intensity = None, None, None
    for i, an_ang in enumerate(an_angles):
        an_ang = int(round(an_ang))
        if i == 0:
            psd = PsdInformation(ki=ki, sa_ang=sam_rot, an_deg=an_ang, psd_form=psd_form)
            psd_x, psd_y, psd_intensity = psd.x_1d, psd.y_1d, psd.intensities
        else:
            psd_intensity += PsdInformation(ki=ki, sa_ang=sam_rot, an_deg=an_ang, psd_form=psd_form).intensities
    return psd_x, psd_y, psd_intensity


def merge_sample(ki, mush_ctx: MushroomContext):
    def data_per_rot(sam_rot, ki, mush_ctx: MushroomContext):
        x, y, inten = merge_analysers(ki, sam_rot, FLAT)
        qx, qy, qz = position2dispersion(x=x, y=y, ki=ki, psd_type=FLAT, mush_ctx=mush_ctx)
        qvectors = np.array([qx, qy, qz])
        return qvectors, inten

    qvectors_per_ki, inten_per_ki = None, None
    for i, sa_ang in enumerate(sa_angles):
        sa_ang = int(round(sa_ang))
        print("Sample {:d} deg".format(sa_ang))
        if i == 0:
            qvectors_per_ki, inten_per_ki = data_per_rot(sa_ang, ki=ki, mush_ctx=mush_ctx)
        else:
            qvectors_now, inten_now = data_per_rot(sa_ang, ki=ki, mush_ctx=mush_ctx)
            qvectors_per_ki = np.append(qvectors_per_ki, qvectors_now, axis=1)
            inten_per_ki = np.append(inten_per_ki, inten_now)
    inidices_finite_inten = inten_per_ki > 0
    qvectors_per_ki = qvectors_per_ki[:, inidices_finite_inten]
    inten_per_ki = inten_per_ki[inidices_finite_inten]
    return qvectors_per_ki, inten_per_ki


def write_dispersion(ki, mush_ctx: MushroomContext):
    q_vectors, intensities = merge_sample(ki=ki, mush_ctx=mush_ctx)
    glb.write_dispersion(prefix=glb.prefix_mcstas, ki=ki, q_vectors=q_vectors, intensities=intensities, order=1)


def qrluxi(point, points, hkl):
    point = np.array(point)
    hkl = np.array(hkl)
    for i, x in enumerate(hkl):
        x = int(round(x))
        if x != 0:
            return (points[i, :] - point[i]) / float(x)


def select_points(points, vec1, vec2, point_on, resol=0.01):
    distance_point2plane = np.apply_along_axis(func1d=geo.point2plane, axis=0, arr=points, vec1=vec1, vec2=vec2,
                                               point_on=point_on)
    accepted_index = distance_point2plane < resol
    return accepted_index


def plot_plane(ki, vec1, vec2, point_on):
    """
    plot the simulated data on the Q vector grid generated by two directions. The plots are collected as an animation
    along the energy axis, which depends on the ki values
    :param vec1: vector1 defining the 1st direction
    :param vec2: vector2 defining the 1st direction
    :param point_on:
    :return: nothing
    """
    print("Step: ki{}, vec1{}, vec2{}, point on-plane{}".format(ki * 1e-10, vec1, vec2, point_on))
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    point_on = np.array(point_on)
    fname = glb.fname_write_dispersion(prefix=glb.prefix_mcstas, ki=ki, order=1, path=glb.path_mcstas)
    data = np.loadtxt(fname, delimiter=",").transpose()
    q_vectors = data[:-1, :]
    intensities = data[-1, :]
    hw = nctx.wavenumber2energy(ki) - nctx.wavenumber2energy(mushroomctx.wavenumber_f)
    q_rlu = nctx.q2rlu(q_value=q_vectors, l_const=latt_const)
    select_indices = select_points(q_rlu, vec1, vec2, point_on)
    q_plot = q_rlu[:, select_indices]
    if q_plot is None or q_plot.shape[1] == 0:
        fig, ax = plt.subplots()
        fig.savefig("{}IntensityPlane_{:.1f}.png".format(glb.path_mcstas, ki * 1e-10), bbox_inches='tight')
        plt.close(fig)
        return
    inten_plot = intensities[select_indices]

    # xi1 = qrluxi(point=point_on, points=q_rlu, hkl=vec1)
    # xi2 = qrluxi(point=point_on, points=q_rlu, hkl=vec2)
    xi1 = np.apply_along_axis(np.dot, axis=0, arr=q_plot, b=vec1)
    xi2 = np.apply_along_axis(np.dot, axis=0, arr=q_plot, b=vec2)

    fig, ax = plt.subplots()
    cnt = ax.scatter(xi1, xi2, c=inten_plot)
    cbar = fig.colorbar(cnt)
    cbar.set_label("Intensity")
    ax.set_xlabel("{}, {}, {}".format(*vec1))
    ax.set_ylabel("{}, {}, {}".format(*vec2))
    ax.set_title("Energy transfer{:.1f}".format(nctx.joule2mev(hw)))
    ax.tick_params(axis="both", direction="in")
    fig.savefig("{}IntensityPlane_{:.1f}.png".format(glb.path_mcstas, ki * 1e-10), bbox_inches='tight')
    plt.close(fig)


latt_const = 4.5 * 1e-10
mushroomctx = MushroomContext()
ki_values = np.linspace(1.2, 1.6, num=5) * 1e10
sa_angles = np.linspace(0, 90, num=10)
an_angles = np.linspace(10, 170, num=17)
direction1 = (1, 1, 0)
direction2 = (0, 0, 1)
point_onplane = (1, 0, 0)
for ki in ki_values:
    # write_dispersion(ki, mush_ctx=mushroomctx)
    plot_plane(ki, vec1=direction1, vec2=direction2, point_on=point_onplane)
