import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
from magnonmodel import MagnonModel
import geometry_calculation as geo
import neutron_context as neutron
import instrument_context as instr
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 18})

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082
"""

# Comment from Alex <3
# line: ax + by + c = 0 -> (a, b, c)
TERM_MAGNON = "MagnonModel"
TERM_SCATTERING = "scattering"  # energy conservation at the scattering
TERM_CROSS_SECTION = "cross_section"  # Delta function is approximated by Gaussian function with intensity distribution
TERM_MUSHROOM = "Mushroom"
TERM_MUSHROOM_MAGNON = "MushroomMagnon"
CALC_TERMS = [TERM_MUSHROOM_MAGNON]  # TERM_MUSHROOM, TERM_MAGNON, TERM_MUSHROOM_MAGNON

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]
NORM = "norm"

ROTATION_STEP = np.deg2rad(1)  # sample rotation step size
ROTATION_NUMBERS = [10, 30]  # number of steps, 0, 10, 30, 90, 180
DIM_1 = "norm"
DIM_LINE = "line"
DIM_2 = "2d"
DIM_PLANE = "plane"
DIM_3 = "3d"
DIM_4 = "4d"
DIM_PROJ = [DIM_1, DIM_2, DIM_3, DIM_4]
DIM_INTERSECT = [DIM_LINE, DIM_PLANE]
DIMENSIONS = [DIM_LINE]  # DIM_1, DIM_2, DIM_3,DIM_4, DIM_1_LINE, DIM_2_HW

EXTENSION_PDF = "pdf"
EXTENSION_PNG = "png"

MAGNON_DEFAULT = "Default"
MAGNON_IRON = "Fe"  # Fe BCC

Q_UNIT_REAL = r"$\AA^{-1}$"  # the real unit of Q-vector
Q_UNIT_RLU = "r.l.u."  # reciprocal lattice unit
HW_LABEL = r"$\hbar\omega$ (meV)"

SAMPLE_INIT_ROT = np.array([1, 0, 0])
# SAMPLE_TILT_AXIS1 = np.array([0, 1, 0])
# SAMPLE_TILT_ANGLES = [90]
# SAMPLE_TILT_ANGLE2 = np.deg2rad(0)
# POINTS_INTEREST = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
POINTS_INTEREST = [np.array([0, 0, 0])]

HKL_100_NAME = "100"
HKL_010_NAME = "010"
HKL_001_NAME = "001"
HKL_110_NAME = "110"
HKL_101_NAME = "101"
HKL_011_NAME = "011"
HKL_111_NAME = "111"
HKL_211_NAME = "211"
HKL_121_NAME = "121"
HKL_221_NAME = "221"
HKL_100 = np.array([1, 0, 0])
HKL_010 = np.array([0, 1, 0])
HKL_001 = np.array([0, 0, 1])
HKL_110 = np.array([1, 1, 0])
HKL_101 = np.array([1, 0, 1])
HKL_011 = np.array([0, 1, 1])
HKL_111 = np.array([1, 1, 1])
HKL_211 = np.array([2, 1, 1])
HKL_121 = np.array([1, 2, 1])
HKL_221 = np.array([2, 2, 1])
HKL_NAMES = [HKL_100_NAME, HKL_010_NAME, HKL_001_NAME, HKL_110_NAME, HKL_101_NAME, HKL_011_NAME, HKL_111_NAME,
             HKL_211_NAME, HKL_121_NAME, HKL_221_NAME]
# HKL_100_NAME, HKL_010_NAME, HKL_001_NAME, HKL_110_NAME, HKL_101_NAME, HKL_011_NAME, HKL_111_NAME
HKL_VECTORS = [HKL_100, HKL_010, HKL_001, HKL_110, HKL_101, HKL_011, HKL_111, HKL_211, HKL_121, HKL_221]
# HKL_100, HKL_010, HKL_001, HKL_110, HKL_101, HKL_011, HKL_111
HKL_DICT = dict(zip(HKL_NAMES, HKL_VECTORS))

WAVENUMBERS_IN = [1.5e10]  # , 1.6e10


# PLOTPATH = "..\Report\Calculations\\"


def monochromator_angular_spread(divergence_in, divergence_out, mosaic):
    # For the formula see [Paper1]
    # alpha_i, alpha_f, eta = divergence_in, divergence_out, mosaic
    numerator = divergence_in ** 2 * divergence_out ** 2 + mosaic ** 2 * divergence_in ** 2 + \
                mosaic ** 2 * divergence_out ** 2
    denominator = 4 * mosaic ** 2 + divergence_in ** 2 + divergence_out ** 2
    return np.sqrt(numerator / denominator)


def angular_res_an(geo_ctx: MushroomContext, an_index):
    eta = instr.moasic_analyser  # mosaic
    alpha_i, alpha_f = vertical_divergence_analyser(geo_ctx=geo_ctx,
                                                    analyser_index=an_index)  # incoming and outgoing divergence
    # For the formula see [Paper1]
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2

    return np.sqrt(numerator / denominator)


def uncert_kf(geo_ctx: MushroomContext, an_ind_now, an_ind_near):
    kf_now = geo_ctx.wavenumbers_out[an_ind_now]
    kf_near = geo_ctx.wavenumbers_out[an_ind_near]
    angular_uncertainty_analyser = monochromator_angular_spread(
        *vertical_divergence_analyser(geo_ctx=geo_ctx, analyser_index=an_ind_now),
        mosaic=instr.moasic_analyser)
    twotheta_an = geo_ctx.an_2theta[an_ind_now]
    uncertainty_kf_bragg = kf_now * np.linalg.norm(
        [instr.deltad_d, angular_uncertainty_analyser / np.tan(twotheta_an / 2.0)])  # from Bragg's law
    uncertainty_kf_segment = abs(kf_now - kf_near)
    return max(uncertainty_kf_bragg, uncertainty_kf_segment)


def uncert_pol(geo_ctx: MushroomContext, an_index):
    divergence = angular_res_an(geo_ctx=geo_ctx, an_index=an_index)
    # the uncertainty of the polar angle is given by the angular resolution at the analyser
    # return get_angular_resolution_analyser(geo_ctx=geo_ctx,  analyser_point=analyser_point)
    return divergence


def uncert_azi(geo_ctx: MushroomContext, an_index):
    an_point = (geo_ctx.analyser_points[0][an_index], geo_ctx.analyser_points[1][an_index])
    # sa: sample-analyser; af: analyser-focus
    distance_sa = geo.points_distance(point1=geo_ctx.sample_point, point2=an_point)
    distance_af = geo.points_distance(point1=an_point, point2=geo_ctx.foc_point)
    uncert_azi_sa = 2.0 * np.arctan((instr.an_seg + instr.sample_diameter) / (2.0 * distance_sa))
    uncert_azi_af = 2.0 * np.arctan((instr.an_seg + geo_ctx.foc_size) / (2.0 * distance_af))
    return min(uncert_azi_sa, uncert_azi_af)


def de_of_e_from_an(geo_ctx: MushroomContext, an_ind_now, an_ind_near):
    kf_now = geo_ctx.wavenumbers_out[an_ind_now]
    delta_kf = uncert_kf(geo_ctx, an_ind_now=an_ind_now, an_ind_near=an_ind_near) * spread_factor_detector(
        geo_ctx=geo_ctx, index_now=an_ind_now, index_nearest=an_ind_near)[0]
    return 2. * delta_kf / kf_now


# def get_divergence(sample, analyser_point, focus, sample_size, focus_size):
def vertical_divergence_analyser(geo_ctx: MushroomContext, analyser_index):
    # sa: sample-analyser; af: analyser-focus
    analyser_point = (geo_ctx.analyser_points[0][analyser_index], geo_ctx.analyser_points[1][analyser_index])
    vector_sa = geo.points_to_vector(point1=geo_ctx.sample_point, point2=analyser_point)
    vector_af = geo.points_to_vector(point1=analyser_point, point2=geo_ctx.foc_point)
    vector_tangential = geo.vector_bisector(vector_sa, vector_af)
    segment_analyser = geo.unit_vector(vector_tangential) * instr.an_seg
    analyser_incoming_projection = geo.vector_project_a2b(segment_analyser, vector_sa)
    analyser_incoming_rejection = segment_analyser - analyser_incoming_projection
    analyser_outgoing_projection = geo.vector_project_a2b(segment_analyser, vector_af)
    analyser_outgoing_rejection = segment_analyser - analyser_outgoing_projection

    divergence_in = 2 * np.arctan((instr.sample_height * abs(
        np.cos(geo.points_to_slope_radian(point1=geo_ctx.sample_point, point2=analyser_point))) + np.linalg.norm(
        analyser_incoming_rejection)) / (2.0 * np.linalg.norm(vector_sa)))
    divergence_out = 2 * np.arctan((geo_ctx.foc_size * abs(
        np.sin(geo.points_to_slope_radian(point1=analyser_point, point2=geo_ctx.foc_point))) + np.linalg.norm(
        analyser_outgoing_rejection)) / (2.0 * np.linalg.norm(vector_af)))
    # divergence_in = instr.sample_size / distance_sa
    # divergence_out = geo_ctx.focus_size / distance_af
    # print(divergence_in, divergence_out)
    return divergence_in, divergence_out


def kf_resol_mcstas(geo_ctx: MushroomContext, index_now, index_nearest):
    factor_polar, factor_azimuth = spread_factor_detector(geo_ctx=geo_ctx, index_now=index_now,
                                                          index_nearest=index_nearest)
    # factor_polar, factor_azimuth = 1, 1
    dkf = uncert_kf(geo_ctx, an_ind_now=index_now,
                    an_ind_near=index_nearest) * factor_polar
    dphi = uncert_pol(geo_ctx, an_index=index_now) * factor_polar
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta = uncert_azi(geo_ctx=geo_ctx,
                        an_index=index_now) * factor_azimuth
    kf = geo_ctx.wavenumbers_out[index_now]
    dkf_x = kf * np.tan(dtheta)
    dkf_y = kf * np.tan(dphi)
    dkf_z = dkf
    return [dkf_x, dkf_y, dkf_z]


# to compare the analyser generated by the two different methods
def plot_analyser_comparison(geo_ctx: MushroomContext):
    points_analyser_x, points_analyser_y = geo_ctx.theo_ellipse_points[0], geo_ctx.theo_ellipse_points[1]
    points_x, points_y = geo_ctx.analyser_points[0], geo_ctx.analyser_points[1]
    plt.figure(10)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(points_x, points_y)
    plt.plot(points_analyser_x, points_analyser_y)
    plt.legend((r"Segments with 1x1 cm$^2$", "Ideal ellipse"))
    plt.text(0.3, -0.3, "Number of segments in one cut-plane: {:d}".format(len(points_x)))
    plt.text(0.3, -0.35, "Largest deviation from the ideal ellipse: {:5.2f} m".format(
        geo.points_distance([points_x[-1], points_y[-1]], [points_analyser_x[-1], points_analyser_y[-1]])))
    plt.xlabel("x axis (m)")
    plt.ylabel("y axis (m)")
    plt.plot(*geometryctx.sample_point, "ro")
    plt.plot(*geometryctx.foc_point, "ro")
    plt.text(x=0, y=-0.05, s="Sample")
    plt.text(x=0.1, y=-0.4, s="Focus")

    plt.plot([geometryctx.sample_point[0], geometryctx.foc_point[0]],
             [geometryctx.sample_point[1], geometryctx.foc_point[1]])
    bisecting_x = np.array([0.75, 1])
    plt.plot(bisecting_x, geo.line_to_y(bisecting_x, geo.points_bisecting_line(point1=geometryctx.sample_point,
                                                                               point2=geometryctx.foc_point)))
    plt.axis("equal")

    plot_filename = "Geometry_Comparison.pdf"
    plt.tick_params(axis="both", direction="in")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(10)
    print("Plot saved: {:s}".format(plot_filename))


def plot_geometry(geo_ctx: MushroomContext):
    # to plot the geometry with important parameters
    def plot_for_analyser_point(index_now, index_nearest):
        nonlocal geo_ctx
        energy_out = neutron.wavelength2energy(
            wavelength=geo_ctx.wavelength_bragg(index=index_now))
        res_ef = de_of_e_from_an(geo_ctx=geo_ctx, an_ind_now=index_now,
                                 an_ind_near=index_nearest)
        res_ef *= energy_out
        analyser_point = (geo_ctx.analyser_points[0][index_now], geo_ctx.analyser_points[1][index_now])
        detector_point = (geo_ctx.detector_points[0][index_now], geo_ctx.detector_points[1][index_now])
        line_sp_plot = ([geo_ctx.sample_point[0], analyser_point[0]], [geo_ctx.sample_point[1], analyser_point[1]])
        line_pf_plot = ([analyser_point[0], detector_point[0]], [analyser_point[1], detector_point[1]])
        plt.plot(*line_sp_plot, color='#17becf')
        plt.plot(*line_pf_plot, color='#17becf')

        line_sp_plot = ([geo_ctx.sample_point[0], -analyser_point[0]], [geo_ctx.sample_point[1], analyser_point[1]])
        line_pf_plot = ([-analyser_point[0], -detector_point[0]], [analyser_point[1], detector_point[1]])
        plt.plot(*line_sp_plot, color='#17becf')
        plt.plot(*line_pf_plot, color='#17becf')

        plt.plot(analyser_point[0], analyser_point[1], "ko")
        plt.text(x=-analyser_point[0] * 1.05 - 0.8, y=analyser_point[1] * 1.05 + 0.05,
                 s="{:5.2f}".format(neutron.joule2mev(energy_out)))
        plt.text(x=analyser_point[0] * 1.05, y=analyser_point[1] * 1.05 + 0.05,
                 s="{:5.2f}".format(neutron.joule2mev(res_ef) * 1e3))

    # first plot the analyser on both sides
    plt.plot(geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.plot(-geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.xlabel("Radial axis (m)")
    plt.ylabel("Vertical axis (m)")

    plt.text(x=-2.3, y=0.4, s=r"$E_f$ (meV)")
    plt.text(x=1, y=0.4, s=r"$\Delta E_f$ ($\mu$eV)")

    plt.text(-2.6, -3.4,
             r"Wavenumber $k_f \in$ [{:.2f}, {:.2f}] $\AA^{{-1}}$".format(np.min(geometryctx.wavenumbers_out) * 1e-10,
                                                                          np.max(geometryctx.wavenumbers_out) * 1e-10))

    plot_for_analyser_point(index_now=0, index_nearest=1)
    plot_for_analyser_point(index_now=-1, index_nearest=-2)

    index_largest_energy = np.argmax(geo_ctx.wavenumbers_out)
    plot_for_analyser_point(index_now=index_largest_energy, index_nearest=index_largest_energy + 1)

    # mark the position of the sample and focus, and plot the detector
    plt.plot(*geo_ctx.sample_point, "ro")
    plt.text(x=-0.15, y=-1.2, s="Sample", rotation=90)
    plt.plot(*geo_ctx.foc_point, "ro", alpha=0.5)
    plt.text(x=geo_ctx.foc_point[0] - 1, y=geo_ctx.foc_point[1] - 0.1, s="Focus")
    plt.plot(*geo_ctx.detector_points, '.', color='#8c564b')
    plt.plot(-geo_ctx.detector_points[0], geo_ctx.detector_points[1], '.', color='#8c564b')

    plt.axis("equal")
    # plt.title("Geometry (sectional view)")
    plot_filename = ".".join([geo_ctx.filename_geo, EXTENSION_PNG])
    plt.tick_params(axis="both", direction="in")
    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.savefig(geo_ctx.filename_geo + '.png', bbox_inches='tight')
    plt.close(1)
    print("Plot saved: {:s}".format(plot_filename))


def write_mcstas(geo_ctx: MushroomContext):
    # to write the file giving the information of the analyser array for the McStas simulation
    f = open(geo_ctx.filename_mcstas, 'w+')
    value_width_z = instr.an_seg
    value_height_y = instr.an_seg
    value_mosaic_horizontal = geo.deg2min(np.rad2deg(instr.moasic_analyser))
    value_mosaic_vertical = geo.deg2min(np.rad2deg(instr.moasic_analyser))
    value_lattice_distance = instr.lattice_distance_pg002 * 1e10  # it is in angstrom for McStas
    value_position_y = geo_ctx.analyser_points[1]
    value_position_z = geo_ctx.analyser_points[0]
    value_rotation_x = -np.rad2deg(geo_ctx.mcstas_rotation_rad)

    # This is the code for analyser segments at one azimuthal angle without arms
    for i in range(geo_ctx.analyser_points[0].shape[0]):
        string_an1 = 'COMPONENT {}{} = {}({} = {}, {} = {}, {} = {}, {} = {}, {} = {})\n'.format(
            geo_ctx.component_name_prefix, i, geo_ctx.component_type, geo_ctx.parameter_width_z, value_width_z,
            geo_ctx.parameter_height_y, value_height_y, geo_ctx.parameter_mosaic_horizontal,
            value_mosaic_horizontal,
            geo_ctx.parameter_mosaic_vertical, value_mosaic_vertical, geo_ctx.parameter_lattice_distance,
            value_lattice_distance)
        string_an2 = 'AT (0, {}, {}) RELATIVE {}\n'.format(value_position_y[i], value_position_z[i],
                                                           geo_ctx.component_reference)
        string_an3 = 'ROTATED ({}, 0, 90) RELATIVE {}\n'.format(value_rotation_x[i], geo_ctx.component_reference)
        # string_an3 = 'ROTATED (90, 90, 0) RELATIVE {}\n'.format(geo_ctx.component_reference)
        string_an4 = 'GROUP {}\n\n'.format(geo_ctx.group_name)
        string_analyser = string_an1 + string_an2 + string_an3 + string_an4
        f.write(string_analyser)
    f.close()
    print("McStas file saved: {:s}".format(geo_ctx.filename_mcstas))


def plot_resolution_polarangles(geo_ctx: MushroomContext):
    # to plot the resolution with the x-axis being the polar angle
    # is more realistic since the same kf-value can originate from two angles

    def forward(x):
        return np.interp(x, polar_angles, all_kf * 1e-10)

    def inverse(x):
        kf_sort_indeces = np.argsort(all_kf)
        kf_sort = all_kf[kf_sort_indeces]
        polar_angles_sort = polar_angles[kf_sort_indeces]
        return np.interp(x, kf_sort * 1e-10, polar_angles_sort)

    polar_angles = np.rad2deg(geo_ctx.pol_angles)
    all_kf = geo_ctx.wavenumbers_out
    all_dqx_m, all_dqy_m, all_dqz_m = resolution_calculation(geo_ctx=geo_ctx)

    fig, ax = plt.subplots()
    ax.plot(polar_angles, all_dqx_m * 1e-10, color="blue")
    ax.plot(polar_angles, all_dqy_m * 1e-10, color="red")
    ax.plot(polar_angles, all_dqz_m * 1e-10, color="gold")
    ax.set_xlabel(r"Polar angle $\varphi$ (degree)")
    ax.set_ylabel(r"$\Delta k_f$ ($\AA^{-1}$)")
    ax.grid()
    ax.legend(("x: horizontal", "y: vertical", r"z: along $k_f$"))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    ax2.plot(polar_angles, all_dqz_m / all_kf * 1e2, '1', color=colour_ax2)
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"$\dfrac{\Delta k_f}{k_f}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)

    secax = ax.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel(r"Wavenumber $k_f$ ($\AA^{-1}$)")
    secax.tick_params(axis="x", direction="in", labelsize=10)

    ax.set_title('Q-resolution of the secondary spectrometer')
    filename = "Resolution_phi.pdf"
    plt.savefig(filename, bbox_inches='tight')
    # plt.savefig(filename + '.png', bbox_inches='tight')
    print("Plot saved: {:s}".format(filename))


def plot_resolution_kf(geo_ctx: MushroomContext):
    # to plot the resolution with the x-axis being the outgoing wave-number
    all_kf = geo_ctx.wavenumbers_out
    max_index = np.argmax(all_kf)
    all_dqx_m, all_dqy_m, all_dqz_m = resolution_calculation(geo_ctx=geo_ctx)

    fig, ax = plt.subplots()
    ax.plot(all_kf[max_index:] * 1e-10, all_dqx_m[max_index:] * 1e-10, color="blue", label="x: horizontal")
    ax.plot(all_kf[max_index:] * 1e-10, all_dqy_m[max_index:] * 1e-10, color="red", label="y: vertical")
    ax.plot(all_kf[max_index:] * 1e-10, all_dqz_m[max_index:] * 1e-10, color="darkgoldenrod", label="z: along $k_i$")
    ax.set_xlabel(r'Wavenumber $k_f$ ($\AA^{-1}$)')
    ax.set_ylabel(r"Uncertainty $\Delta k_{f,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    ax.grid()
    ax.legend(labelcolor=["blue", "red", "darkgoldenrod"], loc='upper left', bbox_to_anchor=(0, 1),
              framealpha=0.5)
    ax.tick_params(axis="both", direction="in")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    dkf_percent = all_dqz_m / all_kf * 1e2
    ax2.plot(all_kf[max_index:] * 1e-10, dkf_percent[max_index:], color=colour_ax2)
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"$\dfrac{\Delta k_f}{k_f}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', color=colour_ax2, labelcolor=colour_ax2)
    ax2.legend(["Relative uncertainty"], loc='lower left', bbox_to_anchor=(0, 0.5), labelcolor=colour_ax2)
    ax.set_title('Secondary spectrometer')
    filename = geo_ctx.filename_res
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.savefig(filename + '.png', bbox_inches='tight')
    print("Plot saved: {:s}".format(filename + '.png'))


def resolution_calculation(geo_ctx: MushroomContext):
    # only the dimension of the polar angles is considered, since it is symmetric in the respective horizontal planes
    all_dqx_m = np.array(list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, index_now=i,
                                                            index_nearest=1 if i == 0 else i - 1)[0],
                                  range(geo_ctx.pol_angles.shape[0]))))
    all_dqy_m = np.array(list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, index_now=i,
                                                            index_nearest=1 if i == 0 else i - 1)[1],
                                  range(geo_ctx.pol_angles.shape[0]))))
    all_dqz_m = np.array(list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, index_now=i,
                                                            index_nearest=1 if i == 0 else i - 1)[2],
                                  range(geo_ctx.pol_angles.shape[0]))))
    return all_dqx_m, all_dqy_m, all_dqz_m


def spread_factor_detector(geo_ctx: MushroomContext, index_now, index_nearest):
    an_point = (geo_ctx.analyser_points[0][index_now], geo_ctx.analyser_points[1][index_now])
    detector_now = [geo_ctx.detector_points[0][index_now], geo_ctx.detector_points[1][index_now]]
    detector_next = [geo_ctx.detector_points[0][index_nearest], geo_ctx.detector_points[1][index_nearest]]
    spread_factor_polar = max(1, instr.detector_resolution / geo.points_distance(detector_now, detector_next))

    detector_spread_azi = instr.an_seg * geo.points_distance(detector_now, geo_ctx.foc_point) / geo.points_distance(
        an_point, geo_ctx.foc_point)
    spread_factor_azi = max(1, instr.detector_resolution / detector_spread_azi)
    return spread_factor_polar, spread_factor_azi


def distances_fd_as(geo_ctx: MushroomContext, index):
    distance_fd = geo.points_distance(point1=geo_ctx.foc_point,
                                      point2=[geo_ctx.detector_points[0][index], geo_ctx.detector_points[1][index]])
    distance_as = geo.points_distance(point1=geo_ctx.foc_point,
                                      point2=[geo_ctx.analyser_points[0][index], geo_ctx.analyser_points[1][index]])
    return distance_fd / distance_as


def plot_distance_fd_as(geo_ctx: MushroomContext):
    ratio_fd_af = np.array(
        list(map(lambda i: distances_fd_as(geo_ctx=geo_ctx, index=i), range(geometryctx.pol_angles.shape[0]))))
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(geometryctx.pol_angles), ratio_fd_af)
    # plotting_format(ax=ax, grid=True)
    ax.tick_params(axis="both", direction="in")
    # if grid is True:
    ax.grid()
    ax.set_xlabel("Polar angle of analyser segment (degree)")
    ax.set_ylabel(r"Ratio $\frac{D_{FD}}{D_{AF}}$")
    ax.set_title("Distance-ratio focus-detector / analyser-focus")
    plt.tight_layout()
    filename = "Distance_fd_af.png"
    plt.savefig(filename, bbox_inches='tight')
    print("{:s} plotted.".format(filename))
    plt.close(fig)


def wavenumbers_psd(geo_ctx: MushroomContext):
    kf = geo_ctx.wavenumbers_out
    detec_hori_x = geo_ctx.dete_hori_x
    detec_vert_y = geo_ctx.dete_vert_y
    fig, axs = plt.subplots(1, 2, sharey="all")
    axs[0].plot(detec_vert_y, kf[detec_hori_x.shape[0]:] * 1e-10)
    axs[0].set_xlim(axs[0].get_xlim()[::-1])
    axs[1].plot(detec_hori_x, kf[:detec_hori_x.shape[0]] * 1e-10)
    axs[0].set_xlabel("Vertical position of\n cyl. PSDs (m)")
    axs[1].set_xlabel("Radial position of\n flat PSD (m)")
    axs[0].set_ylabel(r"Wavenumber $k_f$ ($\AA^{-1}$)")
    axs[0].grid()
    axs[1].grid()
    axs[0].tick_params(axis="both", direction="in")
    axs[1].tick_params(axis="both", direction="in")
    plot_filename = "kf_values_psd_positions.pdf"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: {:s}".format(plot_filename))


def magnon_scattered(scattering_de, magnon_de, de_of_e):
    if abs((scattering_de - magnon_de) / magnon_de) < de_of_e or abs(
            (scattering_de + magnon_de) / magnon_de) < de_of_e:
        return scattering_de
    else:
        return np.nan


def wavevector_transfer_rotation(rot_angle, wavevector_transfer):
    new_qx, new_qy = geo.rotation_around_z(rot_angle=rot_angle, old_x=wavevector_transfer[0],
                                           old_y=wavevector_transfer[1])
    wavevector_transfer[:2] = new_qx, new_qy
    return wavevector_transfer


def mushroom_wavevector_transfer(geo_ctx: MushroomContext):
    mushroom_qx = np.array(list(map(lambda i: np.array(list(
        map(lambda j: geo_ctx.wavevector_transfer(index_pol=i, index_azi=j)[0], range(geo_ctx.azi_angles.shape[0])))),
                                    range(geo_ctx.pol_angles.shape[0]))))
    mushroom_qy = np.array(list(map(lambda i: np.array(list(
        map(lambda j: geo_ctx.wavevector_transfer(index_pol=i, index_azi=j)[1], range(geo_ctx.azi_angles.shape[0])))),
                                    range(geo_ctx.pol_angles.shape[0]))))
    mushroom_qz = np.array(list(map(lambda i: np.array(list(
        map(lambda j: geo_ctx.wavevector_transfer(index_pol=i, index_azi=j)[2], range(geo_ctx.azi_angles.shape[0])))),
                                    range(geo_ctx.pol_angles.shape[0]))))
    return np.array([mushroom_qx, mushroom_qy, mushroom_qz])


def mushroom_energy_transfer(geo_ctx: MushroomContext):
    return np.array(list(map(lambda i: np.array(list(map(
        lambda j: neutron.planck_constant ** 2 * (geo_ctx.wavenumber_in ** 2 - geo_ctx.wavenumbers_out[i] ** 2) / (
                2 * neutron.mass_neutron), range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.pol_angles.shape[0]))))


def mushroom_q_resolution(geo_ctx: MushroomContext):
    resol_qx, resol_qy, resol_qz = np.array(
        list(map(lambda i: resolution_calculation(geo_ctx=geo_ctx), geo_ctx.azi_angles.shape[0])))
    resol_qx = np.transpose(resol_qx)
    resol_qy = np.transpose(resol_qy)
    resol_qz = np.transpose(resol_qz)
    return resol_qx, resol_qy, resol_qz


def sample_init(vert_axis, rot_axis, angle):
    """
    initialise the sample orientation
    :param vert_axis: original axis of the sample pointing to the (0 ,0, 1) direction
    :param rot_axis: rotation axis around which the vertical axis is rotated to the new vertical one
    :param angle: the opposite value will be passed to the rotation formula, since the axis itself rotates
    :return: the new vertical axis
    """
    return geo.rotation_3d(vector=vert_axis, rot_axis=rot_axis, angle=-angle)


def dispersion_calc_plot(geo_ctx: MushroomContext, extension, q_unit):
    # def energy_transfer(qx_per_rot, qy_per_rot, qz_per_rot):
    def energy_transfer(qvector_per_rot, calc_term):

        #  values of all the qx, qy, qz in each sample rotation
        nonlocal geo_ctx
        if calc_term == TERM_MAGNON:
            # magnon_hw = np.array(list(map(lambda i: magnonmdl.magnon_energy(
            #     wavevector_transfer=np.array([qx_per_rot[i], qy_per_rot[i], qz_per_rot[i]])),
            #                               range(qx_per_rot.shape[0]))))
            magnon_hw = np.apply_along_axis(magnonmdl.magnon_energy, axis=0, arr=qvector_per_rot)
            return magnon_hw
        elif calc_term == TERM_MUSHROOM:
            mushroom_hw = mushroom_energy_transfer(geo_ctx=geo_ctx).flatten()
            return mushroom_hw
        elif calc_term == TERM_MUSHROOM_MAGNON:
            magnon_hw = np.apply_along_axis(magnonmdl.magnon_energy, axis=0, arr=qvector_per_rot)
            # magnon_hw = np.array(list(map(lambda i: magnonmdl.magnon_energy(
            #     wavevector_transfer=np.array([qx_per_rot[i], qy_per_rot[i], qz_per_rot[i]])),
            #                               range(qx_per_rot.shape[0]))))
            mushroom_hw = mushroom_energy_transfer(geo_ctx=geo_ctx)
            rela_uncer_hw = np.array(list(map(lambda i: np.array(list(
                map(lambda j: de_of_e_from_an(geo_ctx=geo_ctx, an_ind_now=i, an_ind_near=1 if i == 0 else i - 1),
                    range(mushroom_hw.shape[1])))), range(mushroom_hw.shape[0]))))
            mushroom_hw, rela_uncer_hw = mushroom_hw.flatten(), rela_uncer_hw.flatten()
            # scatter_hw = np.array(list(map(
            #     lambda i: magnon_scattered(scattering_de=mushroom_hw[i], magnon_de=magnon_hw[i],
            #                                de_of_e=rela_uncer_hw[i]), range(qx_per_rot.shape[0]))))
            scatter_hw = np.array(list(map(
                lambda i: magnon_scattered(scattering_de=mushroom_hw[i], magnon_de=magnon_hw[i],
                                           de_of_e=rela_uncer_hw[i]), range(mushroom_hw.shape[0]))))
            return scatter_hw
        else:
            raise ValueError("Invalid term to define the dispersion.")

    def plot_q(q_value):
        nonlocal q_unit
        if q_unit == Q_UNIT_REAL:
            return q_value * 1e-10
        else:
            return q_value / (2 * np.pi / magnonmdl.l_const)

    def rlu2q(rlu):
        return np.array(rlu) * (2 * np.pi / magnonmdl.l_const)

    def line_through_points(line_direction, points, point=None):
        def numbers_one_point(point):
            nonlocal resol, line_direction, points
            return np.count_nonzero(
                np.apply_along_axis(geo.point2line_3d, axis=0, arr=points, line_direction=line_direction,
                                    point_on=point) < resol)

        resol = 0.025e10  # unit m^⁻1
        # TODO:implement the true resolution at each point
        if point is not None:
            distances = np.apply_along_axis(geo.point2line_3d, axis=0, arr=points, line_direction=line_direction,
                                            point_on=point)
            print(np.min(distances) * 1e-10, np.max(distances) * 1e-10)
            other_points = distances < resol
            return other_points
        else:
            which_point = points[:, np.argmax(np.apply_along_axis(numbers_one_point, axis=0, arr=points))]
            other_points = np.apply_along_axis(geo.point2line_3d, axis=0, arr=points, line_direction=line_direction,
                                               point_on=which_point) < resol
            return which_point, other_points

    def q_label(component, unit):
        if component in AXES:
            return r"$Q_{:s}$ ({:s})".format(component, unit)
        elif component == NORM:
            return r"$|\vec{{Q}}|$ ({:s})".format(unit)
        else:
            raise ValueError("Unknown component given. It must be either an axis or the norm")

    def do_plot(calc_term):
        nonlocal fig, ax, rotation, dim_info, geo_ctx, extension
        # sample_tilt1 = int(np.rad2deg(tilt_angle))
        # sample_tilt2 = 0
        # if sample_tilt1 != 0 or sample_tilt2 != 0:
        #     sample_tilt = "({:d},{:d})".format(sample_tilt1, sample_tilt2)
        # else:
        #     sample_tilt = ""
        if calc_term == TERM_MUSHROOM:
            filename = "{:s}_ki{:.1f}_Rotx{:d}_{:s}.{:s}".format(calc_term, geo_ctx.wavenumber_in * 1e-10, rotation,
                                                                 dim_info, extension)
        else:
            filename = "{:s}_ki{:.1f}_{:s}_Rotx{:d}_{:s}.{:s}".format(calc_term, geo_ctx.wavenumber_in * 1e-10,
                                                                      magnonmdl.name, rotation, dim_info, extension)
        ax.tick_params(axis="both", direction="in")  # , pad=-5
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print("Plot saved: {:s}".format(filename))

    def hkl_plot(hkl_element):
        hkl_element = int(round(hkl_element))
        if hkl_element == 0:
            return ""
        else:
            return r"$+ {:d}\xi$".format(hkl_element)

    def points2xi(point, points, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        for x, i in enumerate(hkl):
            x = int(round(x))
            if x != 0:
                return (points[i, :] - point[i]) / float(x)

    def xi2points(point, hkl, xi):
        if isinstance(xi, float):
            point = np.array(point)
            hkl = np.array(hkl)
            return xi * hkl + point
        else:
            raise ValueError("xi should be a float")

    def mark_energy(mark, point, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        return magnonmdl.magnon_energy(wavevector_transfer=np.add(point, mark * hkl))

    # Mushroom values are independent of the sample --> r.l.u. does not make much sense for this
    # if calc_term == TERM_MUSHROOM:
    #     q_unit = Q_UNIT_REAL

    # sample_rot_axis = sample_init(vert_axis=SAMPLE_INIT_ROT, rot_axis=SAMPLE_TILT_AXIS1, angle=tilt_angle)
    print("The axis now: {:.2f},{:.2f},{:.2f}".format(*SAMPLE_INIT_ROT))

    collect_qx, collect_qy, collect_qz = mushroom_wavevector_transfer(geo_ctx=geo_ctx)
    collect_qx = collect_qx.flatten()
    collect_qy = collect_qy.flatten()
    collect_qz = collect_qz.flatten()
    collect_q_vector = np.apply_along_axis(geo.rotation_3d, axis=0, arr=np.array([collect_qx, collect_qy, collect_qz]),
                                           rot_axis=SAMPLE_INIT_ROT, angle=-np.deg2rad(0))

    # collect_dqx, collect_dqy, collect_dqz = mushroom_q_resolution(geo_ctx=geo_ctx)
    # collect_dqx = collect_dqx.flatten()
    # collect_dqy = collect_dqy.flatten()
    # collect_dqz = collect_dqz.flatten()

    rotation_max = np.max(ROTATION_NUMBERS)
    size_per_rotation = collect_qx.shape[0]

    q_vector_now = collect_q_vector
    collect_hw_mush = energy_transfer(qvector_per_rot=q_vector_now, calc_term=TERM_MUSHROOM)
    # collect_hw_mag = energy_transfer(qvector_per_rot=q_vector_now,calc_term=TERM_MUSHROOM)

    # calculate all the data for the maximal times of rotation
    if rotation_max > 0:
        for r in range(1, rotation_max + 1):
            q_vector_now = np.apply_along_axis(geo.rotation_3d, axis=0, arr=q_vector_now, rot_axis=SAMPLE_INIT_ROT,
                                               angle=-ROTATION_STEP)
            collect_q_vector = np.append(collect_q_vector, q_vector_now, axis=1)
            collect_hw_mush = np.append(collect_hw_mush,
                                        energy_transfer(qvector_per_rot=q_vector_now, calc_term=TERM_MUSHROOM))
    for rotation in ROTATION_NUMBERS:
        # take the respective data from the data collection for each rotation times
        size = size_per_rotation * (rotation + 1)
        data_q_vector = collect_q_vector[:, :size + 1]
        # data_dqx = collect_dqx[:size + 1]
        # data_dqy = collect_dqy[:size + 1]
        # data_dqz = collect_dqz[:size + 1]
        data_hw_mush = collect_hw_mush[:size + 1]
        for dim in DIMENSIONS:
            dim_info = dim
            if dim in DIM_PROJ:
                pass
                # if dim == DIM_1:
                #     fig, ax = plt.subplots(constrained_layout=True)
                #     ax.scatter(plot_q(np.linalg.norm(data_q_vector, axis=0)), neutron.joule2mev(data_hw),
                #                c=neutron.joule2mev(data_hw), marker=".")
                #     ax.set_xlabel(q_label(NORM, q_unit))
                #     ax.set_ylabel(HW_LABEL)
                #     do_plot()
                # elif dim == DIM_2:
                #     fig, ax = plt.subplots(constrained_layout=True)
                #     plt.axis("equal")
                #     cnt = ax.scatter(plot_q(data_q_vector[:, 0]), plot_q(data_q_vector[:, 1]),
                #                      c=neutron.joule2mev(data_hw), marker=".")
                #     ax.set_xlabel(q_label(AXIS_X, q_unit))
                #     ax.set_ylabel(q_label(AXIS_Y, q_unit))
                #     cbar_scatt = fig.colorbar(cnt, ax=ax)
                #     cbar_scatt.set_label(HW_LABEL)
                #     do_plot()
                # elif dim == DIM_3:
                #     fig = plt.figure(constrained_layout=True)
                #     ax = fig.add_subplot(111, projection='3d')
                #     cnt = ax.scatter(plot_q(data_q_vector[:, 0]), plot_q(data_q_vector[:, 1]),
                #                      neutron.joule2mev(data_hw), c=neutron.joule2mev(data_hw), marker=".")
                #     ax.tick_params(axis="z", direction="in")
                #     ax.set_xlabel(q_label(AXIS_X, q_unit), labelpad=-3)
                #     ax.set_ylabel(q_label(AXIS_Y, q_unit), labelpad=-3)
                #     cbar_scatt = fig.colorbar(cnt, ax=ax, pad=0.1)
                #     cbar_scatt.set_label(HW_LABEL)
                # elif dim == DIM_4:
                #     fig = plt.figure(constrained_layout=True)
                #     ax = fig.add_subplot(111, projection='3d')
                #     cnt = ax.scatter(plot_q(data_q_vector[:, 0]), plot_q(data_q_vector[:, 1]),
                #                      plot_q(data_q_vector[:, 2]), c=neutron.joule2mev(data_hw), marker=".")
                #     # ax.set_xlim(plot_q(np.min(data_qx)) * 1.1, plot_q(np.max(data_qx)) * 1.1)
                #     # ax.set_ylim(plot_q(np.min(data_qy)) * 1.1, plot_q(np.max(data_qy)) * 1.1)
                #     ax.tick_params(axis="z", direction="in")
                #     ax.set_xlabel(q_label(AXIS_X, q_unit), labelpad=-3)
                #     ax.set_ylabel(q_label(AXIS_Y, q_unit), labelpad=-3)
                #     ax.set_zlabel(q_label(AXIS_Z, q_unit), labelpad=7)
                #     cbar_scatt = fig.colorbar(cnt, ax=ax, pad=0.1)
                #     cbar_scatt.set_label(HW_LABEL)
                # else:
                #     raise ValueError("Unknown dimension or the given dimension is not a simple version.")
            elif dim == DIM_LINE:
                # if term != TERM_MUSHROOM_MAGNON:
                #     raise ValueError("This plot works only for Mushroom measuring the dispersion")
                # index_finite_hw = np.isfinite(data_hw)
                # points_finite_hw = data_q_vector[:, index_finite_hw]
                for point_interest in POINTS_INTEREST:
                    for hkl_key in HKL_DICT:
                        hkl_value = HKL_DICT[hkl_key]
                        fig, ax = plt.subplots(constrained_layout=True)
                        # index_point, index_points = line_through_points(hkl_value, points_finite_hw)
                        index_points = line_through_points(hkl_value, data_q_vector, point=rlu2q(point_interest))
                        if np.count_nonzero(index_points) > 0:
                            points_near_line = data_q_vector[:, index_points]
                            mush_hw = data_hw_mush[index_points]
                            mush_xi = points2xi(point=rlu2q(point_interest), points=points_near_line, hkl=hkl_value)
                            indices_measure = np.isfinite(np.array(list(map(
                                lambda i: magnon_scattered(de_of_e=0.05, scattering_de=mush_hw[i],
                                                           magnon_de=magnonmdl.magnon_energy(
                                                               xi2points(point=point_interest, hkl=hkl_value,
                                                                         xi=mush_xi[i]))), range(mush_hw.shape[0])))))
                            measure_xi = mush_xi[indices_measure]
                            measure_hw = mush_hw[indices_measure]
                            max_distance = np.max(
                                np.apply_along_axis(func1d=geo.points_distance, axis=0, arr=points_near_line,
                                                    point2=rlu2q(point_interest)))
                            points_marks = np.linspace(-max_distance, max_distance, num=100)
                            theory_hw = np.array(list(map(lambda m: mark_energy(m, rlu2q(point_interest), hkl_value),
                                                          points_marks)))
                            if measure_hw.shape[0] > 0:
                                theory_hw *= np.average(measure_hw) / abs(np.average(measure_hw))
                            ax.plot(plot_q(points_marks), neutron.joule2mev(theory_hw), "blue")
                            ax.scatter(plot_q(mush_xi), neutron.joule2mev(mush_hw), c="red", marker="o")
                            ax.scatter(plot_q(measure_xi), neutron.joule2mev(measure_hw), c="lime", marker="*")
                            # ax.scatter(0, neutron.joule2mev(data_hw[index_finite_hw][index_point]), c="red", marker="1")
                            ax.set_xlabel(r"({:.1f}{:s}, {:.1f}{:s}, {:.1f}{:s}) ({:s})".format(point_interest[0],
                                                                                                hkl_plot(hkl_value[0]),
                                                                                                point_interest[1],
                                                                                                hkl_plot(hkl_value[1]),
                                                                                                point_interest[2],
                                                                                                hkl_plot(hkl_value[2]),
                                                                                                q_unit))
                            ax.set_ylabel(HW_LABEL)
                        else:
                            print("Direction {:s} does not have any points".format(hkl_key))
                            continue
                        dim_info = "[{:s}]({:.1f},{:.1f},{:.1f})".format(hkl_key, *point_interest)
                        do_plot(calc_term=TERM_MUSHROOM_MAGNON)
            elif dim == DIM_PLANE:
                index_finite_hw = np.isfinite(data_hw_mush)
                hw_hist_counts, hw_hist_bins = np.histogram(a=data_hw_mush[index_finite_hw], bins=np.sort(
                    neutron.energy_transfer(wavenumber_in=geo_ctx.wavenumber_in,
                                            wavenumber_out=geo_ctx.wavenumbers_out)))
                count_index = np.argmax(hw_hist_counts[:-1])
                layer_number = 20
                if count_index < hw_hist_counts.shape[0] - layer_number:
                    index_range = range(0, layer_number)
                else:
                    index_range = range(-layer_number, 1)
                for i in index_range:
                    index = count_index + i
                    hw1, hw2 = hw_hist_bins[index], hw_hist_bins[index + 1]
                    hw_index = np.where(np.bitwise_and(hw1 <= data_hw_mush, data_hw_mush < hw2))[0]
                    fig, ax = plt.subplots(constrained_layout=True)
                    plt.axis("equal")
                    ax.scatter(plot_q(data_q_vector[0, hw_index]), plot_q(data_q_vector[1, hw_index]),
                               c=neutron.joule2mev(data_hw_mush[hw_index]), marker=".")  # cnt
                    ax.set_xlim(plot_q(np.min(data_q_vector[0, :])), plot_q(np.max(data_q_vector[0, :])))
                    ax.set_ylim(plot_q(np.min(data_q_vector[1, :])), plot_q(np.max(data_q_vector[1, :])))
                    # cbar_scatt = fig.colorbar(cnt, ax=ax)
                    # cbar_scatt.set_label(HW_LABEL)
                    ax.set_xlabel(q_label(AXIS_X, q_unit))
                    ax.set_ylabel(q_label(AXIS_Y, q_unit))
                    ax.set_title(r"$\hbar\omega={:.2f}$ meV".format(neutron.joule2mev(hw1)))
                    dim_info = "hw{:.2f}".format(neutron.joule2mev(hw1))
                    do_plot(calc_term=TERM_MUSHROOM_MAGNON)
            else:
                raise ValueError("The given dimension is unknown {:s}.".format(dim))
                # if rotation == 0:
                #     ax.set_title("No sample rotation")
                # else:
                #     ax.set_title(r"Sample Rotation {:.0f}° x {:d}".format(np.rad2deg(ROTATION_STEP), rotation))

                # ax.set_title(r"{:s}, $\vec{{k}}_i$={:.1f} $\AA^{{-1}}$".format(calc_term, geo_ctx.wavenumber_in * 1e-10))


for ki in WAVENUMBERS_IN:
    geometryctx = MushroomContext(wavenumber_in=ki)
    magnonmdl = MagnonModel(model_name=MAGNON_DEFAULT, latt_const=4.5 * 1e-10, spin_coupling=neutron.mev2joule(0.3))
    # magnonmdl = MagnonModel(model_name=MAGNON_IRON, latt_const=2.859e-10, stiff_const=neutron.mev2joule(266 * 1e-20))

    # print("The index of the middle segment of the polar angle range: {:d}".format(int(geometryctx.pol_middle_index + 1)))

    # plot_geometry(geo_ctx=geometryctx)

    # plot_distance_fd_as(geo_ctx=geometryctx)

    # plot_analyser_comparison(geo_ctx=geometryctx)

    plot_resolution_kf(geo_ctx=geometryctx)

    # plot_resolution_polarangles(geo_ctx=geometryctx)
    # This might not be the best strategy to plot the resolution

    # write_mcstas(geo_ctx=geometryctx)

    # wavenumbers_psd(geo_ctx=geometryctx)

    # for term in CALC_TERMS:
    # dispersion_calc_plot(geo_ctx=geometryctx, extension=EXTENSION_PNG, q_unit=Q_UNIT_RLU)
