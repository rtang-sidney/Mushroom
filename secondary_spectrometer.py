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
CALC_TERMS = [TERM_MAGNON, TERM_MUSHROOM_MAGNON]  # TERM_MUSHROOM,, TERM_MUSHROOM_MAGNON

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

ROTATION_STEP = np.deg2rad(1)  # sample rotation step size
# TODO: search the larges rotation times and calculate all data at once before selecting the respective parts to plot
ROTATION_NUMBERS = [10, 30, 90]  # number of steps, 0, 30, 90

DIM_1 = "1d"
DIM_2 = "2d"
DIM_3 = "3d"
DIMENSIONS = [DIM_1, DIM_2, DIM_3]  # DIM_1, DIM_2, DIM_3

EXTENSION_PDF = "pdf"
EXTENSION_PNG = "png"

MAGNON_MCSTAS = "McStas"
MAGNON_IRON = "Fe"  # Fe BCC

Q_UNIT_REAL = r"$\AA^{-1}$"  # the real unit of Q-vector
Q_UNIT_RLU = "r.l.u."  # reciprocal lattice unit


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
        plt.text(x=-analyser_point[0] * 1.1 - 0.85, y=analyser_point[1] * 1.05 + 0.05,
                 s="{:5.2f}".format(neutron.joule2mev(energy_out)))
        plt.text(x=analyser_point[0] * 1.05, y=analyser_point[1] * 1.05 + 0.05,
                 s="{:5.2f}".format(neutron.joule2mev(res_ef) * 1e3))

    # first plot the analyser on both sides
    plt.plot(geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.plot(-geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.xlabel("Radial axis (m)")
    plt.ylabel("Vertical axis (m)")

    plt.text(x=-2.3, y=0.4, s=r"$E$ (meV)")
    plt.text(x=1, y=0.4, s=r"$\Delta E$ ($\mu$eV)")

    plt.text(-2.9, -3.4,
             r"Wavenumber $k_f \in$ [{:.2f}, {:.2f}] $\AA^{{-1}}$".format(np.min(geometryctx.wavenumbers_out) * 1e-10,
                                                                          np.max(geometryctx.wavenumbers_out) * 1e-10))

    plot_for_analyser_point(index_now=0, index_nearest=1)
    plot_for_analyser_point(index_now=-1, index_nearest=-2)

    index_largest_energy = np.argmax(geo_ctx.wavenumbers_out)
    plot_for_analyser_point(index_now=index_largest_energy, index_nearest=index_largest_energy + 1)

    # mark the position of the sample and focus, and plot the detector
    plt.plot(*geo_ctx.sample_point, "ro")
    plt.text(x=-0.25, y=-1.5, s="Sample", rotation=90)
    plt.plot(*geo_ctx.foc_point, "ro", alpha=0.5)
    plt.text(x=geo_ctx.foc_point[0] - 1.25, y=geo_ctx.foc_point[1] - 0.1, s="Focus")
    plt.plot(*geo_ctx.detector_points, '.', color='#8c564b')
    plt.plot(-geo_ctx.detector_points[0], geo_ctx.detector_points[1], '.', color='#8c564b')

    plt.axis("equal")
    plt.tight_layout()
    plot_filename = geo_ctx.filename_geo + '.pdf'
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

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(all_kf[max_index:] * 1e-10, all_dqx_m[max_index:] * 1e-10, color="blue")
    ax.plot(all_kf[max_index:] * 1e-10, all_dqy_m[max_index:] * 1e-10, color="red")
    ax.plot(all_kf[max_index:] * 1e-10, all_dqz_m[max_index:] * 1e-10, color="gold")
    ax.set_xlabel(r'Wavenumber $k_f$ ($\AA^{-1}$)')
    ax.set_ylabel(r"Uncertainty $\Delta k_{f,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    ax.grid()
    ax.legend(("x: horizontal", "y: vertical", r"z: along $k_f$"))
    ax.tick_params(axis="both", direction="in")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    dkf_percent = all_dqz_m / all_kf * 1e2
    ax2.plot(all_kf[max_index:] * 1e-10, dkf_percent[max_index:], '1', color=colour_ax2)
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"$\dfrac{\Delta k_f}{k_f}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', color=colour_ax2, labelcolor=colour_ax2)
    ax2.legend(["Relative uncertainty"], loc='lower left', bbox_to_anchor=(0, 0.5),
               labelcolor=colour_ax2)
    # ax.set_title('Resolution, secondary spectrometer')
    filename = geo_ctx.filename_res
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.savefig(filename + '.png', bbox_inches='tight')
    print("Plot saved: {:s}".format(filename + '.pdf'))


def resolution_calculation(geo_ctx: MushroomContext):
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
    return mushroom_qx, mushroom_qy, mushroom_qz


def mushroom_energy_transfer(geo_ctx: MushroomContext):
    return np.array(list(map(lambda i: np.array(list(map(
        lambda j: neutron.planck_constant ** 2 * (geo_ctx.wavenumber_in ** 2 - geo_ctx.wavenumbers_out[i] ** 2) / (
                2 * neutron.mass_neutron), range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.pol_angles.shape[0]))))


def dispersion_calc_plot(geo_ctx: MushroomContext, rotation, calc_term, extension, q_unit):
    def energy_transfer(qx, qy, qz):
        nonlocal calc_term, geo_ctx
        if calc_term == TERM_MAGNON:
            magnon_hw = np.array(list(
                map(lambda i: magnonmdl.magnon_energy(wavevector_transfer=np.array([qx[i], qy[i], qz[i]])),
                    range(qx.shape[0]))))
            return magnon_hw
        elif calc_term == TERM_MUSHROOM:
            data_hw = mushroom_energy_transfer(geo_ctx=geo_ctx).flatten()
            return data_hw
        elif calc_term == TERM_MUSHROOM_MAGNON:
            magnon_hw = np.array(list(
                map(lambda i: magnonmdl.magnon_energy(wavevector_transfer=np.array([qx[i], qy[i], qz[i]])),
                    range(qx.shape[0]))))
            data_hw = mushroom_energy_transfer(geo_ctx=geo_ctx)
            rela_uncer_hw = np.array(list(map(lambda i: np.array(list(
                map(lambda j: de_of_e_from_an(geo_ctx=geo_ctx, an_ind_now=i, an_ind_near=1 if i == 0 else i - 1),
                    range(data_hw.shape[1])))), range(data_hw.shape[0]))))
            data_hw, rela_uncer_hw = data_hw.flatten(), rela_uncer_hw.flatten()
            scatter_hw = np.array(list(map(
                lambda i: magnon_scattered(scattering_de=data_hw[i], magnon_de=magnon_hw[i], de_of_e=rela_uncer_hw[i]),
                range(qx.shape[0]))))
            return scatter_hw
        else:
            raise ValueError("Invalid term to define the dispersion.")

    def plot_q(q_value):
        nonlocal q_unit
        if q_unit == Q_UNIT_REAL:
            return q_value * 1e-10
        else:
            return q_value / (2 * np.pi / magnonmdl.l_const)

    data_qx, data_qy, data_qz = mushroom_wavevector_transfer(geo_ctx=geo_ctx)
    data_qx = data_qx.flatten()
    data_qy = data_qy.flatten()
    data_qz = data_qz.flatten()

    qx_now, qy_now, qz_now = data_qx, data_qy, data_qz
    hw = energy_transfer(qx=qx_now, qy=qy_now, qz=qz_now)
    if rotation > 0:
        for r in range(1, rotation + 1):
            qx_now, qy_now = geo.rotation_around_z(rot_angle=ROTATION_STEP, old_x=qx_now, old_y=qy_now)
            data_qx = np.append(data_qx, qx_now)
            data_qy = np.append(data_qy, qy_now)
            data_qz = np.append(data_qz, qz_now)
            hw = np.append(hw, energy_transfer(qx=qx_now, qy=qy_now, qz=qz_now))

    for dim in DIMENSIONS:
        if dim == DIM_1:
            fig, ax = plt.subplots()
            ax.scatter(plot_q(np.linalg.norm([data_qx, data_qy, data_qz], axis=0)), neutron.joule2mev(hw),
                       c=neutron.joule2mev(hw), marker=".")
            ax.set_xlim(plot_q(np.min(data_qx)) * 1.1, plot_q(np.max(data_qx)) * 1.1)
            ax.set_xlabel(r"$|\vec{{Q}}|=|\vec{{k}}_{{i}}-\vec{{k}}_{{f}}|$ ({:s})".format(q_unit))
            ax.set_ylabel(r"$\hbar\omega=E_{i}-E_{f}$ (meV)")
        elif dim == DIM_2:
            fig, ax = plt.subplots()
            plt.axis("equal")
            cnt = ax.scatter(plot_q(data_qx), plot_q(data_qy), c=neutron.joule2mev(hw), marker=".")
            ax.set_xlim(plot_q(np.min(data_qx)) * 1.1, plot_q(np.max(data_qx)) * 1.1)
            ax.set_ylim(plot_q(np.min(data_qy)) * 1.1, plot_q(np.max(data_qy)) * 1.1)
            ax.set_xlabel(r"$Q_x=k_{{i,x}}-k_{{f,x}}$ ({:s})".format(q_unit))
            ax.set_ylabel(r"$Q_y=k_{{i,y}}-k_{{f,y}}$ ({:s})".format(q_unit))
            cbar_scatt = fig.colorbar(cnt, ax=ax)
            cbar_scatt.set_label(r"$\hbar\omega=E_{i}-E_{f}$ (meV)")
        elif dim == DIM_3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cnt = ax.scatter(plot_q(data_qx), plot_q(data_qy), neutron.joule2mev(hw), c=neutron.joule2mev(hw),
                             marker=".")
            ax.set_xlim(plot_q(np.min(data_qx)) * 1.1, plot_q(np.max(data_qx)) * 1.1)
            ax.set_ylim(plot_q(np.min(data_qy)) * 1.1, plot_q(np.max(data_qy)) * 1.1)
            ax.tick_params(axis="z", direction="in")
            ax.set_xlabel(r"$Q_x$ ({:s})".format(q_unit))
            ax.set_ylabel(r"$Q_y$ ({:s})".format(q_unit))
            cbar_scatt = fig.colorbar(cnt, ax=ax)
            cbar_scatt.set_label(r"$\hbar\omega$ (meV)")
            plt.tight_layout()
        else:
            raise ValueError("The given dimension is invalid for plotting.")
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        if rotation == 0:
            ax.set_title("No sample rotation")
        else:
            ax.set_title(r"Sample Rotation {:.0f}° x {:d}".format(np.rad2deg(ROTATION_STEP), rotation))
        if calc_term == TERM_MUSHROOM:
            filename = "{:s}_Rot{:d}_{:s}.{:s}".format(calc_term, rotation, dim, extension)
        else:
            filename = "{:s}_Rot{:d}_{:s}_{:s}.{:s}".format(calc_term, rotation, dim, magnonmdl.name, extension)
        # filename = PLOTPATH + filename
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print("Plot saved: {:s}".format(filename))


geometryctx = MushroomContext()
# magnonmdl = MagnonModel(model_name=MAGNON_MCSTAS, latt_const=4.5 * 1e-10, spin_coupling=neutron.mev2joule(0.3))
magnonmdl = MagnonModel(model_name=MAGNON_IRON, latt_const=2.859e-10, stiff_const=neutron.mev2joule(266 * 1e-20))

# print("The index of the middle segment of the polar angle range: {:d}".format(int(geometryctx.pol_middle_index + 1)))

# plot_geometry(geo_ctx=geometryctx)

# plot_distance_fd_as(geo_ctx=geometryctx)

# plot_analyser_comparison(geo_ctx=geometryctx)

# plot_resolution_kf(geo_ctx=geometryctx)

# plot_resolution_polarangles(geo_ctx=geometryctx)
# This might not be the best strategy to plot the resolution

# write_mcstas(geo_ctx=geometryctx)

# wavenumbers_psd(geo_ctx=geometryctx)

for rotation_number in ROTATION_NUMBERS:
    for term in CALC_TERMS:
        dispersion_calc_plot(geo_ctx=geometryctx, rotation=rotation_number, calc_term=term, extension=EXTENSION_PNG,
                             q_unit=Q_UNIT_RLU)
