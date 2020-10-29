import matplotlib.pyplot as plt
import numpy as np
from geometry_context import GeometryContext
from helper import wavelength_to_eV, points_distance, vector_bisector, InstrumentContext, points_to_vector, \
    points_to_slope_radian, unit_vector, vector_project_a2b, deg2min, points_bisecting_line, line_to_y, \
    PLANCKS_CONSTANT, MASS_NEUTRON, CONVERSION_JOULE_PER_EV, data2range, dispersion_signal, rotation_around_z, \
    wavenumber_vector
from magnon import magnon_energy, scatt_cross_qxqyde, scatt_cross_kikf

plt.rcParams.update({'font.size': 12})

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082
"""

E_RESOL = 0.02  # energy resolution (relative uncertainty)
# Comment from Alex <3
# line: ax + by + c = 0 -> (a, b, c)
TERM_MAGNON = "magnon"
TERM_SCATTERING = "scattering"  # energy conservation at the scattering
TERM_CROSSSECTION = "crosssection"  # Delta function is approximated by Gaussian function with intensity distribution

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

ROTATION_STEP = np.deg2rad(1)  # sample rotation step size
ROTATION_NUMBER = 10 + 1  # number of steps, +1 because the Python range starts from 0

PLOT_NUMBER = 500


# def get_analyser_angular_spread(geo_ctx: GeometryContext, sample, analyser_point, focus_point):
def monochromator_angular_spread(divergence_in, divergence_out, mosaic):
    # For the formula see [Paper1]
    # alpha_i, alpha_f, eta = divergence_in, divergence_out, mosaic
    numerator = divergence_in ** 2 * divergence_out ** 2 + mosaic ** 2 * divergence_in ** 2 + \
                mosaic ** 2 * divergence_out ** 2
    denominator = 4 * mosaic ** 2 + divergence_in ** 2 + divergence_out ** 2
    return np.sqrt(numerator / denominator)


def angular_res_an(geo_ctx: GeometryContext, instrument: InstrumentContext, an_index):
    eta = instrument.moasic_analyser  # mosaic
    alpha_i, alpha_f = vertical_divergence_analyser(geo_ctx=geo_ctx, instrument=instrument,
                                                    analyser_index=an_index)  # incoming and outgoing divergence
    # For the formula see [Paper1]
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2

    return np.sqrt(numerator / denominator)


def uncert_kf(geo_ctx: GeometryContext, instrument: InstrumentContext, an_ind_now, an_ind_near):
    kf_now = geo_ctx.wavenumbers_out[an_ind_now]
    kf_near = geo_ctx.wavenumbers_out[an_ind_near]
    angular_uncertainty_analyser = monochromator_angular_spread(
        *vertical_divergence_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_index=an_ind_now),
        mosaic=instrument.moasic_analyser)
    twotheta_an = geo_ctx.an_2theta[an_ind_now]
    uncertainty_kf_bragg = kf_now * np.linalg.norm(
        [instrument.deltad_d, angular_uncertainty_analyser / np.tan(twotheta_an / 2.0)])  # from Bragg's law
    uncertainty_kf_segment = abs(kf_now - kf_near)
    # print(angular_uncertainty_analyser, np.rad2deg(twotheta_an), uncertainty_kf_bragg * 1e-10,
    #       uncertainty_kf_segment * 1e-10)
    return max(uncertainty_kf_bragg, uncertainty_kf_segment)


def uncert_pol(geo_ctx: GeometryContext, instrument: InstrumentContext, an_index):
    divergence = angular_res_an(geo_ctx=geo_ctx, instrument=instrument, an_index=an_index)
    # the uncertainty of the polar angle is given by the angular resolution at the analyser
    # return get_angular_resolution_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_point=analyser_point)
    return divergence


def uncert_azi(geo_ctx: GeometryContext, instrument: InstrumentContext, an_index):
    an_point = (geo_ctx.analyser_points[0][an_index], geo_ctx.analyser_points[1][an_index])
    # sa: sample-analyser; af: analyser-focus
    distance_sa = points_distance(point1=geo_ctx.sample_point, point2=an_point)
    distance_af = points_distance(point1=an_point, point2=geo_ctx.foc_point)
    uncert_azi_sa = 2.0 * np.arctan((instrument.an_seg + instrument.sample_diameter) / (2.0 * distance_sa))
    uncert_azi_af = 2.0 * np.arctan((instrument.an_seg + geo_ctx.foc_size) / (2.0 * distance_af))
    return min(uncert_azi_sa, uncert_azi_af)


def de_of_e_from_an(geo_ctx: GeometryContext, instrument: InstrumentContext, an_ind_now, an_ind_near):
    # factor_polar= spread_factor_detector(geo_ctx=geo_ctx, instrument=instrument,
    #                                                       analyser_now=analyser_point,
    #                                                       analyser_nearest=nearest_point, index_now=index)[0]
    kf_now = geo_ctx.wavenumbers_out[an_ind_now]
    delta_kf = uncert_kf(geo_ctx, instrument, an_ind_now=an_ind_now, an_ind_near=an_ind_near) * spread_factor_detector(
        geo_ctx=geo_ctx, instrument=instrument, index_now=an_ind_now, index_nearest=an_ind_near)[0]
    return 2. * delta_kf / kf_now


# def get_divergence(sample, analyser_point, focus, sample_size, focus_size):
def vertical_divergence_analyser(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_index):
    # sa: sample-analyser; af: analyser-focus
    analyser_point = (geo_ctx.analyser_points[0][analyser_index], geo_ctx.analyser_points[1][analyser_index])
    vector_sa = points_to_vector(point1=geo_ctx.sample_point, point2=analyser_point)
    vector_af = points_to_vector(point1=analyser_point, point2=geo_ctx.foc_point)
    vector_tangential = vector_bisector(vector_sa, vector_af)
    segment_analyser = unit_vector(vector_tangential) * instrument.an_seg
    analyser_incoming_projection = vector_project_a2b(segment_analyser, vector_sa)
    analyser_incoming_rejection = segment_analyser - analyser_incoming_projection
    analyser_outgoing_projection = vector_project_a2b(segment_analyser, vector_af)
    analyser_outgoing_rejection = segment_analyser - analyser_outgoing_projection

    divergence_in = 2 * np.arctan((instrument.sample_height * abs(
        np.cos(points_to_slope_radian(point1=geo_ctx.sample_point, point2=analyser_point))) + np.linalg.norm(
        analyser_incoming_rejection)) / (2.0 * np.linalg.norm(vector_sa)))
    divergence_out = 2 * np.arctan((geo_ctx.foc_size * abs(
        np.sin(points_to_slope_radian(point1=analyser_point, point2=geo_ctx.foc_point))) + np.linalg.norm(
        analyser_outgoing_rejection)) / (2.0 * np.linalg.norm(vector_af)))
    # divergence_in = instrument.sample_size / distance_sa
    # divergence_out = geo_ctx.focus_size / distance_af
    # print(divergence_in, divergence_out)
    return divergence_in, divergence_out


def kf_resol_mcstas(geo_ctx: GeometryContext, instrument: InstrumentContext, index_now, index_nearest):
    factor_polar, factor_azimuth = spread_factor_detector(geo_ctx=geo_ctx, instrument=instrument, index_now=index_now,
                                                          index_nearest=index_nearest)
    # factor_polar, factor_azimuth = 1, 1
    dkf = uncert_kf(geo_ctx, instrument=instrument, an_ind_now=index_now,
                    an_ind_near=index_nearest) * factor_polar
    dphi = uncert_pol(geo_ctx, instrument=instrument, an_index=index_now) * factor_polar
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta = uncert_azi(geo_ctx=geo_ctx, instrument=instrument,
                        an_index=index_now) * factor_azimuth
    kf = geo_ctx.wavenumbers_out[index_now]
    dkf_x = kf * np.tan(dtheta)
    dkf_y = kf * np.tan(dphi)
    dkf_z = dkf
    return [dkf_x, dkf_y, dkf_z]


# to compare the analyser generated by the two different methods
def plot_analyser_comparison(points_x, points_y, points_analyser_x, points_analyser_y):
    plt.figure(10)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(points_x, points_y)
    plt.plot(points_analyser_x, points_analyser_y)
    plt.legend((r"Segments with 1x1 cm$^2$", "Ideal ellipse"))
    plt.text(0.3, -0.3, "Number of segments in one cut-plane: {:d}".format(len(points_x)))
    plt.text(0.3, -0.35, "Largest deviation from the ideal ellipse: {:5.2f} m".format(
        points_distance([points_x[-1], points_y[-1]], [points_analyser_x[-1], points_analyser_y[-1]])))
    plt.xlabel("x axis (m)")
    plt.ylabel("y axis (m)")
    plt.plot(*geometryctx.sample_point, "ro")
    plt.plot(*geometryctx.foc_point, "ro")
    plt.text(x=0, y=-0.05, s="Sample")
    plt.text(x=0.1, y=-0.4, s="Focus")

    plt.plot([geometryctx.sample_point[0], geometryctx.foc_point[0]],
             [geometryctx.sample_point[1], geometryctx.foc_point[1]])
    bisecting_x = np.array([0.75, 1])
    plt.plot(bisecting_x, line_to_y(bisecting_x, points_bisecting_line(point1=geometryctx.sample_point,
                                                                       point2=geometryctx.foc_point)))
    plt.axis("equal")

    plt.savefig("Geometry_Comparison.pdf", bbox_inches='tight')
    plt.close(10)


def plot_geometry(geo_ctx: GeometryContext, instrument: InstrumentContext):
    # to plot the geometry with important parameters
    def plot_for_analyser_point(geo_ctx: GeometryContext, instrument: InstrumentContext, index_now, index_nearest):
        energy_ev = wavelength_to_eV(
            wavelength=geo_ctx.wavelength_bragg(instrument=instrument, index=index_now))
        e_res_ev = de_of_e_from_an(geo_ctx=geo_ctx, instrument=instrument, an_ind_now=index_now,
                                   an_ind_near=index_nearest)
        e_res_ev *= energy_ev
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
        plt.text(x=-analyser_point[0] * 1.1 - 0.5, y=analyser_point[1] * 1.05 + 0.05,
                 s="{:5.2f}".format(energy_ev * 1e3))
        plt.text(x=analyser_point[0] * 1.1, y=analyser_point[1] * 1.05 + 0.05,
                 s="{:5.2f}".format(e_res_ev * 1e6))

    plt.rcParams.update({'font.size': 12})

    # first plot the analyser on both sides
    plt.plot(geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.plot(-geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.xlabel("Radial axis (m)")
    plt.ylabel("Vertical axis (m)")

    plt.text(x=-2.5, y=0.5, s=r"$E$(meV)")
    plt.text(x=2, y=0.5, s=r"$\hbar\omega$($\mu$eV)")

    plt.text(-3, -3.1, "Wavenumber covered by the analyser " + r"$k_f \in$ [{:.2f}, {:.2f}]".format(
        np.min(geometryctx.wavenumbers_out) * 1e-10, np.max(geometryctx.wavenumbers_out) * 1e-10) + r" $\AA^{-1}$")

    plot_for_analyser_point(geo_ctx=geo_ctx, instrument=instrument, index_now=0, index_nearest=1)
    plot_for_analyser_point(geo_ctx=geo_ctx, instrument=instrument, index_now=-1, index_nearest=-2)

    index_largest_energy = np.argmax(geo_ctx.wavenumbers_out)
    plot_for_analyser_point(geo_ctx=geo_ctx, instrument=instrument, index_now=index_largest_energy,
                            index_nearest=index_largest_energy + 1)

    # mark the position of the sample and focus, and plot the detector
    plt.plot(*geo_ctx.sample_point, "ro")
    plt.text(x=-0.065, y=-1, s="Sample", rotation=90)
    plt.plot(*geo_ctx.foc_point, "ro", alpha=0.5)
    plt.text(x=geo_ctx.foc_point[0] - 0.8, y=geo_ctx.foc_point[1] - 0.1, s="Focus")
    plt.plot(*geo_ctx.detector_points, '.', color='#8c564b')
    plt.plot(-geo_ctx.detector_points[0], geo_ctx.detector_points[1], '.', color='#8c564b')

    # plt.xlim(-1.8, 1.8)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(geo_ctx.filename_geo + '.pdf', bbox_inches='tight')
    plt.savefig(geo_ctx.filename_geo + '.png', bbox_inches='tight')
    plt.close(1)
    print("{:s} plotted.".format(geo_ctx.filename_geo))


def write_mcstas(geo_ctx: GeometryContext, instrument: InstrumentContext):
    # to write the file giving the information of the analyser array for the McStas simulation
    f = open(geo_ctx.filename_mcstas, 'w+')
    value_width_z = instrument.an_seg
    value_height_y = instrument.an_seg
    value_mosaic_horizontal = deg2min(np.rad2deg(instrument.moasic_analyser))
    value_mosaic_vertical = deg2min(np.rad2deg(instrument.moasic_analyser))
    value_lattice_distance = instrument.lattice_distance_pg002 * 1e10  # it is in angstrom for McStas
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
    pass


def plot_resolution_polarangles(geo_ctx: GeometryContext, polar_angles, all_dqx_m, all_dqy_m, all_dqz_m, all_kf):
    # to plot the resolution with the x-axis being the polar angle
    # is more realistic since the same kf-value can originate from two angles
    plt.rcParams.update({'font.size': 12})
    polar_angles = np.rad2deg(polar_angles)

    def forward(x):
        return np.interp(x, polar_angles, all_kf * 1e-10)

    def inverse(x):
        kf_sort_indeces = np.argsort(all_kf)
        kf_sort = all_kf[kf_sort_indeces] * 1e-10
        polar_angles_sort = polar_angles[kf_sort_indeces]
        return np.interp(x, kf_sort, polar_angles_sort)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(polar_angles, all_dqx_m * 1e-10, color="blue")
    ax.plot(polar_angles, all_dqy_m * 1e-10, color="red")
    ax.plot(polar_angles, all_dqz_m * 1e-10, color="gold")
    ax.set_xlabel(r"Polar angle $\varphi$ (degree)")
    ax.set_ylabel(r"$\Delta k_f$ (angstrom$^{-1}$)")
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
    secax.set_xlabel(r' Outgoing wavenumber $k_f$ (angstrom$^{-1}$)')
    secax.tick_params(axis="x", direction="in", labelsize=10)

    ax.set_title('Q-resolution of the secondary spectrometer')
    filename = geo_ctx.filename_geo
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.savefig(filename + '.png', bbox_inches='tight')
    print("{:s} plotted.".format(filename))


def plot_resolution_kf(geo_ctx: GeometryContext, all_dqx_m, all_dqy_m, all_dqz_m):
    # to plot the resolution with the x-axis being the outgoing wave-number
    all_kf = geo_ctx.wavenumbers_out
    plt.rcParams.update({'font.size': 12})
    max_index = np.argmax(all_kf)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(all_kf[max_index:] * 1e-10, all_dqx_m[max_index:] * 1e-10, color="blue")
    ax.plot(all_kf[max_index:] * 1e-10, all_dqy_m[max_index:] * 1e-10, color="red")
    ax.plot(all_kf[max_index:] * 1e-10, all_dqz_m[max_index:] * 1e-10, color="gold")
    ax.set_xlabel(r'Outgoing wavenumber $|k_f|$ ($\AA^{-1}$)')
    ax.set_ylabel(r"Component uncertainties $\Delta k_{f,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    ax.grid()
    ax.legend(("x: horizontal", "y: vertical", r"z: along $k_f$"))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    dkf_percent = all_dqz_m / all_kf * 1e2
    ax2.plot(all_kf[max_index:] * 1e-10, dkf_percent[max_index:], '1', color=colour_ax2)
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"Relative uncertainty $\dfrac{|\Delta k_f|}{|k_f|}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)
    ax2.legend(["Relative uncertainty"], loc='lower left', bbox_to_anchor=(0, 0.65))
    ax.set_title('Resolution of the secondary spectrometer')
    filename = geo_ctx.filename_res
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.savefig(filename + '.png', bbox_inches='tight')
    print("{:s} plotted.".format(filename))


def resolution_calculation(geo_ctx: GeometryContext, instrument: InstrumentContext):
    all_dqx_m = np.array(list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, instrument=instrument, index_now=i,
                                                            index_nearest=1 if i == 0 else i - 1)[0],
                                  range(geo_ctx.pol_angles.shape[0]))))
    all_dqy_m = np.array(list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, instrument=instrument, index_now=i,
                                                            index_nearest=1 if i == 0 else i - 1)[1],
                                  range(geo_ctx.pol_angles.shape[0]))))
    all_dqz_m = np.array(list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, instrument=instrument, index_now=i,
                                                            index_nearest=1 if i == 0 else i - 1)[2],
                                  range(geo_ctx.pol_angles.shape[0]))))
    return all_dqx_m, all_dqy_m, all_dqz_m


def spread_factor_detector(geo_ctx: GeometryContext, instrument: InstrumentContext, index_now, index_nearest):
    an_point = (geo_ctx.analyser_points[0][index_now], geo_ctx.analyser_points[1][index_now])
    detector_now = [geo_ctx.detector_points[0][index_now], geo_ctx.detector_points[1][index_now]]
    detector_next = [geo_ctx.detector_points[0][index_nearest], geo_ctx.detector_points[1][index_nearest]]
    spread_factor_polar = max(1, instrument.detector_resolution / points_distance(detector_now, detector_next))

    detector_spread_azi = instrument.an_seg * points_distance(detector_now, geo_ctx.foc_point) / points_distance(
        an_point, geo_ctx.foc_point)
    spread_factor_azi = max(1, instrument.detector_resolution / detector_spread_azi)
    return spread_factor_polar, spread_factor_azi


def distances_fd_as(geo_ctx: GeometryContext, index):
    distance_fd = points_distance(point1=geo_ctx.foc_point,
                                  point2=[geo_ctx.detector_points[0][index], geo_ctx.detector_points[1][index]])
    distance_as = points_distance(point1=geo_ctx.foc_point,
                                  point2=[geo_ctx.analyser_points[0][index], geo_ctx.analyser_points[1][index]])
    return distance_fd / distance_as


def plot_distance_fd_as(geo_ctx: GeometryContext):
    ratio_fd_af = np.array(
        list(map(lambda i: distances_fd_as(geo_ctx=geo_ctx, index=i), range(geometryctx.pol_angles.shape[0]))))
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(geometryctx.pol_angles), ratio_fd_af)
    # plotting_format(ax=ax, grid=True)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    # if grid is True:
    ax.grid()
    ax.set_xlabel("Polar angle of analyser segment (degree)")
    ax.set_ylabel(r"Ratio $\frac{D_{FD}}{D_{AF}}$")
    ax.set_title("Distance-ratio focus-detector / analyser-focus")
    plt.tight_layout()
    plt.savefig("Distance_fd_af.png")
    plt.close(fig)


def wavenumbers_psd(geo_ctx: GeometryContext):
    kf = geo_ctx.wavenumbers_out
    detec_hori_x = geo_ctx.dete_hori_x
    detec_vert_y = geo_ctx.dete_vert_y
    fig, axs = plt.subplots(1, 2, sharey="all")
    axs[0].plot(detec_hori_x, kf[:detec_hori_x.shape[0]] * 1e-10)
    axs[1].plot(detec_vert_y, kf[detec_hori_x.shape[0]:] * 1e-10)
    axs[0].set_xlabel("Radial position of flat PSD (m)")
    axs[1].set_xlabel("Vertical position of cyl. PSDs (m)")
    axs[0].set_ylabel(r"Theoretical values of $k_f$ ($\AA^{-1}$)")
    axs[0].grid()
    axs[1].grid()
    fig.suptitle(r"Outgoing wavenumbers - PSD positions")
    plt.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.savefig("kf_values_psd_positions.pdf")
    plt.close(fig)


geometryctx = GeometryContext()
instrumentctx = InstrumentContext()


# print("The index of the segment in the middle of the polar angle range: {:d}".format(int(geometryctx.pol_middle_index + 1)))

# plot_geometry(geo_ctx=geometryctx, instrument=instrumentctx)

# plot_distance_fd_as(geo_ctx=geometryctx)

# plot_analyser_comparison(points_analyser_x=geometryctx.analyser_ellipse_points[0],
#                          points_analyser_y=geometryctx.analyser_ellipse_points[1],
#                          points_x=geometryctx.analyser_points[0], points_y=geometryctx.analyser_points[1])


# all_dqx_m, all_dqy_m, all_dqz_m = resolution_calculation(geo_ctx=geometryctx, instrument=instrumentctx)
# plot_resolution_kf(geo_ctx=geometryctx, all_dqx_m=all_dqx_m, all_dqy_m=all_dqy_m, all_dqz_m=all_dqz_m)
# plot_resolution_polarangles(geo_ctx=geometryctx, polar_angles=polar_angles, all_dqx_m=all_dqx_m, all_dqy_m=all_dqy_m,
#                             all_dqz_m=all_dqz_m, all_kf=all_kf)
#
# write_mcstas(geo_ctx=geometryctx, instrument=instrumentctx)

# wavenumbers_psd(geo_ctx=geometryctx)


# all_qx = np.array(list(map(lambda j: np.array(list(
#     map(lambda i: geometryctx.vector_transfer(index_pol=i, index_azi=j)[0], range(geometryctx.wavenumbers.shape[0])))),
#                            range(geometryctx.azi_angles.shape[0]))))*1e-10
# all_qy = np.array(list(map(lambda j: np.array(list(
#     map(lambda i: geometryctx.vector_transfer(index_pol=i, index_azi=j)[1], range(geometryctx.wavenumbers.shape[0])))),
#                            range(geometryctx.azi_angles.shape[0]))))*1e-10
# all_qz = np.array(list(map(lambda j: np.array(list(
#     map(lambda i: geometryctx.vector_transfer(index_pol=i, index_azi=j)[2], range(geometryctx.wavenumbers.shape[0])))),
#                            range(geometryctx.azi_angles.shape[0]))))*1e-10


def magnon_scattered(scattering_de, magnon_de):
    if abs(scattering_de - magnon_de) / abs(scattering_de) < E_RESOL:
        return scattering_de
    else:
        return np.nan


def wavevector_transfer_rotation(rot_angle, wavevector_transfer):
    new_qx, new_qy = rotation_around_z(rot_angle=rot_angle, old_x=wavevector_transfer[0], old_y=wavevector_transfer[1])
    wavevector_transfer[:2] = new_qx, new_qy
    return wavevector_transfer


# def rotation_calc(qx, qy, qz, rot_angle, term):
#     data_qx, data_qy = rotation_z(rot_angle, qx, qy)
#     data_qz = qz
#     magnon_de = np.array(list(map(lambda j: np.array(list(
#         map(lambda i: magnon_energy(wavevector_transfer=np.array([data_qx[j, i], data_qy[j, i], data_qz[j, i]])),
#             range(geometryctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0])))) * 1e3 / CONVERSION_JOULE_PER_EV
#     if term == TERM_MAGNON:
#         calculated = magnon_de
#     elif term == TERM_SCATTERING:
#         calculated = np.array(list(map(lambda j: np.array(list(
#             map(lambda i: magnon_scattered(scattering_de=data_de[j, i], magnon_de=magnon_de[j, i], tol=0.05),
#                 range(geometryctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))
#     elif term == TERM_CROSSSECTION:
#         calculated = np.array(list(map(lambda i: np.array(list(
#             map(lambda j: scatt_cross_qxqyde(qq_x=data_qx[i, j], qq_y=data_qy[i, j], hw=hw,
#                                              ki=geometryctx.wavenumber_in,
#                                              qq_z=data_qz[i, j], kf=geometryctx.wavenumbers_out[i], pol_angle=geo_ctx.polar_angles[i],
#                                              mushroom=True),
#                 range(geometryctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))
#     else:
#         raise RuntimeError("Invalid term given. Cannot calculate teh rotated signal.")
#
#     return data_qx, data_qy, calculated


def plot_range_qxqy(qx, qy, rot_step, rot_number):
    qx_min, qx_max, qy_min, qy_max = None, None, None, None
    for i in range(rot_number):
        qx, qy = rotation_around_z(rot_angle=rot_step, old_x=qx, old_y=qy)
        if i == 0:
            qx_min, qx_max = np.min(qx), np.max(qx)
            qy_min, qy_max = np.min(qy), np.max(qy)
        else:
            qx_min = min(qx_min, np.min(qx))
            qy_min = min(qy_min, np.min(qy))
            qx_max = max(qx_max, np.max(qx))
            qy_max = max(qy_max, np.max(qy))
    plot_qx = np.linspace(qx_min, qx_max, num=1000)
    plot_qy = np.linspace(qy_min, qy_max, num=1000)
    return plot_qx, plot_qy


def plot_etransfer(de_2d, energy_transfer):
    if isinstance(energy_transfer, float):
        de_2d = np.where(abs(np.where(de_2d, de_2d, 0) - energy_transfer) < 2e-2, de_2d, None)
    else:
        raise RuntimeError("Invalid energy transfer given")
    return de_2d


def mushroom_wavevector_transfer(geo_ctx: GeometryContext):
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


def mushroom_energy_transfer(geo_ctx: GeometryContext):
    return np.array(list(map(lambda i: np.array(list(map(
        lambda j: PLANCKS_CONSTANT ** 2 * (geo_ctx.wavenumber_in ** 2 - geo_ctx.wavenumbers_out[i] ** 2) / (
                2 * MASS_NEUTRON), range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.pol_angles.shape[0]))))


def magnon_given_energy(geo_ctx: GeometryContext, hw=1.04e-3 * CONVERSION_JOULE_PER_EV):
    """
    calcute the magnon dispersion available in Mushroom at a given energy transfer.
    x-axis: Qx, y-axis: Qy, colour map: intensity (scattering cross-section.
    :param geo_ctx: geometrical information of Mushroom
    :param hw: given value of the energy transfer
    :return: plotted figure saved
    """
    ki_vector = np.array([geo_ctx.wavenumber_in, 0, 0])

    data_qx, data_qy, data_qz = mushroom_wavevector_transfer(geo_ctx=geo_ctx)
    data_de = mushroom_energy_transfer(geo_ctx=geo_ctx)
    if np.min(abs((hw - data_de) / hw)) > E_RESOL:
        hw_alternative = data_de[np.argmin(abs(hw - data_de))]
        raise RuntimeError("Given energy transfer is not available, the closest value is {:.2f} meV".format(
            hw_alternative / CONVERSION_JOULE_PER_EV * 1e3))
    data_crosssection = np.array(list(map(lambda i: np.array(list(
        map(lambda j: scatt_cross_qxqyde(qq_x=data_qx[i, j], qq_y=data_qy[i, j], hw=hw, ki=geo_ctx.wavenumber_in,
                                         qq_z=data_qz[i, j], kf=geo_ctx.wavenumbers_out[i],
                                         pol_angle=geo_ctx.pol_angles[i],
                                         mushroom=True),
            range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.pol_angles.shape[0]))))

    plot_qz = data2range(data_qz)
    # for rot in range(rotation_number):
    rotation_step = np.deg2rad(10)
    rotation_number = 10 + 1
    # data_qx_rot = np.array(list(map(lambda i: np.array(list(
    #     map(lambda j: rotation_qxqy(rotation_step, data_qx[i, j], data_qy[i, j])[0],
    #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))
    # data_qy_rot = np.array(list(map(lambda i: np.array(list(
    #     map(lambda j: rotation_qxqy(rotation_step, data_qx[i, j], data_qy[i, j])[1],
    #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))

    # data_qx, data_qy = rotation_z(rotation_step, data_qx, data_qy)
    # plot_qx = data2range(data_qx)
    # plot_qy = data2range(data_qy)
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    plot_qx, plot_qy = plot_range_qxqy(qx=data_qx, qy=data_qy, rot_step=rotation_step, rot_number=rotation_number)
    qx_2d, qy_2d = np.meshgrid(plot_qx, plot_qy)

    for i in range(rotation_number):
        # data_qx, data_qy, magnon_de, scatt_de = magnon_calc(qx=data_qx, qy=data_qy, qz=data_qz, rot_angle=rotation_step)
        # the energy transfers are already in the unit of meV
        # magnon_2d = dispersion_signal(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
        #                               intensity=magnon_de)
        # # scatt_de = np.array(list(map(lambda j: np.array(list(
        # #     map(lambda i: magnon_scattered(scattering_de=data_de[j, i], magnon_de=magnon_de[j, i], tol=0.05),
        # #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))
        # scatt_2d = dispersion_signal(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
        #                              intensity=scatt_de)
        data_qx, data_qy, data_crosssection = rotation_calc(qx=data_qx, qy=data_qy, qz=data_qz, rot_angle=rotation_step,
                                                            term=TERM_CROSSSECTION)
        crosssection_2d = dispersion_signal(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
                                            intensity=data_crosssection)
        # crosssection_2d /= np.max(crosssection_2d)
        # cnt_magnon = ax.scatter(x=qx_2d * 1e-10, y=qy_2d * 1e-10, c=magnon_2d, alpha=0.5)
        # # scatt_2d = plot_etransfer(scatt_2d, 1.0)
        # cnt_scatt = ax2.scatter(x=qx_2d * 1e-10, y=qy_2d * 1e-10, c=scatt_2d, alpha=0.5)
        cnt_crosssection = ax3.scatter(x=qx_2d * 1e-10, y=qy_2d * 1e-10, c=crosssection_2d, alpha=0.5)
    print("finished1")
    # cbar_magnon = fig.colorbar(cnt_magnon, ax=ax)
    # cbar_magnon.set_label(r"$\hbar\omega_{magnon}=D(\vec{k}_i-\vec{k}_f-\vec{\tau})^2$ (meV)")
    # cbar_scatt = fig2.colorbar(cnt_scatt, ax=ax2)
    # cbar_scatt.set_label(r"$\hbar\omega$ (meV)")
    cbar_scatt = fig3.colorbar(cnt_crosssection, ax=ax3)
    print("finished2")
    cbar_scatt.set_label(r"Intensity")  # normalised to 1
    # # magnon_de = np.array(list(map(lambda j: np.array(list(
    # #     map(lambda i: magnon_energy(wavevector_transfer=np.array([data_qx[j, i], data_qy[j, i], data_qz[j, i]])),
    # #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0])))) * 1e3 / CONVERSION_JOULE_PER_EV
    #
    # # plot magnon from Mushroom Q-values without considering the energy conservation
    # # peak_2d = dispersion2intensity(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
    # #                                intensity=magnon_de)
    # # cbar.set_label(r"$\hbar\omega = \frac{\hbar^2}{2m}(k_i^2-k_f^2)$ (meV)")
    # ax.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
    # ax.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    # ax.tick_params(axis="x", direction="in")
    # ax.tick_params(axis="y", direction="in")
    # ax.set_title(r"Magnon dispersion with $Q_x$ and $Q_y$ values in Mushroom")
    # ax.axis("equal")
    # fig.tight_layout()
    # fig.savefig("MagnonMushroom_QxQy_Rotated_{:d}_{:d}.png".format(rotation_number, int(np.rad2deg(rotation_step))))
    # plt.close(fig)
    #
    # # scatt_de = np.array(list(map(lambda j: np.array(list(
    # #     map(lambda i: magnon_scattered(scattering_de=data_de[j, i], magnon_de=magnon_de[j, i], tol=0.05),
    # #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))
    # # peak_2d = dispersion2intensity(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
    # #                                intensity=scatt_de)
    # ax2.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
    # ax2.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    # ax2.tick_params(axis="x", direction="in")
    # ax2.tick_params(axis="y", direction="in")
    # ax2.set_title(r"$|\frac{\hbar\omega_{MU}-\hbar\omega_{MA}}{\hbar\omega_{MU}}|<0.05$")
    # ax2.axis("equal")
    # fig2.tight_layout()
    # fig2.savefig("MagnonMushroom_QxQydE0.05_Rotated_{:d}_{:d}.png".format(rotation_number, int(np.rad2deg(rotation_step))))
    # plt.close(fig2)

    ax3.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
    ax3.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    ax3.tick_params(axis="x", direction="in")
    ax3.tick_params(axis="y", direction="in")
    ax3.set_title(r"$\hbar\omega=$" + "{:.2f} meV".format(hw / CONVERSION_JOULE_PER_EV * 1e3))
    ax3.axis("equal")
    fig3.tight_layout()
    fig3.savefig("CrosssectionMushroom_QxQydE_{:.2f}meV.png".format(hw / CONVERSION_JOULE_PER_EV * 1e3))
    plt.close(fig3)

    # magnon dispersion with energy conservation
    # fig, ax1 = plt.subplots()
    # scatt_de = np.array(list(map(lambda j: np.array(list(
    #     map(lambda i: magnon_scattered(scattering_de=data_de[j, i], magnon_de=magnon_de[j, i], tol=0.01),
    #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.polar_angles.shape[0]))))
    # peak_2d = dispersion2intensity(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
    #                                intensity=scatt_de)
    # cnt = ax1.scatter(x=qx_2d * 1e-10, y=qy_2d * 1e-10, c=peak_2d)
    # cbar = fig.colorbar(cnt, ax=ax1)
    # cbar.set_label(r"$\hbar\omega$ (meV)")
    # ax1.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
    # ax1.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    # ax1.tick_params(axis="x", direction="in")
    # ax1.tick_params(axis="y", direction="in")
    # ax1.set_title(r"$|\frac{\hbar\omega_{MU}-\hbar\omega_{MA}}{\hbar\omega_{MU}}|<0.01$")
    # ax1.axis("equal")
    # fig.tight_layout()
    # plt.savefig("MagnonMushroom_QxQydE_0.01.png")
    # plt.close(fig)

    # fig, ax = plt.subplots()
    # qx_2d, qy_2d = np.meshgrid(plot_qx, plot_qy)
    # peak_2d = dispersion2intensity(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
    #                                intensity=data_de)
    # cnt = ax.scatter(x=qx_2d * 1e-10, y=qy_2d * 1e-10, c=peak_2d)
    # cbar = fig.colorbar(cnt, ax=ax)
    # cbar.set_label(r"$\hbar\omega = \frac{\hbar^2}{2m}(k_i^2-k_f^2)$ (meV)")
    # ax.set_xlabel(r"$Q_z=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
    # ax.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    # ax.tick_params(axis="x", direction="in")
    # ax.tick_params(axis="y", direction="in")
    # ax.set_title(r"Energy transfer $\hbar\omega$ projected on ($Q_x$, $Q_y$) plane")
    # ax.axis("equal")
    # plt.tight_layout()
    # plt.savefig("QxQy-dE.png")
    # plt.close(fig)

    # fig, ax = plt.subplots()
    # qy_2d, qz_2d = np.meshgrid(plot_qy, plot_qz)
    # peak_2d = dispersion2intensity(range_x=plot_qy, range_y=plot_qz, data_x=data_qy, data_y=data_qz,
    #                                intensity=data_de)
    # cnt = ax.scatter(x=qy_2d * 1e-10, y=qz_2d * 1e-10, c=peak_2d)
    # cbar = fig.colorbar(cnt, ax=ax)
    # cbar.set_label(r"$\hbar\omega = \frac{\hbar^2}{2m}(k_i^2-k_f^2)$ (meV)")
    # ax.set_xlabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    # ax.set_ylabel(r"$Q_z=k_{i,z}-k_{f,z}$ ($\AA^{-1}$)")
    # ax.tick_params(axis="x", direction="in")
    # ax.tick_params(axis="y", direction="in")
    # ax.set_title(r"Energy transfer $\hbar\omega$ projected on ($Q_y$, $Q_z$) plane")
    # ax.axis("equal")
    # plt.tight_layout()
    # plt.savefig("QyQz-dE.png")
    # plt.close(fig)

    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey="all")
    # ax1.scatter(data_qz * 1e-10, data_de)
    # ax1.set_xlabel(r"$Q_z=k_{i,z}-k_{f,z}$ ($\AA^{-1}$)")
    # ax1.set_ylabel(r"$\hbar\omega = \frac{\hbar^2}{2m}(k_i^2-k_f^2)$ (meV)")
    # ax1.tick_params(axis="x", direction="in")
    # ax1.tick_params(axis="y", direction="in")
    # ax1.grid()
    #
    # ax2.scatter(np.linalg.norm([data_qx, data_qy], axis=0) * 1e-10, data_de)
    # ax2.set_xlabel(r"$Q_{xy}=\sqrt{Q_x^2+Q_y^2}$ ($\AA^{-1}$)")
    # # ax2.set_ylabel(r"$\hbar\omega = \frac{\hbar^2}{2m}(k_i^2-k_f^2)$ (meV)")
    # ax2.tick_params(axis="x", direction="in")
    # ax2.tick_params(axis="y", direction="in")
    # ax2.grid()
    # fig.suptitle("Energy and wavevector transfer")
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig("Qxy-Qz-dE.png")
    # plt.close(fig)


def rotation_axis_range(qx, qy, rot_step, rot_number, axis=None):
    range_min, range_max = None, None
    if axis:
        for i in range(rot_number):
            qx, qy = rotation_around_z(rot_angle=rot_step, old_x=qx, old_y=qy)
            if axis == AXIS_X:
                data = qx
            elif axis == AXIS_Y:
                data = qy
            else:
                raise RuntimeError("Wrong axis given.")
            if i == 0:
                range_min, range_max = np.min(data), np.max(data)
            else:
                range_min = min(range_min, np.min(data))
                range_max = max(range_max, np.max(data))
        plot_range = np.linspace(range_min, range_max, num=PLOT_NUMBER)
        return plot_range
    else:
        qx_min, qx_max = np.min(qx), np.max(qx)
        qy_min, qy_max = np.min(qy), np.max(qy)
        for i in range(1, rot_number):
            qx, qy = rotation_around_z(rot_angle=rot_step, old_x=qx, old_y=qy)
            qx_min = min(qx_min, np.min(qx))
            qy_min = min(qy_min, np.min(qy))
            qx_max = max(qx_max, np.max(qx))
            qy_max = max(qy_max, np.max(qy))
        plot_qx = np.linspace(qx_min, qx_max, num=PLOT_NUMBER)
        plot_qy = np.linspace(qy_min, qy_max, num=PLOT_NUMBER)
        return plot_qx, plot_qy


def mushroom_magnon_crosssection(geo_ctx: GeometryContext):
    ki_vector = np.array([geo_ctx.wavenumber_in, 0, 0])
    return np.array(list(map(lambda i: np.array(list(map(lambda j: scatt_cross_kikf(ki_vector=ki_vector,
                                                                                    kf_vector=wavenumber_vector(
                                                                                        wavenumber=
                                                                                        geo_ctx.wavenumbers_out[i],
                                                                                        azi_angle=geo_ctx.azi_angles[j],
                                                                                        pol_angle=geo_ctx.pol_angles[
                                                                                            i])),
                                                         range(geo_ctx.azi_angles.shape[0])))),
                             range(geo_ctx.pol_angles.shape[0]))))


def magnon_dispersion_Qihw(geo_ctx: GeometryContext, axis):
    data_qx, data_qy, data_qz = mushroom_wavevector_transfer(geo_ctx=geo_ctx)
    data_de = mushroom_energy_transfer(geo_ctx=geo_ctx)
    data_crosssection = mushroom_magnon_crosssection(geo_ctx=geo_ctx)

    data_y = data_de
    plot_y = np.linspace(np.min(data_y), np.max(data_y), num=PLOT_NUMBER)
    if axis in AXES:
        plot_x = rotation_axis_range(qx=data_qx, qy=data_qy, axis=axis, rot_step=ROTATION_STEP,
                                     rot_number=ROTATION_NUMBER)
    else:
        raise RuntimeError("Wrong axis given.")
    plot_x2d, plot_y2d = np.meshgrid(plot_x, plot_y)
    fig, ax = plt.subplots()

    if axis == AXIS_X:
        data_x = data_qx
    elif axis == AXIS_Y:
        data_x = data_qy
    else:
        data_x = data_qz
    crosssection_2d = dispersion_signal(range_x=plot_x, range_y=plot_y, data_x=data_x, data_y=data_y,
                                        intensity=data_crosssection)
    for i in range(ROTATION_NUMBER):
        data_qx, data_qy = rotation_around_z(rot_angle=ROTATION_STEP, old_x=data_qx, old_y=data_qy)
        if axis == AXIS_X:
            data_x = data_qx
        elif axis == AXIS_Y:
            data_x = data_qy
        else:
            data_x = data_qz
        crosssection_new = dispersion_signal(range_x=plot_x, range_y=plot_y, data_x=data_x, data_y=data_y,
                                             intensity=data_crosssection)
        crosssection_2d = np.where(np.isfinite(crosssection_2d),
                                   np.where(np.isfinite(crosssection_new), crosssection_2d + crosssection_new,
                                            crosssection_2d), crosssection_new)
    cnt_crosssection = ax.scatter(plot_x2d * 1e-10, plot_y2d * 1e3 / CONVERSION_JOULE_PER_EV, c=crosssection_2d)
    # cnt_crosssection = ax.contour(plot_x2d * 1e-10, plot_y2d * 1e3 / CONVERSION_JOULE_PER_EV, crosssection_2d)
    cbar_scatt = fig.colorbar(cnt_crosssection, ax=ax)
    cbar_scatt.set_label(r"Intensity")  # normalised to 1
    ax.set_xlabel(r"Wavevector transfer, {:s}-component $Q_{:s}=k_{{i,}}$".format(axis, axis) + r"$_{:s}$".format(
        axis) + r"$-k_{{f,}}$" + r"$_{:s}$".format(axis) + r"($\AA^{{-1}}$)")
    ax.set_ylabel(r"Energy transfer $\hbar\omega=E_{i}-E_{f}$ (meV)")
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_title(r"Magnon dispersion")
    fig.tight_layout()
    fig.savefig("MagnonMushroom_Q{:s}hw.pdf".format(axis))
    plt.close(fig)


def magnon_dispersion_Qijhw(geo_ctx: GeometryContext):
    data_qx, data_qy, data_qz = mushroom_wavevector_transfer(geo_ctx=geo_ctx)
    data_de = mushroom_energy_transfer(geo_ctx=geo_ctx)
    # data_crosssection = mushroom_magnon_crosssection(geo_ctx=geo_ctx)
    plot_x, plot_y = rotation_axis_range(qx=data_qx, qy=data_qy, rot_step=ROTATION_STEP, rot_number=ROTATION_NUMBER)
    plot_x2d, plot_y2d = np.meshgrid(plot_x, plot_y)
    magnon_de = np.array(list(map(lambda i: np.array(list(
        map(lambda j: magnon_energy(wavevector_transfer=np.array([data_qx[i, j], data_qy[i, j], data_qz[i, j]])),
            range(data_qx.shape[1])))), range(data_qx.shape[0]))))
    scattered_de = np.array(list(map(lambda i: np.array(list(
        map(lambda j: magnon_scattered(scattering_de=data_de[i, j], magnon_de=magnon_de[i, j]),
            range(data_qx.shape[1])))), range(data_qx.shape[0]))))

    fig, ax = plt.subplots()
    de_2d = dispersion_signal(range_x=plot_x, range_y=plot_y, data_x=data_qx, data_y=data_qy, intensity=scattered_de)
    for i in range(ROTATION_NUMBER - 1):
        data_qx, data_qy = rotation_around_z(rot_angle=ROTATION_STEP, old_x=data_qx, old_y=data_qy)
        de_new = dispersion_signal(range_x=plot_x, range_y=plot_y, data_x=data_qx, data_y=data_qy,
                                   intensity=scattered_de)
        de_2d = np.where(np.isfinite(de_2d), de_2d, de_new)
    cnt_crosssection = ax.scatter(plot_x2d * 1e-10, plot_y2d * 1e-10, c=de_2d * 1e3 / CONVERSION_JOULE_PER_EV)
    # cnt_crosssection = ax.contourf(plot_x2d * 1e-10, plot_y2d * 1e-10, de_2d * 1e3 / CONVERSION_JOULE_PER_EV)
    cbar_scatt = fig.colorbar(cnt_crosssection, ax=ax)
    cbar_scatt.set_label(r"Energy transfer $\hbar\omega=E_{i}-E_{f}$ (meV)")  # normalised to 1
    ax.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_title(r"Magnon dispersion")
    fig.tight_layout()
    fig.savefig("MagnonMushroom_QxQyhw.pdf")
    plt.close(fig)


# magnon_dispersion_Qihw(geometryctx, axis=AXIS_X)
# magnon_dispersion_Qihw(geometryctx, axis=AXIS_Y)
# magnon_dispersion(geometryctx, axis=AXIS_Z)
magnon_dispersion_Qijhw(geometryctx)
