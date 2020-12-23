import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
import geometry_calculation as geo
import neutron_context as neutron
import instrument_context as instr
import format_context as fmt

plt.rcParams.update({'font.size': 18})

"""
Abbreviations in the variable names
an: analyser
dete: detector
ind: index
diver: divergence

[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082
"""


# Comment from Alex <3
# line: ax + by + c = 0 -> (a, b, c)


# under Linux one needs / and Windows \\


def mono_angular_spread(divergence_in, divergence_out, mosaic):
    # For the formula see [Paper1]
    numerator = divergence_in ** 2 * divergence_out ** 2 + mosaic ** 2 * divergence_in ** 2 + \
                mosaic ** 2 * divergence_out ** 2
    denominator = 4 * mosaic ** 2 + divergence_in ** 2 + divergence_out ** 2
    return np.sqrt(numerator / denominator)


def angular_res_an(geo_ctx: MushroomContext, an_index):
    # eta, alpha_i, alpha_f = mosaic, divergence_in, divergence_out
    eta = instr.moasic_an
    alpha_i, alpha_f = an_diver_vert(geo_ctx=geo_ctx, an_ind=an_index)

    # For the formula see [Paper1]
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2
    return np.sqrt(numerator / denominator)


def uncert_kf(geo_ctx: MushroomContext, an_ind):
    kf_start = geo_ctx.wavenumbers_out[an_ind]
    orient_seg = geo_ctx.analyser_orients[:, an_ind]
    seg_end = geo_ctx.analyser_points[:, an_ind + 1]
    twotheta_end = 2 * geo.angle_vectors(vector1=geo.points2vector(point1=geo_ctx.sample_point, point2=seg_end),
                                         vector2=orient_seg)
    kf_end = neutron.bragg_angle2wavenumber(twotheta=twotheta_end, lattice_distance=instr.interplanar_pg002)
    angular_uncertainty_analyser = mono_angular_spread(*an_diver_vert(geo_ctx=geo_ctx, an_ind=an_ind),
                                                       mosaic=instr.moasic_an)
    twotheta_an = geo_ctx.an_2theta[an_ind]
    uncertainty_kf_bragg = kf_start * np.linalg.norm(
        [instr.deltad_d, angular_uncertainty_analyser / np.tan(twotheta_an / 2.0)])
    return uncertainty_kf_bragg


def uncert_pol(geo_ctx: MushroomContext, an_ind):
    divergence = angular_res_an(geo_ctx=geo_ctx, an_index=an_ind)
    return divergence


def uncert_azi(geo_ctx: MushroomContext, an_ind):
    an_point = geo_ctx.analyser_points[:, an_ind]
    # sa: sample-analyser; af: analyser-focus
    distance_sa = geo.points_distance(point1=geo_ctx.sample_point, point2=an_point)
    distance_af = geo.points_distance(point1=an_point, point2=geo_ctx.foc_point)
    uncert_azi_sa = 2.0 * np.arctan((instr.an_seg + instr.sam_dia) / (2.0 * distance_sa))
    uncert_azi_af = 2.0 * np.arctan((instr.an_seg + geo_ctx.foc_size) / (2.0 * distance_af))
    return min(uncert_azi_sa, uncert_azi_af)


def relative_uncert_ef(geo_ctx: MushroomContext, an_ind):
    kf_now = geo_ctx.wavenumbers_out[an_ind]
    # the first component from the function "spread_factor_detector" gives the spread in the polar direction
    delta_kf = uncert_kf(geo_ctx, an_ind=an_ind) * spread_factor_detector(geo_ctx=geo_ctx, index_now=an_ind)[0]
    return 2.0 * delta_kf / kf_now


def an_diver_vert(geo_ctx: MushroomContext, an_ind):
    # sa: sample-analyser; af: analyser-focus
    an_point = geo_ctx.analyser_points[:, an_ind]
    vector_sa = geo.points2vector(point1=geo_ctx.sample_point, point2=an_point)
    vector_af = geo.points2vector(point1=an_point, point2=geo_ctx.foc_point)
    vector_tangential = geo.vector_bisector(vector_sa, vector_af)
    segment_analyser = geo.unit_vector(vector_tangential) * instr.an_seg
    an_in_proj = geo.vector_project_a2b(segment_analyser, vector_sa)
    an_in_rej = segment_analyser - an_in_proj  # rejection
    an_out_proj = geo.vector_project_a2b(segment_analyser, vector_af)
    an_out_rej = segment_analyser - an_out_proj

    diver_in = 2 * np.arctan((instr.sample_height * abs(
        np.cos(geo.points_to_slope_radian(point1=geo_ctx.sample_point, point2=an_point))) + np.linalg.norm(
        an_in_rej)) / (2.0 * np.linalg.norm(vector_sa)))
    diver_out = 2 * np.arctan((geo_ctx.foc_size * abs(
        np.sin(geo.points_to_slope_radian(point1=an_point, point2=geo_ctx.foc_point))) + np.linalg.norm(an_out_rej)) / (
                                      2.0 * np.linalg.norm(vector_af)))
    return diver_in, diver_out


def kf_resol_mcstas(geo_ctx: MushroomContext, index_now):
    factor_pol, factor_azi = spread_factor_detector(geo_ctx=geo_ctx, index_now=index_now)
    # factor_polar, factor_azimuth = 1, 1
    kf = geo_ctx.wavenumbers_out[index_now]
    dkf = uncert_kf(geo_ctx, an_ind=index_now) * factor_pol
    dphi = uncert_pol(geo_ctx, an_ind=index_now) * factor_pol
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta = uncert_azi(geo_ctx=geo_ctx, an_ind=index_now) * factor_azi
    dkf_x = kf * np.sin(dtheta)
    dkf_y = kf * np.sin(dphi)
    dkf_z = dkf
    return np.array([dkf_x, dkf_y, dkf_z])


# to compare the analyser generated by the two different methods
def plot_analyser_comparison(geo_ctx: MushroomContext):
    points_analyser_x, points_analyser_y = geo_ctx.theo_ellipse_points[0], geo_ctx.theo_ellipse_points[1]
    points_x, points_y = geo_ctx.analyser_points[0, :], geo_ctx.analyser_points[1, :]
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

    plot_filename = "".join([fmt.path_geometry, "Geometry_Comparison.pdf"])
    plt.tick_params(axis="both", direction="in")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(10)
    print("Plot saved: {:s}".format(plot_filename))


def plot_geometry(geo_ctx: MushroomContext):
    # to plot the geometry with important parameters
    def plot_for_analyser_point(index_now):
        nonlocal geo_ctx, ax
        energy_out = neutron.wavenumber2energy(geo_ctx.wavenumbers_out[index_now])
        res_ef = relative_uncert_ef(geo_ctx=geo_ctx, an_ind=index_now)
        res_ef *= energy_out
        analyser_point = geo_ctx.analyser_points[:, index_now]
        detector_point = geo_ctx.detector_points[:, index_now]
        line_sp_plot = ([geo_ctx.sample_point[0], analyser_point[0]], [geo_ctx.sample_point[1], analyser_point[1]])
        line_pf_plot = ([analyser_point[0], detector_point[0]], [analyser_point[1], detector_point[1]])
        ax.plot(*line_sp_plot, color='#17becf')
        ax.plot(*line_pf_plot, color='#17becf')

        line_sp_plot = ([geo_ctx.sample_point[0], -analyser_point[0]], [geo_ctx.sample_point[1], analyser_point[1]])
        line_pf_plot = ([-analyser_point[0], -detector_point[0]], [analyser_point[1], detector_point[1]])
        ax.plot(*line_sp_plot, color='#17becf')
        ax.plot(*line_pf_plot, color='#17becf')

        ax.plot(analyser_point[0], analyser_point[1], "ko")
        plt.text(x=-analyser_point[0] * 1.05 - 0.4, y=analyser_point[1] * 1.05,
                 s="{:5.2f}".format(neutron.joule2mev(energy_out)))
        plt.text(x=analyser_point[0] * 1.05 + 0.03, y=analyser_point[1] * 1.05,
                 s="{:5.2f}".format(neutron.joule2mev(res_ef) * 1e3))

    fig, ax = plt.subplots(figsize=(8, 7))
    # first plot the analyser on both sides
    ax.plot(geo_ctx.analyser_points[0, :], geo_ctx.analyser_points[1, :], color='#1f77b4', linewidth=5)
    ax.plot(-geo_ctx.analyser_points[0, :], geo_ctx.analyser_points[1, :], color='#1f77b4', linewidth=5)
    ax.set_xlabel("Radial axis (m)")
    ax.set_ylabel("Vertical axis (m)")

    plt.text(x=-1.5, y=0.25, s=r"$E_f$ (meV)")
    plt.text(x=0.9, y=0.25, s=r"$\Delta E_f$ ($\mu$eV)")

    plt.text(-1.25, -2.5,
             r"Wavenumber $k_f \in$ [{:.2f}, {:.2f}] $\AA^{{-1}}$".format(np.min(geometryctx.wavenumbers_out) * 1e-10,
                                                                          np.max(geometryctx.wavenumbers_out) * 1e-10))

    plot_for_analyser_point(index_now=0)
    plot_for_analyser_point(index_now=-2)

    index_largest_energy = np.argmax(geo_ctx.wavenumbers_out)
    plot_for_analyser_point(index_now=index_largest_energy)

    # mark the position of the sample and focus, and plot the detector
    ax.plot(*geo_ctx.sample_point, "ro")
    plt.text(x=-0.075, y=-0.6, s="Sample", rotation=90)
    ax.plot(*geo_ctx.foc_point, "ro", alpha=0.5)
    plt.text(x=geo_ctx.foc_point[0] - 0.5, y=geo_ctx.foc_point[1] - 0.075, s="Focus")
    ax.plot(*geo_ctx.detector_points, '.', color='#8c564b')
    ax.plot(- geo_ctx.detector_points[0, :], geo_ctx.detector_points[1, :], '.', color='#8c564b')

    ax.axis("equal")
    # plt.title("Geometry (sectional view)")
    plot_filename = "".join([fmt.path_geometry, ".".join([geo_ctx.filename_geo, fmt.extension_png])])
    ax.tick_params(axis="both", direction="in")
    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.savefig(geo_ctx.filename_geo + '.png', bbox_inches='tight')
    plt.close(1)
    print("Plot saved: {:s}".format(plot_filename))


def write_mcstas(geo_ctx: MushroomContext):
    # to write the file giving the information of the analyser array for the McStas simulation
    f = open(geo_ctx.filename_mcstas, 'w+')
    value_width_z = instr.an_seg
    value_height_y = instr.an_seg
    value_mosaic_horizontal = geo.deg2min(np.rad2deg(instr.moasic_an))
    value_mosaic_vertical = geo.deg2min(np.rad2deg(instr.moasic_an))
    value_lattice_distance = instr.interplanar_pg002 * 1e10  # it is in angstrom for McStas
    value_position_y = geo_ctx.analyser_points[1, :]
    value_position_z = geo_ctx.analyser_points[0, :]
    value_rotation_x = -np.rad2deg(geo_ctx.mcstas_rotation_rad)

    # This is the code for analyser segments at one azimuthal angle without arms
    for i in range(geo_ctx.analyser_points.shape[1]):
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


def plot_resolution_kf(geo_ctx: MushroomContext):
    # to plot the resolution with the x-axis being the outgoing wave-number
    all_kf = geo_ctx.wavenumbers_out
    max_index = np.argmax(all_kf)
    # all_dqx_m, all_dqy_m, all_dqz_m = resolution_calculation(geo_ctx=geo_ctx)
    resol_qvec = resolution_calculation(geo_ctx)

    fig, ax = plt.subplots()
    ax.plot(all_kf[max_index:] * 1e-10, resol_qvec[0, max_index:] * 1e-10, color=fmt.colour_x, label="x: horizontal")
    ax.plot(all_kf[max_index:] * 1e-10, resol_qvec[1, max_index:] * 1e-10, color=fmt.colour_y, label="y: vertical")
    ax.plot(all_kf[max_index:] * 1e-10, resol_qvec[2, max_index:] * 1e-10, color=fmt.colour_z, label="z: along $k_f$")
    # ax.plot(all_kf * 1e-10, resol_qvec[0, :] * 1e-10, color=fmt.COLOUR_X, label="x: horizontal")
    # ax.plot(all_kf * 1e-10, resol_qvec[1, :] * 1e-10, color=COLOUR_Y, label="y: vertical")
    # ax.plot(all_kf * 1e-10, resol_qvec[2, :] * 1e-10, color=fmt.COLOUR_Z, label="z: along $k_f$")
    ax.set_xlabel(r'Wavenumber $k_f$ ($\AA^{-1}$)')
    ax.set_ylabel(r"Uncertainty $\Delta k_{f,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    ax.grid()
    ax.legend(labelcolor=[fmt.colour_x, fmt.colour_y, fmt.colour_z], loc='upper left', bbox_to_anchor=(0, 1),
              framealpha=0.5)
    ax.tick_params(axis="both", direction="in")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    dkf_percent = resol_qvec[2, max_index:] / all_kf[max_index:] * 1e2
    ax2.plot(all_kf[max_index:] * 1e-10, dkf_percent, color=colour_ax2, label="Relative uncertainty")
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"$\dfrac{\Delta k_f}{k_f}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', color=colour_ax2, labelcolor=colour_ax2)
    ax2.legend(labelcolor=colour_ax2, loc='lower left', bbox_to_anchor=(0, 0.5), framealpha=0.5)
    ax.set_title('Secondary spectrometer')
    filename = "".join([fmt.path_resolution, geo_ctx.filename_res])
    plt.savefig(".".join([filename, fmt.extension_pdf]), bbox_inches='tight')
    plt.savefig(".".join([filename, fmt.extension_png]), bbox_inches='tight')
    print("Plot saved: {:s} & {:s}".format(".".join([filename, fmt.extension_png]),
                                           ".".join([filename, fmt.extension_png])))


def resolution_calculation(geo_ctx: MushroomContext):
    # only the dimension of the polar angles is considered, since it is symmetric in the respective horizontal planes
    resol_qvector = np.array(
        list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, index_now=i), range(geo_ctx.pol_angles.shape[0] - 1))))
    return resol_qvector.transpose()


def spread_factor_detector(geo_ctx: MushroomContext, index_now):
    an_point = geo_ctx.analyser_points[:, index_now]
    detector_now = geo_ctx.detector_points[:, index_now]
    detector_next = geo_ctx.detector_points[:, index_now + 1]
    spread_factor_polar = max(1, instr.detector_resolution / geo.points_distance(detector_now, detector_next))

    detector_spread_azi = instr.an_seg * geo.points_distance(detector_now, geo_ctx.foc_point) / geo.points_distance(
        an_point, geo_ctx.foc_point)
    spread_factor_azi = max(1, instr.detector_resolution / detector_spread_azi)
    return spread_factor_polar, spread_factor_azi


def distances_fd_as(geo_ctx: MushroomContext, index):
    distance_fd = geo.points_distance(point1=geo_ctx.foc_point, point2=geo_ctx.detector_points[:, index])
    distance_as = geo.points_distance(point1=geo_ctx.foc_point, point2=geo_ctx.analyser_points[:, index])
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
    filename = "".join([fmt.path_geometry, "Distance_fd_af.png"])
    plt.savefig(filename, bbox_inches='tight')
    print("Plot saved: {:s}".format(filename))
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
    plot_filename = "".join([fmt.path_resolution, "kf_psd.pdf"])
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: {:s}".format(plot_filename))


def wavevector_transfer_rotation(rot_angle, wavevector_transfer):
    new_qx, new_qy = geo.rotation_around_z(rot_angle=rot_angle, old_x=wavevector_transfer[0],
                                           old_y=wavevector_transfer[1])
    wavevector_transfer[:2] = new_qx, new_qy
    return wavevector_transfer


geometryctx = MushroomContext()

# print("The index of the middle segment of the polar angle range: {:d}".format(int(geometryctx.pol_middle_index + 1)))
# print(geometryctx.analyser_points[:,:10])
# plot_geometry(geo_ctx=geometryctx)

# plot_distance_fd_as(geo_ctx=geometryctx)

# plot_analyser_comparison(geo_ctx=geometryctx)

plot_resolution_kf(geo_ctx=geometryctx)

# write_mcstas(geo_ctx=geometryctx)

# wavenumbers_psd(geo_ctx=geometryctx)
