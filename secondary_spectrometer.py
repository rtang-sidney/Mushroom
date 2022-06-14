import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
import geometry_calculation as geo
import neutron_context as neutron
import instrument_context as instr
import global_context as glb

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


def an_angular_spread(geo_ctx: MushroomContext, an_ind):
    # eta, alpha_i, alpha_f = mosaic, divergence_in, divergence_out
    eta = instr.moasic_an
    alpha_i, alpha_f = an_diver_vert(geo_ctx=geo_ctx, an_ind=an_ind)

    # For the formula see [Paper1]
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2
    return np.sqrt(numerator / denominator)


def uncert_kf(geo_ctx: MushroomContext, an_ind):
    kf = geo_ctx.wavenumber_f
    uncert_theta_an = an_angular_spread(geo_ctx=geo_ctx, an_ind=an_ind)
    twotheta_an = geo_ctx.twotheta_an
    uncert_kf_bragg = kf * np.linalg.norm([instr.deltad_d, uncert_theta_an / np.tan(twotheta_an / 2.0)])
    return uncert_kf_bragg


def uncert_pol(geo_ctx: MushroomContext, an_ind):
    divergence = an_angular_spread(geo_ctx=geo_ctx, an_ind=an_ind)
    return divergence


def uncert_azi(geo_ctx: MushroomContext, an_ind):
    an_point = geo_ctx.an_points[:, an_ind]
    # sa: sample-analyser; af: analyser-focus
    distance_sa = geo.points_distance(point1=geo_ctx.sa_point, point2=an_point)
    distance_af = geo.points_distance(point1=an_point, point2=geo_ctx.foc_point)
    uncert_azi_sa = 2.0 * np.arctan((instr.an_seg + instr.sam_dia) / (2.0 * distance_sa))
    uncert_azi_af = 2.0 * np.arctan((instr.an_seg + geo_ctx.foc_size) / (2.0 * distance_af))
    return min(uncert_azi_sa, uncert_azi_af)


def relative_uncert_ef(geo_ctx: MushroomContext, an_ind):
    kf_now = geo_ctx.wavenumber_f
    # the first component from the function "spread_factor_detector" gives the spread in the polar direction
    delta_kf = uncert_kf(geo_ctx, an_ind=an_ind) * spread_factor_detector(geo_ctx=geo_ctx, index_now=an_ind)[0]
    return 2.0 * delta_kf / kf_now


def an_diver_vert(geo_ctx: MushroomContext, an_ind):
    # sa: sample-analyser; af: analyser-focus
    an_point = geo_ctx.an_points[:, an_ind]
    vector_sa = geo.points2vector(p_start=geo_ctx.sa_point, p_end=an_point)
    vector_af = geo.points2vector(p_start=an_point, p_end=geo_ctx.foc_point)
    vector_tangential = geo.vector_bisector(vector_sa, vector_af)
    segment_analyser = geo.unit_vector(vector_tangential) * instr.an_seg
    an_in_proj = geo.vector_project_a2b(segment_analyser, vector_sa)
    an_in_rej = segment_analyser - an_in_proj  # rejection
    an_out_proj = geo.vector_project_a2b(segment_analyser, vector_af)
    an_out_rej = segment_analyser - an_out_proj

    diver_in = 2 * np.arctan((instr.sam_height * abs(
        np.cos(geo.points_to_slope_radian(point1=geo_ctx.sa_point, point2=an_point))) + np.linalg.norm(
        an_in_rej)) / (2.0 * np.linalg.norm(vector_sa)))
    diver_out = 2 * np.arctan((geo_ctx.foc_size * abs(
        np.sin(geo.points_to_slope_radian(point1=an_point, point2=geo_ctx.foc_point))) + np.linalg.norm(an_out_rej)) / (
                                      2.0 * np.linalg.norm(vector_af)))
    return diver_in, diver_out


def kf_resol_mcstas(geo_ctx: MushroomContext, index_now):
    factor_pol, factor_azi = spread_factor_detector(geo_ctx=geo_ctx, index_now=index_now)
    # factor_polar, factor_azimuth = 1, 1
    dkf = uncert_kf(geo_ctx, an_ind=index_now) * factor_pol
    dphi = uncert_pol(geo_ctx, an_ind=index_now) * factor_pol
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta = uncert_azi(geo_ctx=geo_ctx, an_ind=index_now) * factor_azi
    dkf_x = geo_ctx.wavenumber_f * np.sin(dtheta)
    dkf_y = geo_ctx.wavenumber_f * np.sin(dphi)
    dkf_z = dkf
    return np.array([dkf_x, dkf_y, dkf_z])


# to compare the analyser generated by the two different methods
# def plot_analyser_comparison(geo_ctx: MushroomContext):
#     points_analyser_x, points_analyser_y = geo_ctx.theo_ellipse_points[0], geo_ctx.theo_ellipse_points[1]
#     points_x, points_y = geo_ctx.an_points[0, :], geo_ctx.an_points[1, :]
#     plt.figure(10)
#     ax = plt.gca()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.plot(points_x, points_y)
#     plt.plot(points_analyser_x, points_analyser_y)
#     plt.legend((r"Segments with 1x1 cm$^2$", "Ideal ellipse"))
#     plt.text(0.3, -0.3, "Number of segments in one cut-plane: {:d}".format(len(points_x)))
#     plt.text(0.3, -0.35, "Largest deviation from the ideal ellipse: {:5.2f} m".format(
#         geo.points_distance([points_x[-1], points_y[-1]], [points_analyser_x[-1], points_analyser_y[-1]])))
#     plt.xlabel("x axis (m)")
#     plt.ylabel("y axis (m)")
#     plt.plot(*geometryctx.sa_point, "ro")
#     plt.plot(*geometryctx.foc_point, "ro")
#     plt.text(x=0, y=-0.05, s="Sample")
#     plt.text(x=0.1, y=-0.4, s="Focus")
#
#     plt.plot([geometryctx.sa_point[0], geometryctx.foc_point[0]],
#              [geometryctx.sa_point[1], geometryctx.foc_point[1]])
#     bisecting_x = np.array([0.75, 1])
#     plt.plot(bisecting_x, geo.line_to_y(bisecting_x, geo.points_bisecting_line(point1=geometryctx.sa_point,
#                                                                                point2=geometryctx.foc_point)))
#     plt.axis("equal")
#
#     plot_filename = "".join([glb.path_geometry, "Geometry_Comparison.pdf"])
#     plt.tick_params(axis="both", direction="in")
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close(10)
#     print("Plot saved: {:s}".format(plot_filename))


def plot_geometry(geo_ctx: MushroomContext):
    # to plot the geometry with important parameters
    def plot_for_analyser_point(index_now):
        nonlocal geo_ctx, ax
        energy_out = neutron.wavenumber2energy(geo_ctx.wavenumber_f)
        res_ef = relative_uncert_ef(geo_ctx=geo_ctx, an_ind=index_now)
        res_ef *= energy_out
        analyser_point = geo_ctx.an_points[:, index_now]
        detector_point = geo_ctx.dete_points[:, index_now]
        line_sp_plot = ([geo_ctx.sa_point[0], analyser_point[0]], [geo_ctx.sa_point[1], analyser_point[1]])
        line_pf_plot = ([analyser_point[0], detector_point[0]], [analyser_point[1], detector_point[1]])
        ax.plot(*line_sp_plot, color='#17becf')
        ax.plot(*line_pf_plot, color='#17becf')

        line_sp_plot = ([geo_ctx.sa_point[0], -analyser_point[0]], [geo_ctx.sa_point[1], analyser_point[1]])
        line_pf_plot = ([-analyser_point[0], -detector_point[0]], [analyser_point[1], detector_point[1]])
        ax.plot(*line_sp_plot, color='#17becf')
        ax.plot(*line_pf_plot, color='#17becf')

        ax.plot(analyser_point[0], analyser_point[1], "ko")
        plt.text(x=-analyser_point[0] * 1.05 - 0.3, y=analyser_point[1] * 1.2,
                 s="{:5.2f}".format(neutron.joule2mev(energy_out)))
        plt.text(x=analyser_point[0] * 1.05 + 0.01, y=analyser_point[1] * 1.2,
                 s="{:5.2f}".format(neutron.joule2mev(res_ef) * 1e3))

    # print("Cyl. PSD height: {:.2f} m".format(geo_ctx.dete_points[1, -1] - (-geo_ctx.dete_points[-1])))
    fig, ax = plt.subplots(figsize=(8, 7))
    # first plot the analyser on both sides
    ax.plot(geo_ctx.an_points[0, :], geo_ctx.an_points[1, :], color='#1f77b4', linewidth=5)
    ax.plot(-geo_ctx.an_points[0, :], geo_ctx.an_points[1, :], color='#1f77b4', linewidth=5)
    ax.set_xlabel("Radial axis (m)")
    ax.set_ylabel("Vertical axis (m)")

    plt.text(x=-1.2, y=0.7, s=r"$E_f$ (meV)")
    plt.text(x=0.7, y=0.7, s=r"$\Delta E_f$ ($\mu$eV)")

    # plt.text(-1.25, -2.5,
    #          r"Wavenumber $k_f \in$ [{:.2f}, {:.2f}] $\AA^{{-1}}$".format(np.min(geometryctx.wavenumbers_out) * 1e-10,
    #                                                                       np.max(geometryctx.wavenumbers_out) * 1e-10))

    plot_for_analyser_point(index_now=0)
    plot_for_analyser_point(index_now=-2)

    index_middle = int(geo_ctx.an_points.shape[1] / 2)
    plot_for_analyser_point(index_now=index_middle)

    # mark the position of the sample and focus, and plot the detector
    ax.plot(*geo_ctx.sa_point, "ro")
    plt.text(x=-0.075, y=-0.6, s="Sample", rotation=90)
    ax.plot(*geo_ctx.foc_point, "ro", alpha=0.5)
    plt.text(x=geo_ctx.foc_point[0] - 0.5, y=geo_ctx.foc_point[1] - 0.075, s="Focus")
    ax.plot(*geo_ctx.dete_points, '.', color='#8c564b')
    ax.plot(- geo_ctx.dete_points[0, :], geo_ctx.dete_points[1, :], '.', color='#8c564b')

    ax.axis("equal")
    # plt.title("Geometry (sectional view)")
    plot_filename = "".join([glb.path_geometry, ".".join([geo_ctx.filename_geo, glb.extension_pdf])])
    ax.tick_params(axis="both", direction="in")
    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.savefig(geo_ctx.filename_geo + '.png', bbox_inches='tight')
    plt.close(1)
    print("Plot saved: {:s}".format(plot_filename))


def mcstas_analyser(geo_ctx: MushroomContext):
    # to write the file giving the information of the analyser array for the McStas simulation
    mcstas_fname = "".join([glb.path_mcstas, geo_ctx.filename_mcstas])
    f = open(mcstas_fname, 'w+')
    value_width_z = instr.an_seg
    value_height_y = instr.an_seg
    value_mosaic_horizontal = geo.deg2min(np.rad2deg(instr.moasic_an))
    value_mosaic_vertical = geo.deg2min(np.rad2deg(instr.moasic_an))
    value_lattice_distance = instr.interplanar_pg002 * 1e10  # it is in angstrom for McStas
    value_position_y = geo_ctx.an_points[1, :]
    value_position_z = geo_ctx.an_points[0, :]
    value_rotation_x = -np.rad2deg(geo_ctx.mcstas_rot_rad)

    # This is the code for analyser segments at one azimuthal angle without arms
    for i in range(geo_ctx.an_points.shape[1] - 1):
        if i == 0:
            string_an1 = 'COMPONENT {}{} = {}({} = {:.2f}, {} = {:.2f}, {} = {:.0f}, {} = {:.0f}, {} = {:.3f})\n'.format(
                geo_ctx.component_name_prefix, i, geo_ctx.component_ana, geo_ctx.parameter_width_z, value_width_z,
                geo_ctx.parameter_height_y, value_height_y, geo_ctx.parameter_mosaic_horizontal,
                value_mosaic_horizontal, geo_ctx.parameter_mosaic_vertical, value_mosaic_vertical,
                geo_ctx.parameter_lattice_distance, value_lattice_distance)
        else:
            string_an1 = "COMPONENT {}{} = COPY ({}0)\n".format(geo_ctx.component_name_prefix, i,
                                                                geo_ctx.component_name_prefix)
        string_an2 = 'AT (0, {:.3f}, {:.3f}) RELATIVE {}\n'.format(value_position_y[i], value_position_z[i],
                                                                   geo_ctx.component_reference)
        string_an3 = 'ROTATED ({:.3f}, 0, 90) RELATIVE {}\n'.format(value_rotation_x[i], geo_ctx.component_reference)
        # string_an3 = 'ROTATED (90, 90, 0) RELATIVE {}\n'.format(geo_ctx.component_reference)
        string_an4 = 'GROUP {}\n\n'.format(geo_ctx.group_name)
        string_analyser = string_an1 + string_an2 + string_an3 + string_an4
        f.write(string_analyser)
    f.close()
    print("McStas file saved: {:s}".format(mcstas_fname))


def kf_resolution(geo_ctx: MushroomContext):
    # the components are calculated in the coordinate system used on McStas

    resol_qvec = np.array(
        list(map(lambda i: kf_resol_mcstas(geo_ctx=geo_ctx, index_now=i), range(geo_ctx.pol_angles.shape[0]))))
    resol_qvec = resol_qvec.transpose()
    pol_deg = np.rad2deg(geo_ctx.pol_angles)

    fig, ax = plt.subplots()
    ax.plot(pol_deg, resol_qvec[0, :] * 1e-10, color=glb.colour_x, label="x")  #: horizontal
    ax.plot(pol_deg, resol_qvec[1, :] * 1e-10, color=glb.colour_y, label="y")  #: vertical
    ax.plot(pol_deg, resol_qvec[2, :] * 1e-10, color=glb.colour_z, label="z")  #: along $k_f$

    ax.set_xlabel(r'Polar angle (degree)')
    ax.set_ylabel(r"Uncertainty $\Delta k_{f,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    # ax.grid()
    # ax.legend(labelcolor=[glb.colour_x, glb.colour_y, glb.colour_z], loc='upper left', bbox_to_anchor=(0, 1),
    #           framealpha=0.5)
    ax.tick_params(axis="both", direction="in")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    dkf_percent = resol_qvec[2, :] / geo_ctx.wavenumber_f * 1e2
    print("Relative uncertainty in plane:", dkf_percent[np.argmin(abs(geo_ctx.pol_angles))])
    ax2.plot(pol_deg, dkf_percent, color=colour_ax2, label="Relative uncertainty")
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"$\dfrac{\Delta k_f}{k_f}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', color=colour_ax2, labelcolor=colour_ax2)
    # ax2.legend(labelcolor=colour_ax2, loc='lower left', bbox_to_anchor=(0, 0.5), framealpha=0.5)
    ax.set_title('Secondary spectrometer')
    filename = "".join([glb.path_resolution, geo_ctx.filename_res])
    filename = ".".join([filename, glb.extension_png])
    plt.savefig(filename, bbox_inches='tight')
    print("Plot saved: {:s} ".format(filename),
          "line colours: {:s}: x-horizontal, {:s}: y-horizontal, {:s}: z-horizontal, {:s}: relative uncertainty".format(
              glb.colour_x, glb.colour_y, glb.colour_z, colour_ax2))


def spread_factor_detector(geo_ctx: MushroomContext, index_now):
    an_point = geo_ctx.an_points[:, index_now]
    detector_now = geo_ctx.dete_points[:, index_now]
    detector_next = geo_ctx.dete_points[:, index_now + 1]
    spread_factor_polar = max(1, instr.detector_resolution / geo.points_distance(detector_now, detector_next))

    detector_spread_azi = instr.an_seg * geo.points_distance(detector_now, geo_ctx.foc_point) / geo.points_distance(
        an_point, geo_ctx.foc_point)
    spread_factor_azi = max(1, instr.detector_resolution / detector_spread_azi)
    return spread_factor_polar, spread_factor_azi


def distance_sd(geo_ctx: MushroomContext, index):
    distance_sa = geo.points_distance(point1=geo_ctx.sa_point, point2=geo_ctx.an_points[:, index])
    distance_ad = geo.points_distance(point1=geo_ctx.an_points[:, index], point2=geo_ctx.dete_points[:, index])
    distance_sd = distance_sa + distance_ad
    return distance_sd


def write_an_segs():
    fname = glb.path_mcstas + "AnalyserSegments.dat"
    file = open(fname, "w+")
    file.write("# arm_ang (deg), pos_y (m), pos_z (m), rot_x (deg) \n")

    for azi_ang in instr.azi_rad:
        # for j in range(geo_ctx.pol_angles.shape[0]):
        file.write("{:.0f} \n".format(-np.rad2deg(azi_ang)))

        print("Written for azi {:.0f}".format(np.rad2deg(azi_ang)))


def plot_distances(geo_ctx: MushroomContext):
    pol_deg = np.rad2deg(geo_ctx.pol_angles)
    distances_sa = np.array(list(
        map(lambda i: geo.points_distance(point1=geo_ctx.sa_point, point2=geo_ctx.an_points[:, i]),
            range(geo_ctx.pol_angles.shape[0]))))
    azi_angs = np.rad2deg(2 * np.arctan(instr.an_seg * 0.5 / distances_sa))

    distances_sd = np.array(list(map(lambda i: distance_sd(geo_ctx, i), range(geo_ctx.pol_angles.shape[0]))))
    print(np.max(distances_sd), np.min(distances_sd))

    fig, ax = plt.subplots()

    ax.plot(pol_deg, distances_sd)
    ax.set_xlabel(r'Polar angle (degree)')
    ax.set_ylabel(r"Distance (m)")
    # ax.plot(pol_deg, azi_angs)
    # ax.set_xlabel(r'Polar angle (degree)')
    # ax.set_ylabel(r"Azimuthal angle coverage (degree)")

    ax.tick_params(axis="both", direction="in")
    plt.show()


# def plot_distance_fd_as(geo_ctx: MushroomContext):
#     ratio_fd_af = np.array(
#         list(map(lambda i: distances_fd_as(geo_ctx=geo_ctx, index=i), range(geometryctx.pol_angles.shape[0]))))
#     fig, ax = plt.subplots()
#     ax.plot(np.rad2deg(geometryctx.pol_angles), ratio_fd_af)
#     # plotting_format(ax=ax, grid=True)
#     ax.tick_params(axis="both", direction="in")
#     # if grid is True:
#     ax.grid()
#     ax.set_xlabel("Polar angle of analyser segment (degree)")
#     ax.set_ylabel(r"Ratio $\frac{D_{FD}}{D_{AF}}$")
#     ax.set_title("Distance-ratio focus-detector / analyser-focus")
#     plt.tight_layout()
#     filename = "".join([glb.path_geometry, "Distance_fd_af.png"])
#     plt.savefig(filename, bbox_inches='tight')
#     print("Plot saved: {:s}".format(filename))
#     plt.close(fig)
#
#
# def wavenumbers_psd(geo_ctx: MushroomContext):
#     kf = geo_ctx.wavenumbers_out
#     detec_hori_x = geo_ctx.dete_hori_x
#     detec_vert_y = geo_ctx.dete_vert_y
#     fig, axs = plt.subplots(1, 2, sharey="all")
#     axs[0].plot(detec_vert_y, kf[detec_hori_x.shape[0]:] * 1e-10)
#     axs[0].set_xlim(axs[0].get_xlim()[::-1])
#     axs[1].plot(detec_hori_x, kf[:detec_hori_x.shape[0]] * 1e-10)
#     axs[0].set_xlabel("Vertical position of\n cyl. PSDs (m)")
#     axs[1].set_xlabel("Radial position of\n flat PSD (m)")
#     axs[0].set_ylabel(r"Wavenumber $k_f$ ($\AA^{-1}$)")
#     axs[0].grid()
#     axs[1].grid()
#     axs[0].tick_params(axis="both", direction="in")
#     axs[1].tick_params(axis="both", direction="in")
#     plot_filename = "".join([glb.path_resolution, "kf_psd.pdf"])
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close(fig)
#     print("Plot saved: {:s}".format(plot_filename))


geometryctx = MushroomContext()
print(geometryctx.pol_angles.shape[0])
# write_an_segs(geo_ctx=geometryctx)
# print("The prefix of the middle segment of the polar angle range: {:d}".format(int(geometryctx.pol_middle_index)))
# print(geometryctx.analyser_points[:, :10])
# plot_geometry(geo_ctx=geometryctx)

# plot_distance_fd_as(geo_ctx=geometryctx)

# plot_analyser_comparison(geo_ctx=geometryctx)

kf_resolution(geo_ctx=geometryctx)
# plot_distances(geo_ctx=geometryctx)

# write_mcstas(geo_ctx=geometryctx)

# wavenumbers_psd(geo_ctx=geometryctx)
