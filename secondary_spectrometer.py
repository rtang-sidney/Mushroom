import matplotlib.pyplot as plt
import numpy as np

from geometry_context import GeometryContext
from helper import wavelength_to_eV, points_distance, vector_bisector, InstrumentContext, \
    points_to_vector, \
    get_kf_vector, points_to_slope_radian, unit_vector, vector_project_a2b, deg2min

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082

"""


# Comment from Alex <3
# line: ax + by + c = 0 -> (a, b, c)


# def get_analyser_angular_spread(geo_ctx: GeometryContext, sample, analyser_point, focus_point):
def get_angular_resolution_analyser(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    eta = instrument.moasic_analyser  # mosaic
    alpha_i, alpha_f = vertical_divergence_analyser(geo_ctx=geo_ctx, instrument=instrument,
                                                    analyser_point=analyser_point)  # incoming and outgoing divergence
    # See [Paper1]
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2

    return np.sqrt(numerator / denominator)


def get_uncertainty_kf(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point_now,
                       analyser_point_nearest, kf):
    angular_uncertainty_analyser = get_angular_resolution_analyser(geo_ctx=geo_ctx, instrument=instrument,
                                                                   analyser_point=analyser_point_now)
    twotheta_analyser = geo_ctx.get_twotheta_analyser(analyser_point=analyser_point_now)
    uncertainty_kf_bragg = kf * np.sqrt(np.sum(np.square(
        [instrument.deltad_d, angular_uncertainty_analyser / np.tan(twotheta_analyser / 2.0)])))  # from Bragg's law
    kf_nearest = geo_ctx.wavenumber_bragg(instrument=instrument, analyser_point=analyser_point_nearest)
    uncertainty_kf_segment = abs(kf - kf_nearest)
    return max(uncertainty_kf_bragg, uncertainty_kf_segment)


def get_uncertainty_phi(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    divergence_out = \
        vertical_divergence_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_point=analyser_point)[1]
    # the uncertainty of the polar angle is given by the angular resolution at the analyser
    # return get_angular_resolution_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_point=analyser_point)
    return divergence_out


def get_uncertainty_theta(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    # sa: sample-analyser; af: analyser-focus
    distance_sa = points_distance(point1=geo_ctx.sample_point, point2=analyser_point)
    distance_af = points_distance(point1=analyser_point, point2=geo_ctx.focus_point)
    uncertainty_azimuthal_sa = 2.0 * np.arctan(
        (instrument.analyser_segment + instrument.sample_diameter) / (2.0 * distance_sa))
    uncertainty_azimuthal_af = 2.0 * np.arctan((instrument.analyser_segment + geo_ctx.focus_size) / (2.0 * distance_af))
    return min(uncertainty_azimuthal_sa, uncertainty_azimuthal_af)


def get_relative_uncertainty_energy(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point,
                                    nearest_point):
    kf = geo_ctx.wavenumber_bragg(instrument=instrument, analyser_point=analyser_point)  # outgoing wave number
    delta_kf = get_uncertainty_kf(geo_ctx, instrument, analyser_point_now=analyser_point,
                                  analyser_point_nearest=nearest_point, kf=kf)
    return 2. * delta_kf / kf


# def get_divergence(sample, analyser_point, focus, sample_size, focus_size):
def vertical_divergence_analyser(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    # sa: sample-analyser; af: analyser-focus
    vector_sa = points_to_vector(point1=geo_ctx.sample_point, point2=analyser_point)
    vector_af = points_to_vector(point1=analyser_point, point2=geo_ctx.focus_point)
    vector_tangential = vector_bisector(vector_sa, vector_af)
    segment_analyser = unit_vector(vector_tangential) * instrument.analyser_segment
    analyser_incoming_projection = vector_project_a2b(segment_analyser, vector_sa)
    analyser_incoming_rejection = segment_analyser - analyser_incoming_projection
    analyser_outgoing_projection = vector_project_a2b(segment_analyser, vector_af)
    analyser_outgoing_rejection = segment_analyser - analyser_outgoing_projection

    divergence_in = 2.0 * np.arctan((instrument.sample_height * abs(np.sin(
        points_to_slope_radian(point1=geo_ctx.sample_point, point2=analyser_point))) + np.linalg.norm(
        analyser_incoming_rejection)) / (2.0 * np.linalg.norm(vector_sa)))
    divergence_out = 2.0 * np.arctan((geo_ctx.focus_size * abs(np.sin(
        points_to_slope_radian(point1=analyser_point, point2=geo_ctx.focus_point))) + np.linalg.norm(
        analyser_outgoing_rejection)) / (2.0 * np.linalg.norm(vector_af)))
    # divergence_in = instrument.sample_size / distance_sa
    # divergence_out = geo_ctx.focus_size / distance_af
    return divergence_in, divergence_out


def get_spread_from_detector(analyser_point, focus, detector, angular_spread_analyser, size_focus):
    if len(analyser_point) != 2:
        return RuntimeError("Invalid analyser point given {}".format(analyser_point))
    if len(focus) != 2:
        return RuntimeError("Invalid focus point given {}".format(focus))
    if len(detector) != 2:
        return RuntimeError("Invalid detector point given {}".format(detector))

    distance_af = points_distance(analyser_point, focus)
    distance_ad = points_distance(analyser_point, detector)
    spread_focus = np.sqrt(2 * distance_af ** 2 * (1 - np.cos(angular_spread_analyser)))
    if spread_focus > size_focus:
        spread_detector = size_focus * distance_ad / distance_af
    else:
        spread_detector = np.sqrt(2 * distance_ad ** 2 * (1 - np.cos(angular_spread_analyser)))
    return spread_detector


def get_resolution_qxy(geo_ctx: GeometryContext, instrument: InstrumentContext, kf, phi, theta, analyser_point_now,
                       analyser_point_nearest, qxy, ki=None):
    if ki is None:
        ki = kf
    delta_kf = get_uncertainty_kf(geo_ctx, instrument=instrument, analyser_point_now=analyser_point_now,
                                  analyser_point_nearest=analyser_point_nearest, kf=kf)
    delta_phi = get_uncertainty_phi(geo_ctx, instrument=instrument, analyser_point=analyser_point_now)
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta_sample = np.sqrt(np.sum(np.square(
        [vertical_divergence_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_point=analyser_point_now)])))

    qxy_kf = np.cos(phi) * (kf * np.cos(phi) - ki * np.cos(theta)) / qxy
    qxy_phi = -kf * np.sin(phi) * (kf * np.cos(phi) - ki * np.cos(theta)) / qxy
    qxy_theta = ki * kf * np.cos(phi) * np.sin(theta) / qxy
    delta_qxy = np.sqrt(np.sum(np.square([qxy_kf * delta_kf, qxy_phi * delta_phi, qxy_theta * dtheta_sample])))
    return delta_qxy


def get_dq_mcstas_coordinate(geo_ctx: GeometryContext, instrument: InstrumentContext, kf, analyser_point_now,
                             analyser_point_nearest, index):
    factor_polar, factor_azimuth = spread_factor_detector(geo_ctx=geo_ctx, instrument=instrument,
                                                          analyser_now=analyser_point_now,
                                                          analyser_nearest=analyser_point_nearest, index_now=index)
    dkf = get_uncertainty_kf(geo_ctx, instrument=instrument, analyser_point_now=analyser_point_now,
                             analyser_point_nearest=analyser_point_nearest, kf=kf) * factor_polar
    dphi = get_uncertainty_phi(geo_ctx, instrument=instrument, analyser_point=analyser_point_now) * factor_polar
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta = get_uncertainty_theta(geo_ctx=geo_ctx, instrument=instrument,
                                   analyser_point=analyser_point_now) * factor_azimuth

    dqx = kf * np.tan(dtheta)
    dqy = kf * np.tan(dphi)
    dqz = dkf
    # print(np.rad2deg(np.arctan(analyser_point_now[1] / analyser_point_now[0])), np.rad2deg(dtheta), kf * 1e-10,
    #       dqx * 1e-10)
    return [dqx, dqy, dqz]


def get_resolution_qy(geo_ctx: GeometryContext, instrument: InstrumentContext, kf, phi, theta, analyser_point_now,
                      analyser_point_nearest, qxy, ki=None):
    delta_kf = get_uncertainty_kf(geo_ctx, instrument=instrument, analyser_point_now=analyser_point_now,
                                  analyser_point_nearest=analyser_point_nearest, kf=kf)
    delta_phi = get_uncertainty_phi(geo_ctx, instrument=instrument, analyser_point=analyser_point_now)
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta_sample = np.sqrt(np.sum(np.square(
        [vertical_divergence_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_point=analyser_point_now)])))

    qy_kf = np.cos(phi) * np.sin(theta)
    qy_phi = -kf * np.sin(phi) * np.sin(theta)
    qy_theta = kf * np.cos(phi) * np.cos(theta)
    delta_qy = np.sqrt(np.sum(np.square([qy_kf * delta_kf, qy_phi * delta_phi, qy_theta * dtheta_sample])))
    return delta_qy


def get_resolution_qz(geo_ctx: GeometryContext, instrument: InstrumentContext, kf, phi, analyser_point_now,
                      analyser_point_nearest):
    delta_kf = get_uncertainty_kf(geo_ctx, instrument=instrument, analyser_point_now=analyser_point_now,
                                  analyser_point_nearest=analyser_point_nearest, kf=kf)
    delta_phi = get_uncertainty_phi(geo_ctx, instrument=instrument, analyser_point=analyser_point_now)

    qz_kf = np.sin(phi)
    qz_phi = kf * np.cos(phi)
    delta_qz = np.sqrt(np.sum(np.square([qz_kf * delta_kf, qz_phi * delta_phi])))
    return delta_qz


def get_qxy(kf_vector):
    ki_vector = np.array([np.linalg.norm(kf_vector), 0, 0])  # k_i is along x-axis and has the same magnitude as k_f
    q_vector = kf_vector - ki_vector
    return np.linalg.norm(q_vector[:2])


def get_qz(kf, polar_angle):
    return kf * np.sin(polar_angle)


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
    plt.plot(*geometryctx.focus_point, "ro")
    plt.text(x=0, y=-0.05, s="Sample")
    plt.text(x=0.1, y=-0.4, s="Focus")
    plt.savefig("Geometry_Comparison.pdf", bbox_inches='tight')
    plt.close(10)


def coordinate_transformation(theta, phi, vector):
    theta = -(np.pi / 2.0 + theta)
    phi = -(np.pi / 2.0 - phi)
    matrix_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    matrix_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(matrix_x, np.dot(matrix_z, vector))


def plot_whole_geometry(geo_ctx: GeometryContext, instrument: InstrumentContext):
    def plot_for_analyser_point(instrument: InstrumentContext, analyser_point, nearest_point, detector_point):
        energy_ev = wavelength_to_eV(
            wavelength=geo_ctx.wavelength_bragg(instrument=instrument, analyser_point=analyser_point))
        e_resolution_ev = get_relative_uncertainty_energy(geo_ctx=geo_ctx, analyser_point=analyser_point,
                                                          nearest_point=nearest_point,
                                                          instrument=instrument)
        e_resolution_ev *= energy_ev

        line_sp_plot = ([geo_ctx.sample_point[0], analyser_point[0]], [geo_ctx.sample_point[1], analyser_point[1]])
        line_pf_plot = ([analyser_point[0], detector_point[0]], [analyser_point[1], detector_point[1]])
        plt.plot(*line_sp_plot, color='#17becf')
        plt.plot(*line_pf_plot, color='#17becf')

        line_sp_plot = ([geo_ctx.sample_point[0], -analyser_point[0]], [geo_ctx.sample_point[1], analyser_point[1]])
        line_pf_plot = ([-analyser_point[0], -detector_point[0]], [analyser_point[1], detector_point[1]])
        plt.plot(*line_sp_plot, color='#17becf')
        plt.plot(*line_pf_plot, color='#17becf')

        plt.plot(analyser_point[0], analyser_point[1], "ko")
        plt.text(x=-analyser_point[0] - 0.35, y=analyser_point[1], s="{:5.2f}".format(energy_ev * 1e3))
        plt.text(x=analyser_point[0] + 0.1, y=analyser_point[1], s="{:5.2f}".format(e_resolution_ev * 1e6))

    # first plot the analyser on both sides
    plt.plot(geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.plot(-geo_ctx.analyser_points[0], geo_ctx.analyser_points[1], color='#1f77b4', linewidth=5)
    plt.xlabel("Radial axis (m)")
    plt.ylabel("Vertical axis (m)")

    plt.text(x=-0.7, y=0.75, s=r"$E$(meV)")
    plt.text(x=0.5, y=0.75, s=r"$\Delta E$($\mu$eV)")

    first_point_analyser = [geo_ctx.analyser_points[0][0], geo_ctx.analyser_points[1][0]]
    first_point_detector = [geo_ctx.detector_points[0][0], geo_ctx.detector_points[1][0]]

    last_point_analyser = [geo_ctx.analyser_points[0][-1], geo_ctx.analyser_points[1][-1]]
    last_point_detector = [geo_ctx.detector_points[0][-1], geo_ctx.detector_points[1][-1]]

    plot_for_analyser_point(instrument=instrument, analyser_point=first_point_analyser,
                            detector_point=first_point_detector,
                            nearest_point=[geo_ctx.analyser_points[0][1], geo_ctx.analyser_points[1][1]])
    plot_for_analyser_point(instrument=instrument, analyser_point=last_point_analyser,
                            detector_point=last_point_detector,
                            nearest_point=[geo_ctx.analyser_points[0][-2], geo_ctx.analyser_points[1][-2]])

    index_largest_energy = np.argmax(np.array(list(
        map(lambda x, y: geo_ctx.wavenumber_bragg(instrument=instrument, analyser_point=[x, y]),
            geo_ctx.analyser_points[0], geo_ctx.analyser_points[1]))))
    plot_for_analyser_point(instrument=instrument, analyser_point=[geo_ctx.analyser_points[0][index_largest_energy],
                                                                   geo_ctx.analyser_points[1][index_largest_energy]],
                            detector_point=[geo_ctx.detector_points[0][index_largest_energy],
                                            geo_ctx.detector_points[1][index_largest_energy]],
                            nearest_point=[geo_ctx.analyser_points[0][index_largest_energy + 1],
                                           geo_ctx.analyser_points[1][index_largest_energy + 1]])

    # mark the position of the sample and focus, and plot the detector
    plt.plot(*geo_ctx.sample_point, "ro")
    plt.text(x=-0.275, y=-0.25, s="Sample", fontsize=15)
    plt.plot(*geo_ctx.focus_point, "ro", alpha=0.5)
    plt.text(x=geo_ctx.focus_point[0] + 0.1, y=geo_ctx.focus_point[1] - 0.1, s="Focus", fontsize=15)
    plt.plot(*geo_ctx.detector_points, color='#8c564b')

    plt.xlim(-1.8, 1.8)

    plt.tight_layout()
    plt.savefig(geo_ctx.filename_geometry + '.pdf', bbox_inches='tight')
    plt.savefig(geo_ctx.filename_geometry + '.png', bbox_inches='tight')
    plt.close(1)
    print("{:s} plotted.".format(geo_ctx.filename_geometry))


# def get_resolution_robbewley(geo_ctx: GeometryContext, instrument: InstrumentContext, all_qxy, all_qz):
#     analyser_x, analyser_y = geo_ctx.analyser_points
#     detector_x, detector_y = geo_ctx.detector_points
#     if analyser_x.shape[0] != detector_x.shape[0]:
#         raise RuntimeError("Analyser and detector points have different sizes, {:d}, {:d}".format(analyser_x.shape[0],
#                                                                                                   detector_x.shape[0]))
#     all_delta_qxy = []
#     all_delta_qz = []
#     for i in range(analyser_x.shape[0]):
#         analyser_point = np.array([analyser_x[i], analyser_y[i]])
#         detector_point = np.array([detector_x[i], detector_y[i]])
#         distance_ad = np.linalg.norm(analyser_point - detector_point)
#
#         dx = np.sqrt((np.tan(2 * instrument.moasic_analyser) * distance_ad) ** 2 + instrument.analyser_segment ** 2)
#         x = detector_x[i]
#         thi = abs(dx / x)
#         dtheta = abs(instrument.analyser_segment / abs(analyser_point[0] - geo_ctx.sample_point[0]))
#         for j in range(len(azimuthal_angles)):
#             k = i * len(azimuthal_angles) + j
#             try:
#                 if j == 0:
#                     dqxy = all_qxy[k] - all_qxy[k + 1]
#                 else:
#                     dqxy = all_qxy[k] - all_qxy[k - 1]
#                 dqxy = abs(dqxy)
#                 delta_qxy = dqxy * thi / dtheta
#                 all_delta_qxy.append(delta_qxy)
#             except IndexError:
#                 print(i, j, k, analyser_x.shape[0], len(azimuthal_angles))
#
#         vector_ad = points_to_vector(analyser_point, detector_point)
#         theta0 = np.arctan(abs(vector_ad[1] / vector_ad[0]))
#         dxy = dx / np.sin(theta0)
#         x_spread = abs(dxy)
#         if i == 0:
#             next_point = np.array([detector_x[i + 1], detector_y[i + 1]])
#             x_point = abs(detector_point[0] - next_point[0])
#             dqz = all_qz[i] - all_qz[i + 1]
#
#         else:
#             last_point = np.array([detector_x[i - 1], detector_y[i - 1]])
#             x_point = abs(detector_point[0] - last_point[0])
#             dqz = all_qz[i] - all_qz[i - 1]
#         dqz = abs(dqz)
#         delta_qz = dqz * x_spread / x_point
#         all_delta_qz.append(delta_qz)
#     # print(len(all_delta_qxy), len(all_delta_qz))
#     return np.array(all_delta_qxy), np.array(all_delta_qz)


def plot_resolution(geo_ctx: GeometryContext, all_qxy, all_dqxy, all_qz, all_dqz):
    # plot the horizontal component of the q-resolution calculated by us
    plt.figure(2)
    plt.plot(all_qxy * 1e-10, all_dqxy * 1e-10, '.')
    plt.xlabel(r"$Q_{xy}$ (Angstrom -1)")
    plt.ylabel(r"$\Delta Q_{xy}$ (Angstrom -1)")
    plt.title("Q resolution - horizontal")
    plt.grid()
    plt.savefig(geo_ctx.filename_horizontal, bbox_inches='tight')
    plt.close(2)

    # plot the vertical component of the q-resolution calculated by us
    plt.figure(3)
    plt.plot(all_qz * 1e-10, all_dqz * 1e-10, '.')
    plt.xlabel(r"$Q_{z}$ (Angstrom -1)")
    plt.ylabel(r"$\Delta Q_{z}$ (Angstrom -1)")
    plt.title("Q resolution - vertical")
    plt.grid()
    plt.savefig(geo_ctx.filename_vertical, bbox_inches='tight')
    plt.close(3)


def plot_resolution_comparison(all_qxy, all_dqxy, all_delta_qxy_rob, all_qz, all_dqz, all_delta_qz_rob):
    # compare the horizontal component of the q-resolution calculated by us and by Rob Bewley
    plt.figure(4)
    plt.subplot(121)
    plt.plot(all_qxy * 1e-10, all_dqxy * 1e-10, '.')
    plt.xlabel(r"$Q_{xy}$ (Angstrom -1)")
    plt.ylabel(r"$\Delta Q_{xy}$ (Angstrom -1)")
    plt.grid()
    plt.subplot(122)
    plt.plot(all_qxy * 1e-10, all_delta_qxy_rob * 1e-10, '.')
    plt.xlabel(r"$Q_{xy}$ (Angstrom -1)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Comparison_Horizontal.pdf", bbox_inches='tight')
    plt.close(4)

    # compare the vertical component of the q-resolution calculated by us and by Rob Bewley
    plt.figure(5)
    plt.subplot(121)
    plt.plot(all_qz * 1e-10, all_dqz * 1e-10, '.')
    plt.xlabel(r"$Q_{z}$ (Angstrom -1)")
    plt.ylabel(r"$\Delta Q_{z}$ (Angstrom -1)")
    plt.grid()
    plt.subplot(122)
    plt.plot(all_qz * 1e-10, all_delta_qz_rob * 1e-10, '.')
    plt.xlabel(r"$Q_{z}$ (Angstrom -1)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Comparison_Vertical.pdf", bbox_inches='tight')
    plt.close(5)


def write_mcstas(geo_ctx: GeometryContext, instrument: InstrumentContext):
    f = open(geo_ctx.mcstas_filename, 'w+')
    value_width_z = instrument.analyser_segment
    value_height_y = instrument.analyser_segment
    value_mosaic_horizontal = deg2min(np.rad2deg(instrument.moasic_analyser))
    value_mosaic_vertical = deg2min(np.rad2deg(instrument.moasic_analyser))
    value_lattice_distance = instrument.lattice_distance_pg002 * 1e10  # it is in angstrom for McStas
    value_position_y = geo_ctx.analyser_points[1]
    value_position_z = geo_ctx.analyser_points[0]
    value_rotation_x = -np.rad2deg(geo_ctx.mcstas_rotation_radian)

    # for j, azimuthal_angle in enumerate(geo_ctx.azimuthal_angles):
    #     arm_sa_name = "{}{}".format(geo_ctx.arm_sa_name_prefix, j)
    #     string_arm_sa1 = "COMPONENT {} = Arm()\n".format(arm_sa_name)
    #     string_arm_sa2 = 'AT (0, 0, 0) RELATIVE {}\n'.format(geo_ctx.arm_sa_reference)
    #     string_arm_sa3 = 'ROTATED (0, {}, 0) RELATIVE {}\n\n'.format(np.rad2deg(azimuthal_angle),
    #                                                                  geo_ctx.arm_sa_reference)
    #     string_arm_sa = string_arm_sa1 + string_arm_sa2 + string_arm_sa3
    #     f.write(string_arm_sa)
    #     for i in range(geo_ctx.analyser_points[0].shape[0]):
    #         string_an1 = 'COMPONENT {}{}_{} = {}({} = {}, {} = {}, {} = {}, {} = {}, {} = {})\n'.format(
    #             geo_ctx.component_name_prefix, j, i, geo_ctx.component_type, geo_ctx.parameter_width_z, value_width_z,
    #             geo_ctx.parameter_height_y, value_height_y, geo_ctx.parameter_mosaic_horizontal,
    #             value_mosaic_horizontal,
    #             geo_ctx.parameter_mosaic_vertical, value_mosaic_vertical, geo_ctx.parameter_lattice_distance,
    #             value_lattice_distance)
    #         string_an2 = 'AT (0, {}, {}) RELATIVE {}\n'.format(value_position_y[i], value_position_z[i], arm_sa_name)
    #         string_an3 = 'ROTATED ({}, 0, 90) RELATIVE {}\n'.format(value_rotation_x[i], arm_sa_name)
    #         # string_an3 = 'ROTATED (90, 90, 0) RELATIVE {}\n'.format(geo_ctx.component_reference)
    #         string_an4 = 'GROUP {}\n\n'.format(geo_ctx.group_name)
    #         string_analyser = string_an1 + string_an2 + string_an3 + string_an4
    #         f.write(string_analyser)

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
    filename = geo_ctx.filename_polarangle
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.savefig(filename + '.png', bbox_inches='tight')
    print("{:s} plotted.".format(filename))


def resolution_calculation(geo_ctx: GeometryContext, instrument: InstrumentContext, polar_angles, azimuthal_angles):
    # records the calculated data for plotting later
    all_qxy = []
    all_qz = []
    all_dqxy = []
    all_dqz = []
    all_dqx_m = np.empty_like(polar_angles)  # subscript m for McStas
    all_dqy_m = np.empty_like(polar_angles)
    all_dqz_m = np.empty_like(polar_angles)

    # calculate the q-resolution for each segment on the analyser, which gives different q-vectors
    # the outer loop is for the polar angle, and the inner one for the azimuthal angle
    for i, phi in enumerate(polar_angles):
        # calculates the resolution in the horizontal plane and in the vertical direction.
        # qxy and qz have different dimensions!
        point_now = [geo_ctx.analyser_points[0][i], geo_ctx.analyser_points[1][i]]
        if i == 0:
            j = 1
        else:
            j = i - 1
        point_nearest = [geo_ctx.analyser_points[0][j], geo_ctx.analyser_points[1][j]]
        kf = geo_ctx.wavenumber_bragg(instrument=instrument, analyser_point=point_now)
        qz = get_qz(kf=kf, polar_angle=phi)
        delta_qz = get_resolution_qz(geo_ctx=geo_ctx, instrument=instrument, analyser_point_now=point_now,
                                     analyser_point_nearest=point_nearest, kf=kf, phi=phi)
        for theta in azimuthal_angles:
            kf_vector = get_kf_vector(kf_norm=kf, azimuthal=theta,
                                      polar=phi)  # the kf-vector changes is determined by the azimuthal and polar angles
            qxy = get_qxy(kf_vector=kf_vector)  # horizontal component of q-vector
            delta_qxy = get_resolution_qxy(geo_ctx=geometryctx, instrument=instrument, analyser_point_now=point_now,
                                           analyser_point_nearest=point_nearest, kf=kf, phi=phi, qxy=qxy, theta=theta)
            all_qxy.append(qxy)
            all_dqxy.append(delta_qxy)
        all_dqx_m[i], all_dqy_m[i], all_dqz_m[i] = get_dq_mcstas_coordinate(geo_ctx=geometryctx, instrument=instrument,
                                                                            analyser_point_now=point_now,
                                                                            analyser_point_nearest=point_nearest, kf=kf,
                                                                            index=i)
        # print("kf and dkf", kf * 1e-10, all_dqz_m[i] * 1e-10)

        all_qz.append(qz)
        all_dqz.append(delta_qz)

    all_qxy = np.array(all_qxy)
    all_qz = np.array(all_qz)
    all_dqxy = np.array(all_dqxy)
    all_dqz = np.array(all_dqz)
    return all_qxy, all_qz, all_dqxy, all_dqz, all_dqx_m, all_dqy_m, all_dqz_m


def check_detector_spread(geo_ctx: GeometryContext, instrument: InstrumentContext):
    keep_index = []
    delete_index = []
    spread_factor = np.empty_like(geo_ctx.detector_points[0])
    i = 0
    while i < geo_ctx.detector_points[0].shape[0] - 1:
        keep_index.append(i)
        point_now = [geo_ctx.detector_points[0][i], geo_ctx.detector_points[1][i]]
        j = i + 1
        while j < geo_ctx.detector_points[0].shape[0]:
            point_next = [geo_ctx.detector_points[0][j], geo_ctx.detector_points[1][j]]
            if points_distance(point_now, point_next) < instrument.detector_resolution:
                delete_index.append(j)
                # print(j, points_distance(point_now, point_next))
                j += 1
            else:
                break
        i = j
    print(len(keep_index), keep_index)
    print(len(delete_index), delete_index)


def spread_factor_detector(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_now, analyser_nearest,
                           index_now):
    detector_now = [geo_ctx.detector_points[0][index_now], geo_ctx.detector_points[1][index_now]]
    if index_now == 0:
        index_nearest = 1
    else:
        index_nearest = index_now - 1
    detector_next = [geo_ctx.detector_points[0][index_nearest], geo_ctx.detector_points[1][index_nearest]]
    spread_factor_polar = max(1, instrument.detector_resolution / points_distance(detector_now, detector_next))

    detector_spread_azimuth = instrument.analyser_segment * points_distance(detector_now,
                                                                            geo_ctx.focus_point) / points_distance(
        analyser_now, geo_ctx.focus_point)
    spread_factor_azimuth = max(1, instrument.detector_resolution / detector_spread_azimuth)
    return spread_factor_polar, spread_factor_azimuth


geometryctx = GeometryContext(side="same")
instrumentctx = InstrumentContext()

# points_x, points_y = geometryctx.analyser_points

# plt.figure(1)
# ax = plt.gca()
# ax.set_aspect('equal', 'box')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.add_patch(ellipse)
#
# # generates the azimuthal angle elements based on the size of the analyser segments
# azimuthal_start = np.deg2rad(5.)  # radian
# azimuthal_stop = np.deg2rad(170.)  # radian
# angle_one_segment = np.arcsin(instrumentctx.analyser_segment / geometryctx.start_distance)
# number_points = round(abs(azimuthal_start - azimuthal_stop / angle_one_segment))
# azimuthal_angles = np.linspace(azimuthal_start, azimuthal_stop, num=number_points)
# polar_angles = np.arctan(geometryctx.analyser_points[1][:] / geometryctx.analyser_points[0][:])
#
# all_qxy, all_qz, all_dqxy, all_dqz, all_dqx_m, all_dqy_m, all_dqz_m = resolution_calculation(geo_ctx=geometryctx,
#                                                                                              instrument=instrumentctx,
#                                                                                              polar_angles=polar_angles,
#                                                                                              azimuthal_angles=azimuthal_angles)
# all_kf = np.array(list(map(lambda x, y: wavenumber_bragg(geo_ctx=geometryctx, instrument=instrumentctx,
#                                                          analyser_point=[x, y]), geometryctx.analyser_points[0],
#                            geometryctx.analyser_points[1])))
# plot_whole_geometry(geo_ctx=geometryctx, instrument=instrumentctx)
# plot_resolution_polarangles(geo_ctx=geometryctx, polar_angles=polar_angles, all_dqx_m=all_dqx_m, all_dqy_m=all_dqy_m,
#                             all_dqz_m=all_dqz_m, all_kf=all_kf)
# write_mcstas(geo_ctx=geometryctx, instrument=instrumentctx)

# plot_analyser_comparison(points_analyser_x=geometryctx.analyser_ellipse_points[0],
#                          points_analyser_y=geometryctx.analyser_ellipse_points[1],
#                          points_x=geometryctx.analyser_points[0], points_y=geometryctx.analyser_points[1])

# all_delta_qxy_rob, all_delta_qz_rob = get_resolution_robbewley(geo_ctx=geometryctx, instrument=instrumentctx,
#                                                                all_qxy=all_qxy, all_qz=all_qz)
# plot_resolution_comparison(all_qxy=all_qxy, all_dqxy=all_dqxy, all_delta_qxy_rob=all_delta_qxy_rob, all_qz=all_qz,
#                            all_dqz=all_dqz,
#                            all_delta_qz_rob=all_delta_qz_rob)

# check_detector_spread(geo_ctx=geometryctx, instrument=instrumentctx)

kf = geometryctx.wavenumbers
# detec_hori_x = geometryctx.dete_hori_x
# detec_vert_y = geometryctx.dete_vert_y
# fig, axs = plt.subplots(1, 2, sharey="all")
# axs[0].plot(detec_hori_x, kf[:detec_hori_x.shape[0]] * 1e-10)
# axs[1].plot(detec_vert_y, kf[detec_hori_x.shape[0]:] * 1e-10)
# axs[0].set_xlabel("Radial position of flat PSD (m)")
# axs[1].set_xlabel("Vertical position of cyl. PSDs (m)")
# axs[0].set_ylabel(r"Theoretical values of $k_f$ ($\AA^{-1}$)")
# axs[0].grid()
# axs[1].grid()
# fig.suptitle(r"$k_f$ values - PSD positions")
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("kf_values_psd_positions.pdf")
# # plt.show()
# plt.close(fig)

# plt.figure()
distances = np.linalg.norm([geometryctx.analyser_points[0], geometryctx.analyser_points[1]], axis=0)
scatt_angles = np.array(list(map(lambda i: geometryctx.get_twotheta_analyser(
    analyser_point=[geometryctx.analyser_points[0][i], geometryctx.analyser_points[1][i]]),
                        range(geometryctx.analyser_points[0].shape[0]))))

fig, axs = plt.subplots(1, 3, sharey="all")
# axs[0].plot(geometryctx.detector_points[0], kf * 1e-10)
axs[0].plot(np.rad2deg(geometryctx.polar_angles), kf * 1e-10)
axs[1].plot(geometryctx.detector_points[1], kf * 1e-10)
# axs[1].plot(np.rad2deg(scatt_angles / 2.0), kf * 1e-10)
axs[2].plot(distances, kf * 1e-10)
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.show()
