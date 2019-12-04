import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_line_through_points(point1, point2):
    if len(point1) == 2 and len(point2) == 2:
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        matrix = np.array([[x1, y1], [x2, y2]])
        array = np.full(2, 1.)
        a, b = np.linalg.solve(matrix, array)
        return a, b, 1
    else:
        raise RuntimeError("The point parameters provided are not valid. Try again.")


def get_line_through_parameters(slope, y_intersect=None):
    if y_intersect is None:
        return slope, -1., 0.
    else:
        return slope, -1., -y_intersect


def get_intersect_two_lines(line1, line2):
    if len(line1) == 3 and len(line2) == 3:
        a1 = line1[0]
        b1 = line1[1]
        a2 = line2[0]
        b2 = line2[1]
        matrix = np.array([[a1, b1], [a2, b2]])
        array = np.array([line1[2], line2[2]])
        x1, x2 = np.linalg.solve(matrix, array)
        return x1, x2
    else:
        raise RuntimeError("The line parameters provided are not valid. Try again.")


def get_ellipse_parameters(a, b, alpha):
    linear_eccentricity = np.sqrt(a ** 2 - b ** 2)
    aa = np.cos(alpha) ** 2 / a ** 2 + np.sin(alpha) ** 2 / b ** 2
    bb = 2 * np.cos(alpha) * np.sin(alpha) * (1. / a ** 2 - 1. / b ** 2)
    h = linear_eccentricity * np.cos(alpha)
    k = linear_eccentricity * np.sin(alpha)
    cc = np.sin(alpha) ** 2 / a ** 2 + np.cos(alpha) ** 2 / b ** 2
    return aa, bb, cc, h, k


def get_ellipse_points(semi_major, semi_minor, alpha, detector_line):
    aa, bb, cc, ellipse_centre_x, ellipse_centre_y = get_ellipse_parameters(semi_major, semi_minor, alpha)

    focus = [2. * ellipse_centre_x, 2. * ellipse_centre_y]

    edge1 = get_edge_point(aa, bb, cc, ellipse_centre_x, ellipse_centre_y, np.tan(-np.deg2rad(10)))
    edge2 = get_edge_point(aa, bb, cc, ellipse_centre_x, ellipse_centre_y, np.tan(np.deg2rad(50)))
    line_fd1 = get_line_through_points(focus, edge1)
    line_fd2 = get_line_through_points(focus, edge2)
    detector_point1 = get_intersect_two_lines(line_fd1, detector_line)
    detector_point2 = get_intersect_two_lines(line_fd2, detector_line)
    return focus, edge1, edge2, ellipse_centre_x, ellipse_centre_y, detector_point1, detector_point2


def get_energy(detector_point1, detector_point2, detector_length):
    return abs(abs(detector_point1[0] - detector_point2[0]) - detector_length)


def get_edge_point(aa, bb, cc, h, k, m):
    polynomial_parameters = np.empty(3)
    polynomial_parameters[0] = aa + bb * m + cc * m ** 2
    polynomial_parameters[1] = -2 * aa * h - bb * (m * h + k) - 2 * cc * m * k
    polynomial_parameters[2] = aa * h ** 2 + bb * h * k + cc * k ** 2 - 1
    x = np.roots(polynomial_parameters)
    x = x[x > 0]
    if len(x) == 1:
        x = x[0]
    elif len(x) == 0:
        raise RuntimeError("No x-component of the point has been found.")
    else:
        raise RuntimeError("Too many values of the x-component have been found.")
    y = m * x
    if abs(aa * (x - h) ** 2 + bb * (x - h) * (y - k) + cc * (y - k) ** 2 - 1) < 1e-3:
        return [x, y]
    else:
        raise RuntimeError("Something wrong when solving the ellipse edge points.")


def get_analyser_point(point_y, a, b, alpha, edge):
    def get_one_point_y(point_y, aa, bb, cc, h, k, edge):
        polynomial_parameters = np.empty(3)
        polynomial_parameters[0] = aa
        polynomial_parameters[1] = bb * (point_y - k)
        polynomial_parameters[2] = cc * (point_y - k) ** 2 - 1
        x = np.roots(polynomial_parameters) + h
        point_x = x[0 < x]
        if len(point_x) == 1:
            point_x = point_x[0]
        elif len(point_x) == 0:
            raise RuntimeError("No x-component of the point has been found, found {}.".format(point_x))
        else:
            raise RuntimeError("Too many values of the x-component have been found, found {}.".format(point_x))
        return point_x

    aa, bb, cc, h, k = get_ellipse_parameters(a, b, alpha)
    if isinstance(point_y, float) is True:
        return get_one_point_y(point_y, aa, bb, cc, h, k, edge)
    else:
        return np.array(list(map(lambda y: get_one_point_y(y, aa, bb, cc, h, k, edge), point_y)))


def get_wavelength(crystal_plane_distance, scattering_2theta, order_parameter):
    return 2. * crystal_plane_distance * np.sin(scattering_2theta / 2.) / float(order_parameter)


def get_scattering_angle(sample, focus, point):
    def get_vector_from_points(point1, point2):
        return [point2[0] - point1[0], point2[1] - point1[1]]

    def get_angle_between_vectors(vector1, vector2):
        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

    if len(sample) == 2 and len(focus) == 2 and len(point) == 2:
        vector_sp = get_vector_from_points(sample, point)
        vector_pf = get_vector_from_points(point, focus)
        return get_angle_between_vectors(vector_sp, vector_pf)
    else:
        raise RuntimeError("Given points {}, {} and {} are invalid.".format(sample, focus, point))


def get_energy_from_wavelength(wavelength):
    hbar = 1.0545718e-34  # m2 kg / s
    mass = 1.67492749804e-27  # kg
    energy_in_SI = 2 * np.pi ** 2 * hbar ** 2 / (mass * wavelength ** 2)
    energy_in_eV = energy_in_SI / 1.602176634e-19
    return energy_in_eV


def get_angular_spread(mosaic, incoming_divergence, outgoing_divergence):
    return np.sqrt((
                           incoming_divergence ** 2 * outgoing_divergence ** 2 + mosaic ** 2 * incoming_divergence ** 2 + mosaic ** 2 * outgoing_divergence ** 2) / (
                           4 * mosaic ** 2 + incoming_divergence ** 2 + outgoing_divergence ** 2))


def get_resolution(sample, analyser_point, focus_point, wave_vector):
    def get_distance_two_points(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    if len(sample) != 2:
        raise RuntimeError("Invalid sample position type of {} is given.".format(sample))
    if len(analyser_point) != 2:
        raise RuntimeError("Invalid analyser point type of {} is given.".format(analyser_point))
    if len(focus_point) != 2:
        raise RuntimeError("Invalid detector point type of {} is given.".format(focus_point))

    sample_size = 1e-2  # m
    selector_size = 1e-2  # m
    mosaic_analyser = np.deg2rad(0.4)  # radian
    deltad_d = 6e-4  # given in the paper Keller2002

    distance_sample_analyser = get_distance_two_points(point1=sample, point2=analyser_point)
    distance_analyser_detector = get_distance_two_points(point1=analyser_point, point2=focus_point)
    incoming_divergence = np.arctan(sample_size / distance_sample_analyser)
    outgoing_divergence = np.arctan(selector_size / distance_analyser_detector)
    delta_theta = get_angular_spread(mosaic=mosaic_analyser, incoming_divergence=incoming_divergence,
                                     outgoing_divergence=outgoing_divergence)
    theta = get_scattering_angle(sample, focus_point, analyser_point)
    delta_k = wave_vector * np.sqrt(deltad_d ** 2 + (delta_theta / np.tan(theta)) ** 2)
    wave_vector_outer = wave_vector - delta_k
    delta_k_para = wave_vector - wave_vector_outer * np.cos(delta_theta)
    delta_k_perp = wave_vector_outer * np.sin(delta_theta)
    relative_uncertainty_energy = 2 * delta_k / wave_vector
    return delta_k_para / wave_vector, delta_k_perp / wave_vector, relative_uncertainty_energy


time = datetime.datetime.now()
detector_length = 1.0  # m
detector_line = [0, 1, -1.5]

# Initialising Monte Carlo
temperature = 1.0
scale = 0.5

# a = semi-major axis, b = semi-minor axis, d = vertical distance between the sample and detector upside
semi_major = 1.0
semi_minor = 0.9
alpha = -np.pi / 4.

# Get the parameters after the transformation
focus, edge1, edge2, ellipse_centre_x, ellipse_centre_y, detector_point1, detector_point2 = get_ellipse_points(
    semi_major, semi_minor, alpha, detector_line)

# Doing MCMC to get a suitable combination of the geometric parameters
simulted_annealing_energy = get_energy(detector_point1, detector_point2, detector_length)
for j in range(int(1e5)):
    energy_iteration = np.empty(100)
    for i in range(100):
        # The parameters that have to be defined
        while True:
            random_numbers = np.random.random(3)
            semi_major_new = semi_major + scale * (random_numbers[0] - 0.5)
            semi_minor_new = semi_minor + scale * (random_numbers[1] - 0.5)
            alpha_new = alpha + scale * (random_numbers[2] - 0.5) * np.pi / 4.

            if 2 > semi_major_new > semi_minor_new > 0 and -np.pi / 4. > alpha_new > -np.pi / 2.:
                focus_new, edge1_new, edge2_new, ellipse_centre_x_new, ellipse_centre_y_new, detector_point1_new, detector_point2_new = get_ellipse_points(
                    semi_major_new, semi_minor_new, alpha_new, detector_line)
                if detector_point2_new[0] < 0 and focus_new[1] > detector_line[2] and edge1_new[0] > 1:
                    break
        energy_new = get_energy(detector_point1_new, detector_point2_new, detector_length)
        u = np.random.random()
        delta_energy = energy_new - simulted_annealing_energy
        if u < np.exp(-delta_energy / temperature):
            print(simulted_annealing_energy, energy_new, 'accepted')
            semi_major = semi_major_new
            semi_minor = semi_minor_new
            alpha = alpha_new
            simulted_annealing_energy = energy_new
            focus = focus_new
            edge1 = edge1_new
            edge2 = edge2_new
            ellipse_centre_x = ellipse_centre_x_new
            ellipse_centre_y = ellipse_centre_y_new
            detector_point1 = detector_point1_new
            detector_point2 = detector_point2_new
        else:
            print(simulted_annealing_energy, energy_new, 'abandoned')
        energy_iteration[i] = simulted_annealing_energy
    energy_rms = np.sqrt(np.mean(energy_iteration ** 2))
    print(energy_rms)
    if energy_rms < temperature:
        temperature = temperature / 2.
    if energy_rms < 1e-3:
        break

# file = open('geometry_MCMC.txt', 'a+')
# file.write(
#     '{}, T={:6.4f}, E={:6.4f}, a={:5.3f}m, b={:5.3f}m, alpha={:5.3f}degrees, focus[{:5.3f},{:5.3f}], detector1[{:5.3f},{:5.3f}], detector2[{:5.3f},{:5.3f}]\r\n'.format(
#         time, temperature, energy_rms, semi_major, semi_minor, np.rad2deg(alpha), focus[0],
#         focus[1], detector_point1[0], detector_point1[1], detector_point2[0], detector_point2[1]))
# file.close()

print("Edge points", edge1, edge2)
number_points = int(1e3)
points_analyser_y = np.linspace(edge1[1], edge2[1], number_points)
points_analyser_x = get_analyser_point(points_analyser_y, a=semi_major, b=semi_minor, alpha=alpha,
                                       edge=edge2[0])
all_scattering_2theta = np.array(
    list(map(lambda x, y: get_scattering_angle([0, 0], focus, [x, y]), points_analyser_x, points_analyser_y)))
largest_2theta_position = np.argmax(all_scattering_2theta)
largest_2theta = all_scattering_2theta[largest_2theta_position]
smallest_2theta_position = np.argmin(all_scattering_2theta)
smallest_2theta = all_scattering_2theta[smallest_2theta_position]
print("Scattering angle 2theta: maximum {:5.2f}degrees, minimum {:5.2f}degrees".format(np.rad2deg(largest_2theta),
                                                                                       np.rad2deg(smallest_2theta)))

crystal_plane_distance = 3.35e-10  # m
all_wavelength = get_wavelength(crystal_plane_distance=crystal_plane_distance, scattering_2theta=all_scattering_2theta,
                                order_parameter=1.)
print("Wavelength: maximum {:5.2f}AA, minimum {:5.2f}AA".format(np.max(all_wavelength) * 1e10,
                                                                np.min(all_wavelength) * 1e10))
all_energy = get_energy_from_wavelength(all_wavelength)
print("energy: maximum {:5.2f}meV, minimum {:5.2f}meV".format(np.max(all_energy) * 1e3, np.min(all_energy) * 1e3))

uncertainty_q_para = np.empty(number_points)
uncertainty_q_perp = np.empty_like(uncertainty_q_para)
uncertainty_e = np.empty_like(uncertainty_q_para)
for i in range(number_points):
    uncertainty_q_para[i], uncertainty_q_perp[i], uncertainty_e[i] = get_resolution(sample=[0, 0], analyser_point=[
        points_analyser_x[i], points_analyser_y], focus_point=focus, wave_vector=2 * np.pi /
                                                                                 all_wavelength[i])
print("Resolution: Q_para={} to {}, Q_perp={} to {}, E={} to {}".format(np.min(uncertainty_q_para),
                                                                        np.max(uncertainty_q_para),
                                                                        np.min(uncertainty_q_perp),
                                                                        np.max(uncertainty_q_perp),
                                                                        np.min(uncertainty_e), np.max(uncertainty_e)))
plt.figure()

ellipse = Ellipse(xy=(ellipse_centre_x, ellipse_centre_y), width=2 * semi_major, height=2 * semi_minor,
                  angle=np.rad2deg(alpha), fill=False)
ax = plt.gca()
ax.set_aspect('equal', 'box')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.add_patch(ellipse)
print("alpha={:5.2f}degrees".format(np.rad2deg(alpha)))

plt.plot(points_analyser_x, points_analyser_y, color='#1f77b4', linewidth=5)
plt.plot(-points_analyser_x, points_analyser_y, color='#1f77b4', linewidth=5)
plt.xlabel("x axis (m)")
plt.ylabel("y axis (m)")

point0 = [points_analyser_x[smallest_2theta_position], points_analyser_y[smallest_2theta_position]]
point1 = [points_analyser_x[0], points_analyser_y[0]]
point2 = points_analyser_x[-1], points_analyser_y[-1]

plt.plot(point0[0], point0[1], "ko")
plt.text(x=point0[0] + 0.1, y=point0[1], s=r"$E_{max}$=%5.2f meV" % (all_energy[smallest_2theta_position] * 1e3))

plt.plot(point1[0], point1[1], "ko")
plt.text(x=point1[0] + 0.1, y=point1[1], s=r"$E$=%5.2f meV" % (all_energy[0] * 1e3))

plt.plot(point2[0], point2[1], "ko")
plt.text(x=point2[0] + 0.1, y=point2[1], s=r"$E$=%5.2f meV" % (all_energy[-1] * 1e3))

line_sp0_plot = ([0, point0[0]], [0, point0[1]])
line_sp1_plot = ([0, point1[0]], [0, point1[1]])
line_sp2_plot = ([0, point2[0]], [0, point2[1]])

line_p0f = get_line_through_points(point0, focus)
line_p1f = get_line_through_points(point1, focus)
line_p2f = get_line_through_points(point2, focus)

detector0 = get_intersect_two_lines(line_p0f, detector_line)
detector1 = get_intersect_two_lines(line_p1f, detector_line)
detector2 = get_intersect_two_lines(line_p2f, detector_line)

line_p0f_plot = ([point0[0], detector0[0]], [point0[1], detector0[1]])
line_p1f_plot = ([point1[0], detector1[0]], [point1[1], detector1[1]])
line_p2f_plot = ([point2[0], detector2[0]], [point2[1], detector2[1]])

line_sp0_plot2 = ([0, -point0[0]], [0, point0[1]])
line_sp1_plot2 = ([0, -point1[0]], [0, point1[1]])
line_sp2_plot2 = ([0, -point2[0]], [0, point2[1]])
line_p0f_plot2 = ([-point0[0], -detector0[0]], [point0[1], detector0[1]])
line_p1f_plot2 = ([-point1[0], -detector1[0]], [point1[1], detector1[1]])
line_p2f_plot2 = ([-point2[0], -detector2[0]], [point2[1], detector2[1]])

plt.plot(*line_sp0_plot, color='#17becf')
plt.plot(*line_sp1_plot, color='#17becf')
plt.plot(*line_sp2_plot, color='#17becf')
plt.plot(*line_p0f_plot, color='#17becf')
plt.plot(*line_p1f_plot, color='#17becf')
plt.plot(*line_p2f_plot, color='#17becf')

plt.plot(*line_sp0_plot2, color='#17becf')
plt.plot(*line_sp1_plot2, color='#17becf')
plt.plot(*line_sp2_plot2, color='#17becf')
plt.plot(*line_p0f_plot2, color='#17becf')
plt.plot(*line_p1f_plot2, color='#17becf')
plt.plot(*line_p2f_plot2, color='#17becf')

plt.plot([detector1[0], detector2[0]], [detector1[1], detector2[1]], color='#8c564b', linewidth=5)
plt.plot([-detector1[0], -detector2[0]], [detector1[1], detector2[1]], color='#8c564b', linewidth=5)

plt.plot(0, 0, "ro")
plt.text(x=-0.275, y=-0.25, s="Sample", fontsize=15)

plt.text(0.7, -2.5,
         'Semi-major axes {:5.3f}m\nSemi-minor axes {:5.3f}m\nRotating Angle={:5.3f}°'.format(semi_major, semi_minor,
                                                                                              np.rad2deg(alpha)))
plt.savefig('Mushroom_Geometry_Flat.pdf')
