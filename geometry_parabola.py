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


def get_line_at_point(slope, point):
    x, y = point
    b = y - slope * x
    return slope, -1, -b


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


def get_parabola_parameters(f, theta):
    a = 1. / (4. * f)
    h = f * np.cos(theta - np.pi / 2.)
    k = f * np.sin(theta - np.pi / 2.)
    aa = a * np.cos(theta) ** 2
    bb = a * np.sin(theta) ** 2
    cc = 2 * a * np.cos(theta) * np.sin(theta)
    dd = np.sin(theta)
    ee = -np.cos(theta)

    return aa, bb, cc, dd, ee, h, k


def parabola(f, theta, point):
    x, y = point
    aa, bb, cc, dd, ee, h, k = get_parabola_parameters(f=f, theta=theta)
    return aa * (x - h) ** 2 + bb * (y - k) ** 2 + cc * (x - h) * (y - k) + dd * (x - h) + ee * (y - k)


def get_edge_point(aa, bb, cc, dd, ee, h, k, m):
    polynomial_parameters = np.empty(3)
    polynomial_parameters[0] = aa + bb * m ** 2 + cc * m
    polynomial_parameters[1] = -2 * aa * h - 2 * bb * m * k - cc * (k + h * m) + dd + ee * m
    polynomial_parameters[2] = aa * h ** 2 + bb * k ** 2 + cc * h * k - dd * h - ee * k
    x = np.roots(polynomial_parameters)
    x = x[x > 0]
    if len(x) == 1:
        x = x[0]
    elif len(x) == 0:
        raise RuntimeError("No x-component of the point has been found.")
    else:
        raise RuntimeError("Too many values of the x-component have been found.")
    y = m * x
    if abs(aa * (x - h) ** 2 + bb * (y - k) ** 2 + cc * (x - h) * (y - k) + dd * (x - h) + ee * (y - k)) < 1e-3:
        return [x, y]
    else:
        raise RuntimeError("Something wrong when solving the parabola edge points.")


def get_parabola_points(focus, alpha):
    aa, bb, cc, dd, ee, h, k = get_parabola_parameters(focus, alpha)

    slope = np.tan(alpha - np.pi / 2.)
    upside = [h, k]
    edge1 = get_edge_point(aa, bb, cc, dd, ee, h, k, m=np.tan(5. * np.pi / 12.0))
    edge2 = get_edge_point(aa, bb, cc, dd, ee, h, k, m=np.tan(- np.pi / 12.0))

    line_e1d1 = get_line_at_point(slope, edge1)
    line_e2d2 = get_line_at_point(slope, edge2)

    detector_point1 = get_intersect_two_lines(line_e1d1, np.array([1, 0, -0.5]))
    detector_point2 = get_intersect_two_lines(line_e2d2, np.array([1, 0, -0.5]))
    return slope, upside, edge1, edge2, detector_point1, detector_point2


def get_energy(detector_point1, detector_point2, detector_length):
    return abs(abs(detector_point1[1] - detector_point2[1]) - detector_length)


def get_parabola_point(coordinate, coordinate_name, focus, theta, midline):
    def get_one_point_y(point_x, aa, bb, cc, dd, ee, h, k, midline):
        polynomial_parameters = np.empty(3)
        polynomial_parameters[0] = bb
        polynomial_parameters[1] = cc * (point_x - h) + ee
        polynomial_parameters[2] = aa * (point_x - h) ** 2 + dd * (point_x - h)
        y = np.round(np.roots(polynomial_parameters) + k, 5)

        point_y = y[midline[1] * (midline[0] * point_x + midline[1] * y - midline[2]) > 0]
        if len(point_y) == 1:
            point_y = point_y[0]
        elif len(point_y) == 0:
            print(list(map(lambda q: parabola(f=focus, theta=theta, point=[point_x, q]), y)))
            raise RuntimeError("No y-component of the point has been found, found {} at {}.".format(y, point_x))
        else:
            raise RuntimeError("Too many values of the y-component have been found, found {}.".format(point_y))
        return point_y

    def get_one_point_x(point_y, aa, bb, cc, dd, ee, h, k, midline):
        polynomial_parameters = np.empty(3)
        polynomial_parameters[0] = aa
        polynomial_parameters[1] = cc * (point_y - k) + dd
        polynomial_parameters[2] = bb * (point_y - k) ** 2 + ee * (point_y - k)
        x = np.round(np.roots(polynomial_parameters) + h, 5)

        point_x = x[midline[1] * (midline[0] * x + midline[1] * point_y - midline[2]) < 0]
        if len(point_x) == 1:
            point_x = point_x[0]
        elif len(point_x) == 0:
            print(list(map(lambda q: parabola(f=focus, theta=theta, point=[q, point_y]), x)))
            raise RuntimeError("No x-component of the point has been found, found {} at {}.".format(x, point_y))
        else:
            raise RuntimeError("Too many values of the x-component have been found, found {}.".format(point_x))
        return point_x

    def get_one_point(coordinate, aa, bb, cc, dd, ee, h, k, midline, coordinate_name):
        if coordinate_name == 'x':
            return get_one_point_y(point_x=coordinate, aa=aa, bb=bb, cc=cc, dd=dd, ee=ee, h=h, k=k, midline=midline)
        elif coordinate_name == 'y':
            return get_one_point_x(point_y=coordinate, aa=aa, bb=bb, cc=cc, dd=dd, ee=ee, h=h, k=k, midline=midline)
        else:
            raise RuntimeError("The coordinate name has to be either x or y.")

    if len(midline) != 3:
        raise RuntimeError("The line parameter of the midline is in valid.")
    aa, bb, cc, dd, ee, h, k = get_parabola_parameters(focus, theta)

    if isinstance(coordinate, float) is True:
        return get_one_point(coordinate=coordinate, aa=aa, bb=bb, cc=cc, dd=dd, ee=ee, h=h, k=k, midline=midline,
                             coordinate_name=coordinate_name)
    else:
        return np.array(list(map(
            lambda x: get_one_point(coordinate=x, aa=aa, bb=bb, cc=cc, dd=dd, ee=ee, h=h, k=k, midline=midline,
                                    coordinate_name=coordinate_name), coordinate)))


def get_wavelength(crystal_plane_distance, scattering_2theta, order_parameter):
    return 2. * crystal_plane_distance * np.sin(scattering_2theta / 2.) / float(order_parameter)


def get_scattering_angle(sample, point, slope):
    def get_vector_from_points(point1, point2):
        return [point2[0] - point1[0], point2[1] - point1[1]]

    def get_angle_between_vectors(vector1, vector2):
        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

    if len(sample) == 2 and len(point) == 2 and isinstance(slope, float) is True:
        vector_sp = get_vector_from_points(sample, point)
        vector_pd = [-1., -slope]
        return get_angle_between_vectors(vector_sp, vector_pd)
    else:
        raise RuntimeError("Given points {}, {} and {} are invalid.".format(sample, focus, point))


def get_energy_from_wavelength(wavelength):
    hbar = 1.0545718e-34  # m2 kg / s
    mass = 1.67492749804e-27  # kg
    energy_in_SI = 2 * np.pi ** 2 * hbar ** 2 / (mass * wavelength ** 2)
    energy_in_eV = energy_in_SI / 1.602176634e-19
    return energy_in_eV


time = datetime.datetime.now()
detector_length = 1.0  # m

# Initialising Monte Carlo
temperature = 1.0
scale = 0.5

# a = semi-major axis, b = semi-minor axis, d = vertical distance between the sample and detector upside
focus = 0.6  # m
alpha = 5. * np.pi / 6.

# Get the parameters after the transformation
slope, upside, edge1, edge2, detector_point1, detector_point2 = get_parabola_points(focus=focus, alpha=alpha)

# Doing MCMC to get a suitable combination of the geometric parameters
energy = get_energy(detector_point1, detector_point2, detector_length)
for j in range(int(1e5)):
    energy_iteration = np.empty(100)
    for i in range(100):
        # The parameters that have to be defined
        while True:
            random_numbers = np.random.random(2)
            focus_new = focus + scale * (random_numbers[0] - 0.5)
            alpha_new = alpha + scale * (random_numbers[1] - 0.5) * np.pi / 6.

            if 2 > focus_new > 0 and np.pi / 2. < alpha_new < np.pi:
                slope_new, upside_new, edge1_new, edge2_new, detector_point1_new, detector_point2_new = get_parabola_points(
                    focus_new,
                    alpha_new)
                # print(slope_new, upside_new, edge1_new, edge2_new, detector_point1_new, detector_point2_new)
                if detector_point2_new[1] < -detector_length - 0.1 < detector_point1_new[1] < edge2_new[1] - 0.1 < -0.1:
                    line_p2f_plot2 = get_line_at_point(point=detector_point2_new, slope=slope_new)
                    intersect_the_other_side = get_intersect_two_lines(line_p2f_plot2, [1, 0, 0.5])
                    print(edge1_new, edge2_new, intersect_the_other_side)
                    if intersect_the_other_side[1] > detector_point1_new[1]:
                        print("one point found")
                        break
        energy_new = get_energy(detector_point1_new, detector_point2_new, detector_length)
        u = np.random.random()
        delta_energy = energy_new - energy
        if u < np.exp(-delta_energy / temperature):
            # print(energy, energy_new, 'accepted')
            energy = energy_new
            slope = slope_new
            upside = np.round(upside_new, 5)
            focus = focus_new
            alpha = alpha_new
            edge1 = np.round(edge1_new, 5)
            edge2 = np.round(edge2_new, 5)
            detector_point1 = detector_point1_new
            detector_point2 = detector_point2_new
        # else:
        #     print(energy, energy_new, 'abandoned')
        energy_iteration[i] = energy
    energy_rms = np.sqrt(np.mean(energy_iteration ** 2))
    if energy_rms < temperature:
        temperature = temperature / 2.
    if energy_rms < 1e-3:
        break

# file = open('geometry_MCMC.txt', 'a+')
# file.write(
#     '{}, T={:6.4f}, E={:6.4f}, a={:5.3f}m, b={:5.3f}m, alpha={:5.3f}degrees, focus[{:5.3f},{:5.3f}], detector1[{:5.3f},{:5.3f}], detector2[{:5.3f},{:5.3f}]\r\n'.format(
#         time, temperature, energy_rms, semi_major, np.rad2deg(alpha), focus[0],
#         focus[1], detector_point1[0], detector_point1[1], detector_point2[0], detector_point2[1]))
# file.close()

print("Edge points", edge1, edge2, "Upside", upside)

if abs(parabola(focus, alpha, edge1)) > 1e-3 or abs(parabola(focus, alpha, edge2)) > 1e-3:
    raise RuntimeError("Wrong edge points.")

number_points = int(100)
points_analyser_up_x = np.linspace(edge1[0], upside[0] - 1e-5, number_points)
points_analyser_up_y = get_parabola_point(focus=focus, coordinate=points_analyser_up_x, coordinate_name='x',
                                          theta=alpha,
                                          midline=get_line_through_parameters(slope))
points_analyser_down_y = np.linspace(edge2[1], upside[1] - 1e-5, number_points)
points_analyser_down_x = get_parabola_point(focus=focus, coordinate=points_analyser_down_y, coordinate_name='y',
                                            theta=alpha, midline=get_line_through_parameters(slope))

points_analyser_y = np.append(points_analyser_up_y, points_analyser_down_y)
points_analyser_x = np.append(points_analyser_up_x, points_analyser_down_x)

print("Points got.")

all_scattering_2theta = np.array(list(
    map(lambda x, y: get_scattering_angle(sample=[0, 0], point=[x, y], slope=slope), points_analyser_x,
        points_analyser_y)))

largest_2theta_position = np.argmax(all_scattering_2theta)
largest_2theta = all_scattering_2theta[largest_2theta_position]
smallest_2theta_position = np.argmin(all_scattering_2theta)
smallest_2theta = all_scattering_2theta[smallest_2theta_position]
print("Scattering angle 2theta: maximum {:5.2f}degrees, minimum {:5.2f}degrees".format(np.rad2deg(largest_2theta),
                                                                                       np.rad2deg(smallest_2theta)))

crystal_plane_distance = 3.35e-10  # m
all_wavelength = get_wavelength(crystal_plane_distance=crystal_plane_distance, scattering_2theta=all_scattering_2theta,
                                order_parameter=1)
print("Wavelength: maximum {:5.2f}AA, minimum {:5.2f}AA".format(np.max(all_wavelength) * 1e10,
                                                                np.min(all_wavelength) * 1e10))
all_energy = get_energy_from_wavelength(all_wavelength)
print("energy: maximum {:5.2f}meV, minimum {:5.2f}meV".format(np.max(all_energy) * 1e3, np.min(all_energy) * 1e3))

plt.figure()

# ellipse = Ellipse(xy=(ellipse_centre_x, ellipse_centre_y), width=2 * focus, height=2 * semi_minor,
#                   angle=np.rad2deg(theta), fill=False)
ax = plt.gca()
ax.set_aspect('equal', 'box')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.add_patch(ellipse)
print("alpha={:5.2f}degrees".format(np.rad2deg(alpha)))

plt.plot(points_analyser_x, points_analyser_y, 'o', color='#1f77b4', linewidth=5)
plt.plot(-points_analyser_x, points_analyser_y, 'o', color='#1f77b4', linewidth=5)
plt.xlabel("x axis (m)")
plt.ylabel("y axis (m)")

point0 = [upside[0], upside[1]]
point1 = [points_analyser_up_x[0], points_analyser_up_y[0]]
point2 = [points_analyser_down_x[0], points_analyser_down_y[0]]

plt.plot(point0[0], point0[1], "ko")
plt.text(x=point0[0] + 0.1, y=point0[1], s=r"$E_{max}$=%5.2f meV" % (all_energy[smallest_2theta_position] * 1e3))

plt.plot(point1[0], point1[1], "ko")
plt.text(x=point1[0] + 0.1, y=point1[1], s=r"$E$=%5.2f meV" % (all_energy[0] * 1e3))

plt.plot(point2[0], point2[1], "ko")
plt.text(x=point2[0] + 0.1, y=point2[1], s=r"$E$=%5.2f meV" % (all_energy[-1] * 1e3))

line_sp0_plot = ([0, point0[0]], [0, point0[1]])
line_sp1_plot = ([0, point1[0]], [0, point1[1]])
line_sp2_plot = ([0, point2[0]], [0, point2[1]])

line_p0f = get_line_at_point(slope=slope, point=point0)
line_p1f = get_line_at_point(slope=slope, point=point1)
line_p2f = get_line_at_point(slope=slope, point=point2)

detector0 = get_intersect_two_lines(line_p0f, [1, 0, -0.5])
detector1 = get_intersect_two_lines(line_p1f, [1, 0, -0.5])
detector2 = get_intersect_two_lines(line_p2f, [1, 0, -0.5])

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

plt.text(0.55, -1.5, 'focus {:5.3f}m\nRotating Angle={:5.3f}°'.format(focus, np.rad2deg(alpha)))
plt.savefig('Mushroom_Geometry_Parabola.pdf')
