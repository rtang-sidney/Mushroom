import numpy as np
import matplotlib.pyplot as plt

ZERO_TOL = 1e-6
MASS_NEUTRON = 1.67492749804e-27  # kg
PLANCKS_CONSTANT = 1.0545718e-34  # m2 kg / s, h-bar
CONVERSION_JOULE_PER_EV = 1.602176634e-19  # J / eV
BOLTZMANN = 1.380649e-23  # J / K
FACTOR_GAMMA = 1.913  # dimensionless constant
THOMSON_SCATT_LENGTH = 2.818e-15  # m, classical radius of electrons


class InstrumentContext(object):
    def __init__(self):
        self.moasic_pg002 = np.deg2rad(0.4)  # radian, PG crystal mosaic
        self.moasic_analyser = np.deg2rad(0.4)  # radian, analyser mosaic
        self.deltad_d = 6e-4  # relative uncertainty of the lattice distance, given in [paper2]
        self.lattice_distance_pg002 = 3.35e-10  # m, lattice distance d of a PG crystal
        self.an_seg = 1e-2  # m, the size of an analyser segment in 1D
        self.distance_ms = 1.0  # m, monochromator-sample distance
        self.divergence_initial = np.deg2rad(1.6)  # initial divergence directly from the neutron guide
        self.sample_diameter = 1e-2  # m
        self.sample_height = 1e-2  # m
        self.detector_resolution = 1e-2  # m, the positional resolution of the position-sensitive detectors


def wavelength_to_wavenumber(wavelength):
    return 2.0 * np.pi / wavelength


def wavelength_to_joule(wavelength):
    return 2 * np.pi ** 2 * PLANCKS_CONSTANT ** 2 / (MASS_NEUTRON * wavelength ** 2)


def wavelength_to_eV(wavelength):
    return wavelength_to_joule(wavelength) / CONVERSION_JOULE_PER_EV


def points_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def angle_vectors(vector1, vector2):
    """
    Returns the angle between vector1 and vector2 in radians

    :param vector1:
    :param vector2:
    :return:
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def vector_bisector(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.linalg.norm(vector1) * vector2 + np.linalg.norm(vector2) * vector1


def points_bisecting_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if abs(y1 - y2) < ZERO_TOL:
        return 0, 1, -y3
    else:
        m = -(x2 - x1) / (y2 - y1)
        return m, -1, y3 - m * x3


def points_to_line(point1, point2):
    def get_line_through_origin(point):
        # gives the line parameters if the line goes through the origin (0,0)
        if abs(point[0]) < ZERO_TOL:  # when the other point is on the y-axis, too
            return 0.0, 1.0, 0.0
        else:  # finite slope
            return point[1] / point[0], -1.0, 0.0

    if np.linalg.norm(point1) < ZERO_TOL:
        return get_line_through_origin(point2)
    elif np.linalg.norm(point2) < ZERO_TOL:
        return get_line_through_origin(point1)
    else:  # if no point is at the origin
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        matrix = np.array([[x1, y1], [x2, y2]])
        array = np.full(2, 1.0)
        a, b = np.linalg.solve(matrix, array)
        return a, b, -1.0


def points_to_vector(point1, point2):
    # gives the vector pointing from the point1 to point2 (direction important)
    return np.array([point2[0] - point1[0], point2[1] - point1[1]])


def parameters_to_line(slope, y_intersect=None):
    # gives the line parameters in the form ax+by+c=0 from the form y=ax+b where a is the slope and b is the y intersect
    if y_intersect is None:
        return slope, -1., 0.
    else:
        return slope, -1., y_intersect


def line_to_y(x, line_params):
    a, b, c = line_params
    print(a, b, c)
    if abs(b) < ZERO_TOL:
        raise RuntimeError("Given parameters define a line parallel to the y-axis.")
    else:
        return -(a * x + c) / b


def lines_intersect(line1, line2):
    # gives the intersect of two lines
    if len(line1) == 3 and len(line2) == 3:
        a1 = line1[0]
        b1 = line1[1]
        a2 = line2[0]
        b2 = line2[1]
        matrix = np.array([[a1, b1], [a2, b2]])
        array = -np.array([line1[-1], line2[-1]])
        x1, x2 = np.linalg.solve(matrix, array)
        return x1, x2
    else:
        raise RuntimeError("The line parameters provided are not valid. Try again.")


def angle_triangle(a, c, b=None):
    if b is None:
        b = a
    return np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))


def points_to_slope_radian(point1, point2):
    vector12 = points_to_vector(point1=point1, point2=point2)
    if abs(vector12[0]) > ZERO_TOL:
        return np.arctan(vector12[1] / vector12[0])
    else:
        return np.pi / 2.0


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def vector_project_a2b(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / np.linalg.norm(vector_b) ** 2 * vector_b


def vector_rejection(vector, vector_projection):
    return vector - vector_projection


def deg2min(angle_in_degree):
    if 0 < angle_in_degree < 1:
        return 60 * angle_in_degree
    else:
        raise ValueError("Invalid angle given")


def distance_point2line(point, line):
    # gives the distance from a point (x0,y0) to a line ax+by+c=0
    # works also for multiple points with points=(x_values, y_values), where both x and y values are numpy arrays with
    # the same length
    x0, y0 = point
    a, b, c = line
    return np.abs((a * x0 + b * y0 + c) / np.linalg.norm([a, b]))


def dispersion_signal(range_x, range_y, data_x, data_y, intensity):
    print(range_x.shape, range_y.shape, data_x.shape, data_y.shape, intensity.shape)
    inten_new_2d = np.full((range_y.shape[0], range_x.shape[0]), None)
    x_index = np.searchsorted(range_x, data_x)
    y_index = np.searchsorted(range_y, data_y)
    if len(x_index.shape) == len(y_index.shape) == 2:
        pass
    elif len(x_index.shape) == 1 and len(y_index.shape) == 2:
        x_index = np.array(list(map(lambda x: x_index, range(y_index.shape[0]))))
    elif len(x_index.shape) == 2 and len(y_index.shape) == 1:
        y_index = np.array(list(map(lambda x: np.full(x_index.shape[1], x), y_index)))
    elif len(x_index.shape) == len(y_index.shape) == 1:
        pass
    else:
        raise RuntimeError("Invalid shapes of x and y data given.")
    # inten_new_2d[y_index, x_index] = 0
    # np.add.at(inten_new_2d, (y_index, x_index), intensity)
    inten_new_2d[y_index, x_index] = intensity

    # x_index, y_index = np.meshgrid(x_index, y_index)
    return inten_new_2d


def data2range(data, number_points=None):
    if number_points is None:
        number_points = data.shape[0]
    return np.linspace(np.min(data), np.max(data), num=number_points)


def wavenumber_vector(wavenumber, azi_angle, pol_angle):
    return wavenumber * np.array(
        [np.cos(pol_angle) * np.cos(azi_angle), np.cos(pol_angle) * np.sin(azi_angle), np.sin(pol_angle)])


def plotting_format(ax, grid=True):
    plt.rcParams.update({'font.size': 15})
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    if grid is True:
        ax.grid()


def rotation_z(rot_angle, old_x, old_y):
    new_x = old_x * np.cos(rot_angle) - old_y * np.sin(rot_angle)
    new_y = old_x * np.sin(rot_angle) + old_y * np.cos(rot_angle)
    return new_x, new_y


def dirac_delta_approx(x, x0, resol):
    """
    approximates the delta dirac function with a Gaussian function with expectation value of x0 and with the limit that
    the variance->0, where variance sigma = resol*x0

    :param x: independet variable x
    :param x0: expectation value x0
    :param resol: relative variance, giving the variance sigma = resol*x0
    :return: approximated delta dirac funtion by Gaussian
    """
    a = resol * x0
    return np.exp(-((x - x0) / a) ** 2) / (abs(a) * np.sqrt(np.pi))
