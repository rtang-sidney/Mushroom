import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.patches import Ellipse
from helper import wavelength_to_joule, wavelength_to_eV, points_distance, get_angle

ZERO_TOL = 1e-6

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082

"""


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
        return a, b, 1.0


class GeometryContext(object):
    def __init__(self):
        self.sample_point = (0.0, 0.0)  # m
        self.focus_point = (0.7, -0.7)  # m

        self.sample_size = 1e-2  # m
        self.focus_size = 4e-2  # m

        self.semi_major = 1.0  # m
        self.ellipse_number = 100  # number of points to form the ellipse

        self.angle_plus = np.deg2rad(
            50.)  # radian, the slope of the line from the sample to the upmost point on the analyser
        self.angle_minus = np.deg2rad(
            -10.)  # radian, the slope of the line from the sample to the downmost point on the analyser
        self.start_distance = 0.8  # m, the distance from the sample to the upmost point of the analyser

        # self._ellipse = (h, k, a)
        self.analyser_points = self._generate_analyser_segments()
        # self.analyser_ellipse_points = self._generate_analyser_ellipse()

        self.detector_line = [0.0, 1.0, -1.0]  # [0, 1, h]: h -> vertical position (m)
        self.detector_points = self._detector_from_analyser()

    def _ellipse_points_to_parameters(self):
        """
        gives the parameters of the general form of an ellipse after the transformation, in the form of
        A(x-h)^2 + B(x-h)(y-k) + C(y-k)^2 = 1
        :return: parameters A, B and C denoted by aa, bb, cc
        """
        h, k = np.array(self.focus_point) / 2.0
        a = self.semi_major
        phi = np.arctan(k / h)  # rotational angle of the ellipse, positive sign = anti-clockwise
        c = abs(h / np.cos(phi))  # linear eccentricity, giving the distance of a focus to the ellipse centre
        b = np.sqrt(np.subtract(x1=np.square(a), x2=np.square(c)))  # semi-minor axis

        # parameters of the ellipse after the rotation: aa = A, bb = B, cc = C
        # Ax^2 + Bxy + Cy^2 = 1
        aa = np.cos(phi) ** 2 / a ** 2 + np.sin(phi) ** 2 / b ** 2
        bb = 2 * np.cos(phi) * np.sin(phi) * (1. / a ** 2 - 1. / b ** 2)
        cc = np.sin(phi) ** 2 / a ** 2 + np.cos(phi) ** 2 / b ** 2
        return aa, bb, cc, h, k

    # def _analyser_edges(self, focus, a, detector_line):
    #     def intersect_on_ellipse(aa, bb, cc, h, k, m):
    #         # gives the intersect of one line, y = mx, with the ellipse described by the parameters (aa, bb, cc, h, k)
    #         polynomial_parameters = np.empty(3)
    #         polynomial_parameters[0] = aa + bb * m + cc * m ** 2
    #         polynomial_parameters[1] = -2 * aa * h - bb * (m * h + k) - 2 * cc * m * k
    #         polynomial_parameters[2] = aa * h ** 2 + bb * h * k + cc * k ** 2 - 1
    #         x = np.roots(polynomial_parameters)
    #         x = x[x > 0]
    #         if len(x) == 1:
    #             x = x[0]
    #         elif len(x) == 0:
    #             raise RuntimeError("No x-component of the point has been found.")
    #         else:
    #             raise RuntimeError("Too many values of the x-component have been found.")
    #         y = m * x
    #         if abs(aa * (x - h) ** 2 + bb * (x - h) * (y - k) + cc * (y - k) ** 2 - 1) < ZERO_TOL:
    #             return [x, y]
    #         else:
    #             raise RuntimeError("Something wrong when solving the ellipse edge points.")
    #
    #     aa, bb, cc, = self._ellipse_points_to_parameters()
    #
    #     edge_up = intersect_on_ellipse(aa, bb, cc, h, k, np.tan(np.deg2rad(50)))
    #     edge_down = intersect_on_ellipse(aa, bb, cc, h, k, np.tan(-np.deg2rad(10)))
    #     line_fd1 = get_line_through_points(focus, edge_up)
    #     line_fd2 = get_line_through_points(focus, edge_down)
    #     detector_in = get_intersect_two_lines(line_fd1, detector_line)
    #     detector_out = get_intersect_two_lines(line_fd2, detector_line)
    #     return edge_up, edge_down, detector_in, detector_out

    def _generate_analyser_segments(self, point_start=None):
        # generates the analyser with a finite segment size
        def vector_bisector(vector1, vector2):
            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
            return np.linalg.norm(vector1) * vector2 + np.linalg.norm(vector2) * vector1

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        if point_start is None:
            analyser_x = self.start_distance * np.cos(self.angle_plus)
            analyser_y = self.start_distance * np.sin(self.angle_plus)
        elif len(point_start) == 2:  # if the starting point is given by the edge point
            analyser_x = point_start[0]
            analyser_y = point_start[1]
        else:
            raise RuntimeError("Failed to generate the analyser segments from the given value.")

        point_now = [analyser_x, analyser_y]
        analyser_x = [analyser_x]
        analyser_y = [analyser_y]
        while self.angle_minus - 0.01 < np.arctan(point_now[1] / point_now[0]) < self.angle_plus + 0.01:
            vector_sa = points_to_vector(point1=self.sample_point, point2=point_now)
            vector_af = points_to_vector(point1=point_now, point2=self.focus_point)
            vector_tangential = vector_bisector(vector_sa, vector_af)
            segment_analyser = unit_vector(vector_tangential) * 1e-2  # the size of one segment is 1cm2
            point_now += segment_analyser  # update the next point
            analyser_x.append(point_now[0])
            analyser_y.append(point_now[1])
        return np.array(analyser_x), np.array(analyser_y)

    def _intersect_on_ellipse(self, m):
        # gives the intersect of a line y = mx, with the ellipse described by the parameters (aa, bb, cc, h, k)
        aa, bb, cc, h, k = self._ellipse_points_to_parameters()
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
        if abs(aa * (x - h) ** 2 + bb * (x - h) * (y - k) + cc * (y - k) ** 2 - 1) < ZERO_TOL:
            return [x, y]
        else:
            raise RuntimeError("Something wrong when solving the ellipse edge points.")

    def _generate_analyser_ellipse(self):
        # generate the analyser as a part of an ideal ellipse
        angles = np.linspace(self.angle_minus, self.angle_plus, num=self.ellipse_number)
        analyser_x = []
        analyser_y = []
        for angle in angles:
            slope = np.tan(angle)
            x, y = self._intersect_on_ellipse(m=slope)
            analyser_x.append(x)
            analyser_y.append(y)
        analyser_x = np.array(analyser_x)
        analyser_y = np.array(analyser_y)

        return np.array(analyser_x), np.array(analyser_y)

    def _detector_from_analyser(self):
        detector_x = []
        detector_y = []
        analyser_x, analyser_y = self.analyser_points
        for i in range(analyser_x.shape[0]):
            line_af = points_to_line(self.focus_point, [analyser_x[i], analyser_y[i]])
            detector_point = lines_intersect(line1=line_af, line2=self.detector_line)
            if abs(detector_point[1] - self.detector_line[2]) > ZERO_TOL:
                raise RuntimeError("Calculated detector points not on the detector line.")
            detector_x.append(detector_point[0])
            detector_y.append(detector_point[1])
        if detector_x[0] * detector_x[-1] < 0:
            raise RuntimeError("Detector points overlap.")
        return np.array(detector_x), np.array(detector_y)


class InstrumentContext(object):
    def __init__(self):
        self.moasic_analyser = np.deg2rad(0.4)  # radian, analyser mosaic
        self.deltad_d = 6e-4  # relative uncertainty of the lattice distance, given in [paper2]
        self.pg_lattice_distance = 3.35e-10  # m, lattive distance d of a PG crystal
        self.analyser_segment = 1e-2  # m, the size of an analyser segment in 1D


def points_to_vector(point1, point2):
    # gives the vector pointing from the point1 to point2 (direction important)
    return [point2[0] - point1[0], point2[1] - point1[1]]


def parameters_to_line(slope, y_intersect=None):
    # gives the line parameters in the form ax+by=c from the form y = ax+b where a is the slope and b is the y intersect
    if y_intersect is None:
        return slope, -1., 0.
    else:
        return slope, -1., -y_intersect


# Comment from Alex <3
# line: ax + by = c -> (a, b, c)
def lines_intersect(line1, line2):
    # gives the intersect of two lines
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


# def analyser_edges(focus, a, detector_line):
#     def intersect_on_ellipse(aa, bb, cc, h, k, m):
#         # gives the intersect of one line, y = mx, with the ellipse described by the parameters (aa, bb, cc, h, k)
#         polynomial_parameters = np.empty(3)
#         polynomial_parameters[0] = aa + bb * m + cc * m ** 2
#         polynomial_parameters[1] = -2 * aa * h - bb * (m * h + k) - 2 * cc * m * k
#         polynomial_parameters[2] = aa * h ** 2 + bb * h * k + cc * k ** 2 - 1
#         x = np.roots(polynomial_parameters)
#         x = x[x > 0]
#         if len(x) == 1:
#             x = x[0]
#         elif len(x) == 0:
#             raise RuntimeError("No x-component of the point has been found.")
#         else:
#             raise RuntimeError("Too many values of the x-component have been found.")
#         y = m * x
#         if abs(aa * (x - h) ** 2 + bb * (x - h) * (y - k) + cc * (y - k) ** 2 - 1) < ZERO_TOL:
#             return [x, y]
#         else:
#             raise RuntimeError("Something wrong when solving the ellipse edge points.")
#
#     h, k = focus / 2.0
#     aa, bb, cc, = get_ellipse_parameters(h, k, a)
#
#     edge_up = intersect_on_ellipse(aa, bb, cc, h, k, np.tan(np.deg2rad(50)))
#     edge_down = intersect_on_ellipse(aa, bb, cc, h, k, np.tan(-np.deg2rad(10)))
#     line_fd1 = get_line_through_points(focus, edge_up)
#     line_fd2 = get_line_through_points(focus, edge_down)
#     detector_in = get_intersect_two_lines(line_fd1, detector_line)
#     detector_out = get_intersect_two_lines(line_fd2, detector_line)
#     return edge_up, edge_down, detector_in, detector_out


# def get_analyser_point(point_y, a, h, k, edge):
#     def get_one_point_y(point_y, aa, bb, cc, h, k, edge):
#         polynomial_parameters = np.empty(3)
#         polynomial_parameters[0] = aa
#         polynomial_parameters[1] = bb * (point_y - k)
#         polynomial_parameters[2] = cc * (point_y - k) ** 2 - 1
#         x = np.roots(polynomial_parameters) + h
#         point_x = x[edge - 0.1 < x]
#         if len(point_x) == 1:
#             point_x = point_x[0]
#         elif len(point_x) == 0:
#             raise RuntimeError("No x-component of the point has been found, found {}.".format(x))
#         else:
#             raise RuntimeError("Too many values of the x-component have been found, found {}.".format(point_x))
#         return point_x
#
#     aa, bb, cc = get_ellipse_parameters(a=a, h=h, k=k)
#     if isinstance(point_y, float) is True:
#         return get_one_point_y(point_y, aa, bb, cc, h, k, edge)
#     else:
#         return np.array(list(map(lambda y: get_one_point_y(y, aa, bb, cc, h, k, edge), point_y)))


def wavelength_bragg(instrument: InstrumentContext, scattering_2theta, order_parameter=1):
    # gives the wavelength from the Bragg's law
    return 2. * instrument.pg_lattice_distance * np.sin(scattering_2theta / 2.) / float(order_parameter)


def wavenumber_bragg(instrument: InstrumentContext, scattering_2theta, order_parameter=1):
    # gives the wave number from the Bragg's law
    wavelength = wavelength_bragg(instrument=instrument, scattering_2theta=scattering_2theta,
                                  order_parameter=order_parameter)
    return 2. * np.pi / wavelength


def analyser_twotheta(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    vector_sa = points_to_vector(geo_ctx.sample_point, analyser_point)  # sa = sample_analyser
    vector_af = points_to_vector(analyser_point, geo_ctx.focus_point)  # af = analyser_focus
    return get_angle(vector_sa, vector_af)


# def get_analyser_angular_spread(geo_ctx: GeometryContext, sample, analyser_point, focus_point):
def angular_spread_analyser(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    eta = instrument.moasic_analyser  # mosaic
    alpha_i, alpha_f = divergence_analyser_point(geo_ctx,
                                                 analyser_point=analyser_point)  # incoming and outgoing divergence

    # See [Paper1]
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2

    return np.sqrt(numerator / denominator)

    # incoming_divergence, outgoing_divergence = \
    #     get_divergence(sample=sample, analyser_point=analyser_point,
    #                    focus=focus_point, sample_size=sample_size,
    #                    focus_size=focus_size)
    # return np.sqrt((
    #                        incoming_divergence ** 2 * outgoing_divergence ** 2 + mosaic_analyser ** 2 * incoming_divergence ** 2 + mosaic_analyser ** 2 * outgoing_divergence ** 2) / (
    #                        4 * mosaic_analyser ** 2 + incoming_divergence ** 2 + outgoing_divergence ** 2))


def delta_kf_bragg(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point, kf):
    # gives the deviation of the wave-number by means of the Bragg's law
    dtheta_analyser = angular_spread_analyser(geo_ctx, instrument, analyser_point=analyser_point)
    twotheta_analyser = analyser_twotheta(geo_ctx, instrument, analyser_point=analyser_point)
    delta_kf = kf * np.sqrt(
        np.sum(np.square([instrument.deltad_d, dtheta_analyser / np.tan(twotheta_analyser / 2.0)])))
    return delta_kf


def get_delta_phi(geo_ctx: GeometryContext, instrument: InstrumentContext, analyser_point):
    # TODO: Is this redundant? Since the the analyser angular spread itself depends on the beam divergence, too.
    #  This should probably depend on the detector spread
    analyser_divergence = instrument.analyser_segment / points_distance(geo_ctx.sample_point, analyser_point)
    # analyser_divergence = np.sqrt(
    #     np.sum(np.square([divergence_analyser_point(geo_ctx=geo_ctx, analyser_point=analyser_point)])))
    # dtheta_analyser = angular_spread_analyser(geo_ctx=geo_ctx, instrument=instrument, analyser_point=analyser_point)
    # return np.sqrt(np.sum(np.square([dtheta_analyser, analyser_divergence])))
    return analyser_divergence


def get_resolution_e(geo_ctx: GeometryContext, instrument: InstrumentContext, kf, analyser_point):
    delta_kf = delta_kf_bragg(geo_ctx, instrument, analyser_point=analyser_point, kf=kf)  # kf: outgoing wave number
    return 2. * delta_kf / kf


# def get_divergence(sample, analyser_point, focus, sample_size, focus_size):
def divergence_analyser_point(geo_ctx: GeometryContext, analyser_point):
    # sa: sample-analyser; af: analyser-focus
    distance_sa = points_distance(point1=geo_ctx.sample_point, point2=analyser_point)
    distance_af = points_distance(point1=analyser_point, point2=geo_ctx.focus_point)
    divergence_in = geo_ctx.sample_size / distance_sa
    divergence_out = geo_ctx.focus_size / distance_af
    return divergence_in, divergence_out

    # distance_sample_analyser = euclidean_metric(point1=sample, point2=analyser_point)
    # distance_analyser_focus = euclidean_metric(point1=analyser_point, point2=focus)
    # incoming_divergence = np.arctan(sample_size / distance_sample_analyser)
    # outgoing_divergence = np.arctan(focus_size / distance_analyser_focus)
    # return incoming_divergence, outgoing_divergence


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


def get_resolution_q(geo_ctx: GeometryContext, instrument: InstrumentContext, kf, theta, analyser_point, ki=None):
    """
    def get_delta_theta(sample, analyser_point, focus_point):
        sample_size = 1e-2  # m
        focus_size = 4e-2  # m
        incoming_divergence, outgoing_divergence = get_divergence(sample=sample, analyser_point=analyser_point,
                                                                  focus=focus_point, sample_size=sample_size,
                                                                  focus_size=focus_size)
        return np.sqrt(incoming_divergence ** 2 + outgoing_divergence ** 2)
    """

    # if ki is not specified, the scattering is treated to be elastic
    if ki is None:
        ki = kf

    qxy = np.sqrt(np.sum(np.square([kf * np.cos(phi) * np.cos(theta) - ki, kf * np.cos(phi) * np.sin(theta)])))
    # this is the denominator in the calculation later

    delta_kf = delta_kf_bragg(geo_ctx, instrument=instrument, analyser_point=analyser_point, kf=kf)
    delta_phi = get_delta_phi(geo_ctx, instrument=instrument, analyser_point=analyser_point)
    # delta_phi = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    dtheta_sample = np.sqrt(np.sum(np.square([divergence_analyser_point(geo_ctx, analyser_point=analyser_point)])))
    # dtheta_sample = instrument.analyser_segment / points_distance(geo_ctx.sample_point, analyser_point)
    # dtheta_sample is the deviation of the azimuthal angle of the scattering at the sample,
    # and it is calculated by the incoming and outgoing divergences of the beam

    # derivatives in the form of: function_variable
    qxy_kf = np.cos(phi) * (kf * np.cos(phi) - ki * np.cos(theta)) / qxy
    qxy_phi = -kf * np.sin(phi) * (kf * np.cos(phi) - ki * np.cos(theta)) / qxy
    qxy_theta = ki * kf * np.cos(phi) * np.sin(theta) / qxy

    delta_qxy = np.sqrt(np.sum(np.square([qxy_kf * delta_kf, qxy_phi * delta_phi, qxy_theta * dtheta_sample])))

    # derivatives in the form of: function_variable
    qz_kf = np.sin(phi)
    qz_phi = kf * np.cos(phi)
    delta_qz = np.sqrt(np.sum(np.square([qz_kf * delta_kf, qz_phi * delta_phi])))

    return delta_qxy, delta_qz


def get_q_vector(outgoing_k):
    """
    to calculate the q-vector of the scattering at the sample, with the incoming wave vector defined along the x-axis.
    The elastic scattering is considered in this case because the q-resolution independent of the energy will be +
    calculated based on the result from this step
    :param outgoing_k: k_f as a vector in 3D
    :return: wave vector transfer as a vector in 3D
    """
    if np.shape(outgoing_k)[0] != 3:
        raise RuntimeError("Given incoming wave vector {} invalid.".format(outgoing_k))
    incoming_k = np.array([np.linalg.norm(outgoing_k), 0, 0])  # k_i is along x-axis and has the same magnitude as k_f
    q_vector = outgoing_k - incoming_k
    return q_vector


def get_kf(kf_norm, azimuthal, polar):
    """
    to calculate the full vector of k_f (the wave vector after scattering at the sample and before the analyser)
    :param kf_norm: the norm of k_f
    :param azimuthal: azimuthal angle theta, which is half of the scattering angle at the sample
    :param polar: polar angle phi
    :return: k_f vector with its components in all three dimensions
    """
    if not isinstance(kf_norm, float):
        raise RuntimeError("Wrong type of kf given")
    if not isinstance(azimuthal, float):
        raise RuntimeError("Wrong type of azimuthal angle given")
    if not isinstance(polar, float):
        raise RuntimeError("Wrong type of polar angle given")

    kf = np.array([np.cos(polar) * np.cos(azimuthal), np.cos(polar) * np.sin(azimuthal), np.sin(polar)])
    kf *= kf_norm
    return kf


def get_parameters(side):
    if side == "same":
        # focus = [0.7, -0.7]
        filename_horizontal = 'Q_resolution_Horizontal.pdf'
        filename_vertical = 'Q_resolution_Vertical.pdf'
        plot_ylim = [-1.3, 0.9]
    elif side == "oppisite":
        # focus = [0.15, -0.45]
        filename_horizontal = 'Q_resolution_Horizontal2.pdf'
        filename_vertical = 'Q_resolution_Vertical2.pdf'
        plot_ylim = [-1.1, 0.9]
    else:
        raise RuntimeError("Given information invalid".format(side))
    return plot_ylim, filename_horizontal, filename_vertical
    # return focus, plot_ylim, filename_horizontal, filename_vertical


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
    plt.text(0.3, -0.35, "Largest deviation from the ideal ellipse: {:5.2f} m".format(np.linalg.norm(
        points_to_vector([points_x[-1], points_y[-1]], [points_analyser_x[-1], points_analyser_y[-1]]))))
    plt.xlabel("x axis (m)")
    plt.ylabel("y axis (m)")
    plt.plot(*geometryctx.sample_point, "ro")
    plt.plot(*geometryctx.focus_point, "ro")
    plt.text(x=0, y=-0.05, s="Sample")
    plt.text(x=0.1, y=-0.4, s="Focus")
    plt.savefig("analyser_geometry.pdf", bbox_inches='tight')
    plt.close(10)


def get_resolution_robbewley(geo_ctx: GeometryContext, instrument: InstrumentContext, all_qxy, all_qz):
    analyser_x, analyser_y = geo_ctx.analyser_points
    detector_x, detector_y = geo_ctx.detector_points
    if analyser_x.shape[0] != detector_x.shape[0]:
        raise RuntimeError("Analyser and detector points have different sizes, {:d}, {:d}".format(analyser_x.shape[0],
                                                                                                  detector_x.shape[0]))
    all_delta_qxy = []
    all_delta_qz = []
    for j in range(len(azimuthal_angles)):
        for i in range(analyser_x.shape[0]):
            k = j * (analyser_x.shape[0]) + i
            analyser_point = np.array([analyser_x[i], analyser_y[i]])
            detector_point = np.array([detector_x[i], detector_y[i]])

            distance_ad = np.linalg.norm(analyser_point - detector_point)
            dx = np.sqrt((np.tan(2 * instrument.moasic_analyser) * distance_ad) ** 2 + instrument.analyser_segment ** 2)
            x = detector_x[i]
            thi = abs(dx / x)
            dtheta = abs(instrument.analyser_segment / np.linalg.norm(analyser_point - geo_ctx.sample_point))
            if j == 0:
                dqxy = all_qxy[k] - all_qxy[k + analyser_x.shape[0]]
            else:
                dqxy = all_qxy[k] - all_qxy[k - analyser_x.shape[0]]
            dqxy = abs(dqxy)
            delta_qxy = dqxy * thi / dtheta
            all_delta_qxy.append(delta_qxy)

            vector_ad = points_to_vector(analyser_point, detector_point)
            theta0 = np.arctan(abs(vector_ad[1] / vector_ad[0]))
            dxy = dx / np.sin(theta0)
            x_spread = abs(dxy)
            if i == 0:
                next_point = np.array([detector_x[i + 1], detector_y[i + 1]])
                x_point = abs(detector_point[0] - next_point[0])
                dqz = all_qz[k] - all_qz[k + 1]

            else:
                last_point = np.array([detector_x[i - 1], detector_y[i - 1]])
                x_point = abs(detector_point[0] - last_point[0])
                dqz = all_qz[k] - all_qz[k - 1]
            dqz = abs(dqz)
            delta_qz = dqz * x_spread / x_point
            all_delta_qz.append(delta_qz)
    # print(len(all_delta_qxy), len(all_delta_qz))
    return np.array(all_delta_qxy), np.array(all_delta_qz)


geometryctx = GeometryContext()
instrumentctx = InstrumentContext()

# sample = [0, 0]
# semi_major = 1.0  # semi-major axis of the ellipse
side = "same"
# side = "opposite"
# focus, plot_ylim, filename_horizontal, filename_vertical, = get_parameters(side=side)
plot_ylim, filename_horizontal, filename_vertical, = get_parameters(side=side)
# ellipse_centre_x = focus[0] / 2.0
# ellipse_centre_y = focus[1] / 2.0
#
# edge_up, edge_down, detector_in, detector_out = analyser_edges(focus=np.array(focus), a=semi_major,
#                                                                detector_line=detector_line)
# print("Edge points", edge_up, edge_down)

# to generate the segments of 1x1 cm2 on the analyser
points_x, points_y = geometryctx.analyser_points

# to generate the points on the analyser ideally from the ellipse equation
# points_analyser_y = np.linspace(edge_up[1], edge_down[1], number_points)
# points_analyser_x = get_analyser_point(points_analyser_y, a=semi_major, h=ellipse_centre_x, k=ellipse_centre_y,
#                                        edge=edge_up[0])

# plot_analyser_comparison(points_x=points_x, points_y=points_y, points_analyser_x=points_analyser_x,
#                          points_analyser_y=points_analyser_y)

# to calculate the scattering angle 2theta_A for each point on the analyser
all_scattering_2theta = np.array(
    list(
        map(lambda x, y: analyser_twotheta(geo_ctx=geometryctx, instrument=instrumentctx,
                                           analyser_point=[x, y]),
            points_x, points_y)))

largest_2theta_position = np.argmax(all_scattering_2theta)
largest_2theta = all_scattering_2theta[largest_2theta_position]
smallest_2theta_position = np.argmin(all_scattering_2theta)
smallest_2theta = all_scattering_2theta[smallest_2theta_position]
print("Scattering angle 2theta: maximum {:5.2f}degrees, minimum {:5.2f}degrees".format(np.rad2deg(largest_2theta),
                                                                                       np.rad2deg(smallest_2theta)))

# to calculate the wavelength and energy
all_wavelength = wavelength_bragg(instrument=instrumentctx, scattering_2theta=all_scattering_2theta)
print("Wavelength: maximum {:5.2f}AA, minimum {:5.2f}AA".format(np.max(all_wavelength) * 1e10,
                                                                np.min(all_wavelength) * 1e10))

all_energy_SI, all_energy_eV = wavelength_to_joule(all_wavelength), wavelength_to_eV(all_wavelength)
print("energy: maximum {:5.2f}meV, minimum {:5.2f}meV".format(np.max(all_energy_eV) * 1e3, np.min(all_energy_eV) * 1e3))

plt.figure(1)
ax = plt.gca()
ax.set_aspect('equal', 'box')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.add_patch(ellipse)

plt.xlim(-2, 2)
plt.ylim(*plot_ylim)

# plt.plot(points_analyser_x, points_analyser_y, color='#1f77b4', linewidth=5)
# plt.plot(-points_analyser_x, points_analyser_y, color='#1f77b4', linewidth=5)
# plt.xlabel("x axis (m)")
# plt.ylabel("y axis (m)")

# point0 = [points_analyser_x[smallest_2theta_position], points_analyser_y[smallest_2theta_position]]
# point1 = [points_analyser_x[0], points_analyser_y[0]]
# point2 = points_analyser_x[-1], points_analyser_y[-1]
#
# plt.text(x=-0.7, y=0.75, s=r"$E$(meV)")
# plt.text(x=0.5, y=0.75, s=r"$\Delta E$($\mu$eV)")

# resolution_e0 = get_resolution_e(kf=bragg_condition_wavenumber(scattering_2theta=smallest_2theta),
#                                  analyser_point=point0, focus_point=focus, sample=sample)
# resolution_e1 = get_resolution_e(kf=bragg_condition_wavenumber(scattering_2theta=all_scattering_2theta[0]),
#                                  analyser_point=point0, focus_point=focus, sample=sample)
# resolution_e2 = get_resolution_e(kf=bragg_condition_wavenumber(scattering_2theta=all_scattering_2theta[-1]),
#                                  analyser_point=point0, focus_point=focus, sample=sample)

# plt.plot(point0[0], point0[1], "ko")
# plt.text(x=-point0[0] - 0.3, y=point0[1], s="{:5.2f}".format(all_energy_eV[smallest_2theta_position] * 1e3))
# plt.text(x=point0[0] + 0.1, y=point0[1],
#          s="{:5.2f}".format(all_energy_eV[smallest_2theta_position] * resolution_e0 * 1e6))
#
# plt.plot(point1[0], point1[1], "ko")
# plt.text(x=-point1[0] - 0.3, y=point1[1], s="{:5.2f}".format(all_energy_eV[0] * 1e3))
# plt.text(x=point1[0] + 0.1, y=point1[1], s="{:5.2f}".format(all_energy_eV[0] * resolution_e1 * 1e6))
#
# plt.plot(point2[0], point2[1], "ko")
# plt.text(x=-point2[0] - 0.3, y=point2[1], s="{:5.2f}".format(all_energy_eV[-1] * 1e3))
# plt.text(x=point2[0] + 0.1, y=point2[1], s="{:5.2f}".format(all_energy_eV[-1] * resolution_e2 * 1e6))
#
# line_sp0_plot = ([0, point0[0]], [0, point0[1]])
# line_sp1_plot = ([0, point1[0]], [0, point1[1]])
# line_sp2_plot = ([0, point2[0]], [0, point2[1]])
#
# line_p0f = get_line_through_points(point0, focus)
# line_p1f = get_line_through_points(point1, focus)
# line_p2f = get_line_through_points(point2, focus)
#
# detector0 = get_intersect_two_lines(line_p0f, detector_line)
#
# line_p0f_plot = ([point0[0], detector0[0]], [point0[1], detector0[1]])
# line_p1f_plot = ([point1[0], detector_in[0]], [point1[1], detector_in[1]])
# line_p2f_plot = ([point2[0], detector_out[0]], [point2[1], detector_out[1]])
#
# line_sp0_plot2 = ([0, -point0[0]], [0, point0[1]])
# line_sp1_plot2 = ([0, -point1[0]], [0, point1[1]])
# line_sp2_plot2 = ([0, -point2[0]], [0, point2[1]])
# line_p0f_plot2 = ([-point0[0], -detector0[0]], [point0[1], detector0[1]])
# line_p1f_plot2 = ([-point1[0], -detector_in[0]], [point1[1], detector_in[1]])
# line_p2f_plot2 = ([-point2[0], -detector_out[0]], [point2[1], detector_out[1]])
#
# plt.plot(*line_sp0_plot, color='#17becf')
# plt.plot(*line_sp1_plot, color='#17becf')
# plt.plot(*line_sp2_plot, color='#17becf')
# plt.plot(*line_p0f_plot, color='#17becf')
# plt.plot(*line_p1f_plot, color='#17becf')
# plt.plot(*line_p2f_plot, color='#17becf')
#
# plt.plot(*line_sp0_plot2, color='#17becf')
# plt.plot(*line_sp1_plot2, color='#17becf')
# plt.plot(*line_sp2_plot2, color='#17becf')
# plt.plot(*line_p0f_plot2, color='#17becf')
# plt.plot(*line_p1f_plot2, color='#17becf')
# plt.plot(*line_p2f_plot2, color='#17becf')
#
# plt.plot([detector_in[0], detector_out[0]], [detector_in[1], detector_out[1]], color='#8c564b', linewidth=5)
# plt.plot([-detector_in[0], -detector_out[0]], [detector_in[1], detector_out[1]], color='#8c564b', linewidth=5)
#
# plt.plot(0, 0, "ro")
# plt.text(x=-0.275, y=-0.25, s="Sample", fontsize=15)
#
# plt.tight_layout()
# plt.savefig('Mushroom_Geometry_OppsiteSide.pdf', bbox_inches='tight')

# generates the azimuthal angle elements based on the size of the analyser segments
angle_one_segment = np.arcsin(instrumentctx.analyser_segment / geometryctx.start_distance)
azimuthal_start = np.deg2rad(5.)  # radian
azimuthal_stop = np.deg2rad(170.)  # radian
number_points = round(abs(azimuthal_start - azimuthal_stop / angle_one_segment))
azimuthal_angles = np.linspace(azimuthal_start, azimuthal_stop, num=number_points)

# records the calculated data for plotting later
all_qxy = []
all_qz = []
all_dqxy = []
all_dqz = []

# calculate the q-resolution for each segment on the analyser, which gives different q-vectors

for theta in azimuthal_angles:
    for i in range(len(points_x)):
        kf = wavenumber_bragg(instrument=instrumentctx, scattering_2theta=all_scattering_2theta[i])
        x = points_x[i]
        y = points_y[i]
        phi = np.arctan(y / x)  # the polar angle of one point
        kf_vector = get_kf(kf_norm=kf, azimuthal=theta,
                           polar=phi)  # the kf-vector changes is determined by the azimuthal and polar angles
        q_vector = get_q_vector(kf_vector)  # q-vector is determined by the kf-vector
        qxy = np.sqrt(np.sum(np.square(q_vector[:2])))  # horizontal component of q-vector
        qz = q_vector[2]  # vertical component of q-vector
        delta_qxy, delta_qz = get_resolution_q(analyser_point=[x, y], geo_ctx=geometryctx, instrument=instrumentctx,
                                               theta=theta, kf=kf)
        all_qxy.append(qxy)
        all_qz.append(qz)
        all_dqxy.append(delta_qxy)
        all_dqz.append(delta_qz)

all_qxy = np.array(all_qxy)
all_qz = np.array(all_qz)
all_dqxy = np.array(all_dqxy)
all_dqz = np.array(all_dqz)

# plot the geometry of the analyser, detector, sample and focus
plt.figure(1)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(points_x, points_y)
plt.legend(r"Segments with 1x1 cm$^2$")
plt.text(0.3, -0.3, "Number of segments in one cut-plane: {:d}".format(len(points_x)))
plt.xlabel("x axis (m)")
plt.ylabel("y axis (m)")
plt.plot(*geometryctx.sample_point, "ro")
plt.plot(*geometryctx.focus_point, "ro")
plt.plot(*geometryctx.detector_points)
plt.text(x=0, y=-0.05, s="Sample")
plt.text(x=0.1, y=-0.4, s="Focus")
plt.savefig("analyser_geometry_check.pdf", bbox_inches='tight')
plt.close(1)

# plot the horizontal component of the q-resolution calculated by us
plt.figure(2)
plt.plot(all_qxy * 1e-10, all_dqxy * 1e-10, '.')
plt.xlabel(r"$Q_{xy}$ (Angstrom -1)")
plt.ylabel(r"$\Delta Q_{xy}$ (Angstrom -1)")
plt.title("Q resolution - horizontal")
plt.grid()
plt.savefig(filename_horizontal, bbox_inches='tight')
plt.close(2)

# plot the vertical component of the q-resolution calculated by us
plt.figure(3)
plt.plot(all_qz * 1e-10, all_dqz * 1e-10, '.')
plt.xlabel(r"$Q_{z}$ (Angstrom -1)")
plt.ylabel(r"$\Delta Q_{z}$ (Angstrom -1)")
plt.title("Q resolution - vertical")
plt.grid()
plt.savefig(filename_vertical, bbox_inches='tight')
plt.close(3)

all_delta_qxy_rob, all_delta_qz_rob = get_resolution_robbewley(geo_ctx=geometryctx, instrument=instrumentctx,
                                                               all_qxy=all_qxy, all_qz=all_qz)

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
