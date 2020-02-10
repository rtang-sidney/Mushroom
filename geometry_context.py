import numpy as np

from helper import points_distance, vector_bisector, ZERO_TOL, points_to_line, points_to_vector, lines_intersect


class GeometryContext(object):
    def __init__(self, side="same"):
        self.sample_point = (0.0, 0.0)  # m

        self.sample_size = 4e-2  # m
        self.focus_size = 4e-2  # m

        self.ellipse_number = 100  # number of points to form the ideal ellipse

        self.angle_plus = np.deg2rad(
            50.)  # radian, the slope of the line from the sample to the upmost point on the analyser
        self.angle_minus = np.deg2rad(
            -10.)  # radian, the slope of the line from the sample to the downmost point on the analyser
        self.start_distance = 0.4 * 2.0 ** 0.5  # m, the distance from the sample to the upmost point of the analyser
        self.start_point = [self.start_distance * np.cos(self.angle_plus),
                            self.start_distance * np.sin(self.angle_plus)]

        if side == "same":
            self.focus_point = (0.90, -0.25)  # m
            self.filename_geometry = 'Geometry_SameSide2.pdf'
            self.filename_horizontal = 'QResolution_Horizontal_SameSide.pdf'
            self.filename_vertical = 'QResolution_Vertical_SameSide.pdf'
        elif side == "opposite":
            self.focus_point = (0.15, -0.4)  # m
            self.filename_geometry = 'Geometry_OppositeSide.pdf'
            self.filename_horizontal = 'QResolution_Horizontal_OppositeSide.pdf'
            self.filename_vertical = 'QResolution_Vertical_OppositeSide.pdf'
        else:
            raise RuntimeError("Given information invalid".format(side))

        self.semi_major = (points_distance(self.sample_point, self.start_point) + points_distance(
            self.focus_point, self.start_point)) / 2.0

        self.analyser_points = self._generate_analyser_segments()

        # if the analyser is generated as a part of an ideal ellipse:
        self.analyser_ellipse_points = self._generate_analyser_ellipse()

        self.detector_line1 = [0.0, 1.0, -0.9]  # [0, 1, v]: v -> vertical position (m) of the horizontal bank
        self.detector_line2 = [1.0, 0.0, 0.7]  # [1, 0, h]: h -> horizontal position (m) of the vertical bank
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
        b = np.sqrt(a ** 2 - c ** 2)  # semi-minor axis

        # parameters of the ellipse after the rotation: aa = A, bb = B, cc = C
        # Ax^2 + Bxy + Cy^2 = 1
        aa = np.cos(phi) ** 2 / a ** 2 + np.sin(phi) ** 2 / b ** 2
        bb = 2 * np.cos(phi) * np.sin(phi) * (1. / a ** 2 - 1. / b ** 2)
        cc = np.sin(phi) ** 2 / a ** 2 + np.cos(phi) ** 2 / b ** 2
        return aa, bb, cc, h, k

    def _generate_analyser_segments(self):
        # generates the analyser with a finite segment size

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        point_now = self.start_point
        analyser_x = [self.start_point[0]]
        analyser_y = [self.start_point[1]]
        while self.angle_minus - np.deg2rad(0.1) < np.arctan(
                point_now[1] / point_now[0]) < self.angle_plus + np.deg2rad(0.1):
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
        angles = np.linspace(self.angle_plus, self.angle_minus, num=self.ellipse_number)
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
            detector_point = lines_intersect(line1=line_af, line2=self.detector_line1)
            if detector_point[0] - self.detector_line2[2] < - ZERO_TOL:
                detector_point = lines_intersect(line1=line_af, line2=self.detector_line2)
                if detector_point[1] - self.detector_line1[2] < - ZERO_TOL:
                    raise RuntimeError("Failed to find the detector point.")
            detector_x.append(detector_point[0])
            detector_y.append(detector_point[1])
        if detector_x[0] * detector_x[-1] < 0:
            raise RuntimeError("Detector points overlap.")
        return np.array(detector_x), np.array(detector_y)