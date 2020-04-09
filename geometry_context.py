import numpy as np

from helper import points_distance, vector_bisector, ZERO_TOL, points_to_line, points_to_vector, lines_intersect, \
    unit_vector, InstrumentContext


class GeometryContext(object):
    def __init__(self, side="same"):
        self.sample_point = (0.0, 0.0)  # m

        self.focus_size = 1e-2  # m

        self.ellipse_number = 100  # number of points to form the ideal ellipse

        self.angle_plus = np.deg2rad(
            50.)  # radian, the slope of the line from the sample to the upmost point on the analyser
        self.angle_minus = np.deg2rad(
            -10.)  # radian, the slope of the line from the sample to the downmost point on the analyser
        self.start_distance = 0.8  # m, the distance from the sample to the upmost point of the analyser
        self.start_point = [self.start_distance * np.cos(self.angle_plus),
                            self.start_distance * np.sin(self.angle_plus)]

        # self.detector_line1 = [0.0, 1.0, -1.6]  # [0, 1, v]: v -> vertical position (m) of the horizontal bank
        # self.detector_line2 = [1.0, 0.0, 0.4]  # [1, 0, h]: h -> horizontal position (m) of the vertical bank
        self.detector_line1 = [0.0, 1.0, -1.0]  # [0, 1, v]: v -> vertical position (m) of the horizontal bank
        self.detector_line2 = [1.0, 0.0, 0.6]  # [1, 0, h]: h -> horizontal position (m) of the vertical bank
        detector_suffix = '_{:2.1f}_{:2.1f}'.format(abs(self.detector_line2[2]), abs(self.detector_line1[2]))

        if side == "same":
            self.focus_point = (0.9, -0.4)  # m
            side_suffix = '_SameSide'
            # self.focus_point = (0.85, -0.85)
        elif side == "opposite":
            self.focus_point = (0.15, -0.4)  # m
            side_suffix = 'OppositeSide'
        else:
            raise RuntimeError("Given information invalid".format(side))
        self.filename_geometry = 'Geometry' + side_suffix + detector_suffix
        self.filename_horizontal = 'QResolution_Horizontal' + side_suffix + detector_suffix
        self.filename_vertical = 'QResolution_Vertical' + side_suffix + detector_suffix
        self.filename_polarangle = 'Resolution_PolarAngles' + detector_suffix

        self.semi_major = (points_distance(self.sample_point, self.start_point) + points_distance(
            self.focus_point, self.start_point)) / 2.0

        self.analyser_segment_size = 1e-2  # m
        instrumentctx = InstrumentContext()
        points_x, points_y, orientations_x, orientations_y = self._generate_analyser_segments(instrument=instrumentctx)
        self.analyser_points = (points_x, points_y)
        self.analyser_orientations = (orientations_x, orientations_y)

        # if the analyser is generated as a part of an ideal ellipse:
        self.analyser_ellipse_points = self._generate_analyser_ellipse()
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

    def _analyser_segment_orientation(self, analyser_point):
        """
        gives the unit vector denoting the orientation of an analyser segment at a given point
        :param analyser_point: a 2-D list or numpy array
        :return: a unit vector as a 2-D list
        """
        vector_sa = points_to_vector(point1=self.sample_point, point2=analyser_point)
        vector_af = points_to_vector(point1=analyser_point, point2=self.focus_point)
        vector_tangential = vector_bisector(vector_sa, vector_af)
        return unit_vector(vector_tangential)

    def _generate_analyser_segments(self, instrument: InstrumentContext):
        # generates the analyser with a finite segment size

        point_now = self.start_point
        analyser_x = [self.start_point[0]]
        analyser_y = [self.start_point[1]]
        orientation_now = self._analyser_segment_orientation(point_now)
        analyser_orientation_x = [orientation_now[0]]
        analyser_orientation_y = [orientation_now[1]]

        while self.angle_minus - np.deg2rad(0.1) < np.arctan(
                point_now[1] / point_now[0]) < self.angle_plus + np.deg2rad(0.1):
            segment_analyser = orientation_now * instrument.analyser_segment
            point_now += segment_analyser  # update the next point
            orientation_now = self._analyser_segment_orientation(point_now)
            analyser_x.append(point_now[0])
            analyser_y.append(point_now[1])
            analyser_orientation_x.append(orientation_now[0])
            analyser_orientation_y.append(orientation_now[1])
        return np.array(analyser_x), np.array(analyser_y), np.array(analyser_orientation_x), np.array(
            analyser_orientation_y)

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
        analyser_x, analyser_y = self.analyser_points[:2]
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
