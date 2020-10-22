import numpy as np
from helper import points_distance, vector_bisector, ZERO_TOL, points_to_line, points_to_vector, lines_intersect, \
    unit_vector, InstrumentContext, angle_vectors, wavenumber_vector, rotation_z

SOLAR_RANGE = np.deg2rad(60)  # the range of the solar angles covered by the analyser
CRYO_RADIUS = 0.4  # m, reserving the space for a cryostat
HEIGHT_INSTR = 3.0  # m, from the highest point of the analyser to the horizontal bank of PSD


class GeometryContext(object):
    def __init__(self, angle_plus_deg=30, distance_sf=2.16, polar_focus_deg=-35.17, detector_height=2.49,
                 wavenumber_in=1.5 * 1e10):
        self.wavenumber_in = wavenumber_in  # m-1
        self.azi_nega = np.deg2rad(np.linspace(-170, -5, num=166))
        self.azi_posi = np.deg2rad(np.linspace(5, 170, num=166))
        self.azi_angles = np.append(self.azi_nega, self.azi_posi)

        instrumentctx = InstrumentContext()

        self.sample_point = (0.0, 0.0)  # m
        self.foc_size = 1e-2  # m
        self.elli_points = 100  # number of points for the ideal ellipse

        self.pol_plus = np.deg2rad(
            angle_plus_deg)  # slope of the line from the sample to the upmost point on the analyser
        self.pol_minus = self.pol_plus - SOLAR_RANGE  # slope of the line from the sample to the downmost point on the analyser
        self.pol_middle = (self.pol_plus + self.pol_minus) / 2.0
        self.start_point = (CRYO_RADIUS, CRYO_RADIUS * np.tan(self.pol_plus))

        self.detector_line_hori = [0.0, 1.0, detector_height]
        self.detector_line_vert = [1.0, 0.0, -0.4]  # [1, 0, -h]: h -> horizontal position (m) of the vertical bank
        detector_suffix = '_{:.1f}_{:.1f}'.format(abs(self.detector_line_vert[2]), abs(self.detector_line_hori[2]))

        polar_focus_rad = np.deg2rad(polar_focus_deg)
        if polar_focus_rad < self.pol_minus:
            self.foc_point = distance_sf * np.array([np.cos(polar_focus_rad), np.sin(polar_focus_rad)])  # m
        else:
            raise RuntimeError("Invalid polar angle of the other focus.")

        self.filename_geo = "_".join(['Geometry', str(int(angle_plus_deg)), "{:.2f}".format(distance_sf)])
        # self.filename_polarangle = 'Resolution_PolarAngles' + detector_suffix
        self.filename_res = 'Resolution_Test'  # + detector_suffix

        self.an_seg_size = 1e-2  # m
        points_x, points_y, self.mcstas_rotation_rad = self._generate_analyser_segments(instrument=instrumentctx)
        self.polar_angles = np.arctan2(points_y, points_x)
        self.analyser_points = (points_x, points_y)
        self.wavenumbers_out = self._wavenumbers_from_analyser(instrument=instrumentctx)
        self.an_2theta = np.array(
            list(map(lambda i: self._get_an_twotheta(an_ind=i), range(self.analyser_points[0].shape[0]))))
        # print(self.wavenumbers)
        self.pol_middle_index = np.argmin(np.abs(self.polar_angles - self.pol_middle))

        # if the analyser is generated as a part of an ideal ellipse:
        self.theo_ellipse_points = self._generate_analyser_ellipse()

        self.detector_points = self._detector_from_analyser()[:2]
        self.dete_hori_x, self.dete_vert_y = self._detector_from_analyser()[2:]
        # print(self.dete_hori_x, '\n\n', self.dete_vert_y)

        self.filename_mcstas = 'Analyser_New.dat'
        self.arm_sa_name_prefix = "arm_sa_an"
        self.arm_sa_reference = "arm_sample_orientation"
        self.component_name_prefix = "graphite_analyser"
        self.component_type = "Monochromator_flat"
        self.parameter_width_z = "zwidth"
        self.parameter_height_y = "yheight"
        self.parameter_mosaic_horizontal = "mosaich"
        self.parameter_mosaic_vertical = "mosaicv"
        self.parameter_lattice_distance = "DM"
        self.parameter_radius_horizontal_focusing = "RH"
        self.parameter_number_slabs_horizontal = "NH"
        self.parameter_angle_phi = "phi"
        self.component_reference = "analyser_arm"
        self.group_name = "analyser"

    def _ellipse_points2parameters(self):
        """
        calculates the parameters of the general form of an ellipse from focii, in the form of
        A(x-h)^2 + B(x-h)(y-k) + C(y-k)^2 = 1
        :return: parameters A, B, C (denoted by aa, bb and cc, respectively), h, k
        """
        a = (points_distance(self.sample_point, self.start_point) + points_distance(
            self.foc_point, self.start_point)) / 2.0  # ellipse semi-major
        h, k = np.array(self.foc_point) / 2.0
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
        vector_af = points_to_vector(point1=analyser_point, point2=self.foc_point)
        vector_tangential = vector_bisector(vector_sa, vector_af)
        slope_rad = np.arctan(vector_tangential[1] / vector_tangential[0])
        return unit_vector(vector_tangential), slope_rad

    def _generate_analyser_segments(self, instrument: InstrumentContext):
        # generates the analyser with a finite segment size
        point_now = np.array(self.start_point)
        analyser_x = [self.start_point[0]]
        analyser_y = [self.start_point[1]]
        orientation_now, mcstas_rotation_now = self._analyser_segment_orientation(point_now)
        mcstas_rotation_radian = [mcstas_rotation_now]

        while np.arctan(point_now[1] / point_now[0]) > self.pol_minus:
            segment_analyser = orientation_now * instrument.an_seg
            point_now += segment_analyser  # update the next point
            orientation_now, mcstas_rotation_now = self._analyser_segment_orientation(point_now)
            analyser_x.append(point_now[0])
            analyser_y.append(point_now[1])
            mcstas_rotation_radian.append(mcstas_rotation_now)
        return np.array(analyser_x), np.array(analyser_y), np.array(mcstas_rotation_radian)

    def _intersect_on_ellipse(self, m):
        # gives the intersect of a line y = mx, with the ellipse described by the parameters (aa, bb, cc, h, k)
        aa, bb, cc, h, k = self._ellipse_points2parameters()
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
        angles = np.linspace(self.pol_plus, self.pol_minus, num=self.elli_points)
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
        def an_to_de(x, y):
            line_af = points_to_line(self.foc_point, [x, y])
            detector_point = lines_intersect(line1=line_af, line2=self.detector_line_hori)
            if y < self.foc_point[1] or detector_point[0] + self.detector_line_vert[-1] < - ZERO_TOL:
                detector_point = lines_intersect(line1=line_af, line2=self.detector_line_vert)
                if detector_point[1] + self.detector_line_hori[-1] < - ZERO_TOL:
                    raise RuntimeError("Failed to find a detector point.")
            return detector_point[0], detector_point[1]

        detector_x = np.array(list(
            map(lambda i: an_to_de(self.analyser_points[0][i], self.analyser_points[1][i])[0],
                range(self.analyser_points[0].shape[0]))))
        detector_y = np.array(list(
            map(lambda i: an_to_de(self.analyser_points[0][i], self.analyser_points[1][i])[1],
                range(self.analyser_points[0].shape[0]))))
        detector_hori_x = detector_x[detector_x > -self.detector_line_vert[-1]]
        detector_vert_y = detector_y[detector_y > -self.detector_line_hori[-1]]
        if detector_x[0] * detector_x[-1] < 0:
            raise RuntimeError("Detector points overlap.")
        return detector_x, detector_y, detector_hori_x, detector_vert_y

    def _wavenumbers_from_analyser(self, instrument: InstrumentContext):
        kf = list(map(
            lambda i: self.wavenumber_bragg(instrument=instrument, index=i), range(self.analyser_points[0].shape[0])))
        return np.array(kf)

    def wavelength_bragg(self, instrument: InstrumentContext, index, order_parameter=1):
        # gives the wavelength from the Bragg's law
        scattering_2theta = self._get_an_twotheta(an_ind=index)
        return 2. * instrument.lattice_distance_pg002 * np.sin(scattering_2theta / 2.) / float(order_parameter)

    def wavenumber_bragg(self, instrument: InstrumentContext, index, order_parameter=1):
        # gives the wave number from the Bragg's law
        wavelength = self.wavelength_bragg(instrument=instrument, index=index,
                                           order_parameter=order_parameter)
        return 2. * np.pi / wavelength

    def _get_an_twotheta(self, an_ind):
        # gives the scattering angle twotheta from the given index of the analyser segment
        an_point = (self.analyser_points[0][an_ind], self.analyser_points[1][an_ind])
        vector_sa = points_to_vector(self.sample_point, an_point)  # sa = sample_analyser
        vector_af = points_to_vector(an_point, self.foc_point)  # af = analyser_focus
        return angle_vectors(vector_sa, vector_af)

    def wavevector_transfer(self, index_pol, index_azi, rot_rad=0):
        ki_vector = np.array([self.wavenumber_in, 0, 0])
        kf = self.wavenumbers_out[index_pol]
        pol_rad = self.polar_angles[index_pol]
        azi_rad = self.azi_angles[index_azi]
        kf_vector = wavenumber_vector(wavenumber=kf, azi_angle=azi_rad, pol_angle=pol_rad)
        q_vector = ki_vector - kf_vector
        q_vector[:2] = rotation_z(rot_angle=rot_rad, old_x=q_vector[0], old_y=q_vector[1])
        return q_vector
