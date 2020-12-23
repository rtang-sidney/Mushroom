import numpy as np
import instrument_context as instr
import geometry_calculation as geo
import neutron_context as neutron

HEIGHT_INSTR = 3.0  # m, from the highest point of the analyser to the horizontal bank of PSD
ELLI_POINTS = 100  # number of points for the ideal ellipse


class MushroomContext(object):
    def __init__(self, distance_sf=1.50, polar_focus_deg=-49.41, detector_height=2.28):
        self.sample_point = (0.0, 0.0)  # m
        self.foc_size = 1e-2  # m

        self.azi_angles = np.deg2rad(np.append(np.linspace(-170, -5, num=166), np.linspace(5, 170, num=166)))

        self.pol_plus = np.deg2rad(instr.angle_plus_deg)  # the uppermost point of the analyser
        self.pol_minus = self.pol_plus - np.deg2rad(instr.angle_range_deg)  # the nethermost point of the analyser
        self.pol_middle = (self.pol_plus + self.pol_minus) / 2.0
        self.start_point = (instr.cryo_radius, instr.cryo_radius * np.tan(self.pol_plus))

        self.detector_line_hori = [0.0, 1.0, detector_height]
        self.detector_line_vert = [1.0, 0.0, -instr.cryo_radius]
        # [1, 0, -h]: h -> horizontal position (m) of the vertical bank

        self.polar_focus_rad = np.deg2rad(polar_focus_deg)
        if self.polar_focus_rad < self.pol_minus:
            self.foc_point = distance_sf * np.array([np.cos(self.polar_focus_rad), np.sin(self.polar_focus_rad)])  # m
        else:
            raise RuntimeError("Invalid polar angle of the other focus.")

        self.ellipse_a, self.ellipse_b, self.ellipse_c, ellipse_h, ellipse_k, ellipse_phi = self._ellipse_parameters()

        self.filename_geo = "_".join(['Geometry', str(int(instr.angle_plus_deg)), "{:.2f}".format(distance_sf)])
        self.filename_res = "_".join(['Resolution', str(int(instr.angle_plus_deg)), "{:.2f}".format(distance_sf)])

        self.analyser_points, self.analyser_orients = self._generate_analyser_segments()

        self.mcstas_segments = (self.analyser_points[:, 1:] + self.analyser_points[:, :-1]) / 2.0
        self.mcstas_rotation_rad = np.arctan2(self.analyser_orients[1, :], self.analyser_orients[0, :])
        self.pol_angles = np.arctan2(self.analyser_points[1, :], self.analyser_points[0, :])
        self.wavenumbers_out = np.apply_along_axis(func1d=self._analyser2wavenumber, axis=0,
                                                   arr=self.analyser_points[:, :-1])
        self.an_2theta = np.apply_along_axis(func1d=self._analyser2twotheta, axis=0, arr=self.analyser_points)
        # print(self.wavenumbers)
        self.pol_middle_index = np.argmin(np.abs(self.pol_angles - self.pol_middle))

        # if the analyser is generated as a part of an ideal ellipse:
        self.theo_ellipse_points = self._generate_analyser_ellipse(ellipse_h, ellipse_k, ellipse_phi)

        self.detector_points = self._detector_from_analyser()[0]
        self.dete_hori_x, self.dete_vert_y = self._detector_from_analyser()[1:]

        self.filename_mcstas = 'Analyser_{:.2f}.dat'.format(instr.cryo_radius)
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

    def _ellipse_parameters(self):
        a = (geo.points_distance(self.sample_point, self.start_point) + geo.points_distance(
            self.foc_point, self.start_point)) / 2.0  # ellipse semi-major
        h, k = np.array(self.foc_point) / 2.0
        phi = np.arctan(k / h)  # rotational angle of the ellipse, positive sign = anti-clockwise
        c = abs(h / np.cos(phi))  # linear eccentricity, giving the distance of a focus to the ellipse centre
        b = np.sqrt(a ** 2 - c ** 2)  # semi-minor axis
        return a, b, c, h, k, phi

    def _ellipse_points2parameters(self, phi):
        """
        calculates the parameters of the general form of an ellipse from focii, in the form of
        A(x-h)^2 + B(x-h)(y-k) + C(y-k)^2 = 1
        :return: parameters A, B, C (denoted by aa, bb and cc, respectively), h, k
        """
        # parameters of the ellipse after the rotation: aa = A, bb = B, cc = C
        # Ax^2 + Bxy + Cy^2 = 1
        aa = np.cos(phi) ** 2 / self.ellipse_a ** 2 + np.sin(phi) ** 2 / self.ellipse_b ** 2
        bb = 2 * np.cos(phi) * np.sin(phi) * (
                1. / self.ellipse_a ** 2 - 1. / self.ellipse_b ** 2)
        cc = np.sin(phi) ** 2 / self.ellipse_a ** 2 + np.cos(phi) ** 2 / self.ellipse_b ** 2
        return aa, bb, cc

    def _analyser_segment_orientation(self, analyser_point):
        """
        gives the unit vector denoting the orientation of an analyser segment at a given point
        :param analyser_point: a 2-D list or numpy array
        :return: a unit vector as a 2-D list
        """
        vector_sa = geo.points2vector(point1=self.sample_point, point2=analyser_point)
        vector_af = geo.points2vector(point1=analyser_point, point2=self.foc_point)
        vector_tangential = geo.vector_bisector(vector_sa, vector_af)
        return geo.unit_vector(vector_tangential)

    def _generate_analyser_segments(self):
        '''
        generate the segments of the analyser and their orientations
        the number of segments is one larger than that of the orientations
        :return: segments and orientations as arrays with sizes 2x(N+1) and 2xN, respectively
        '''
        # generates the analyser with a finite segment size
        segment_start_now = np.array(self.start_point)
        segments_start = np.copy(segment_start_now).reshape((2, 1))
        orient_now = self._analyser_segment_orientation(segment_start_now)
        segments_orient = np.copy(orient_now).reshape((2, 1))

        while np.arctan2(segment_start_now[1], segment_start_now[0]) > self.pol_minus:
            # the end point is included, since the last point is the outside the threshold
            segment_analyser = orient_now * instr.an_seg
            segment_start_now += segment_analyser  # update the next point
            orient_now = self._analyser_segment_orientation(segment_start_now)
            segments_start = np.append(segments_start, segment_start_now.reshape((2, 1)), axis=1)
            segments_orient = np.append(segments_orient, orient_now.reshape((2, 1)), axis=1)

        segments_orient = segments_orient[:, :-1]  # the orientation of the endpoint is discharged
        # the number of the analyser points is one larger than that of the segments, hence also one larger than that of
        # the segment orientations
        return segments_start, segments_orient

    def _intersect_on_ellipse(self, m, h, k, phi):
        # gives the intersect of a line y = mx, with the ellipse described by the parameters (aa, bb, cc, h, k)
        aa, bb, cc = self._ellipse_points2parameters(phi=phi)
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
        if abs(aa * (x - h) ** 2 + bb * (x - h) * (y - k) + cc * (y - k) ** 2 - 1) < instr.ZERO_TOL:
            return np.array([x, y])
        else:
            raise RuntimeError("Something wrong when solving the ellipse edge points.")

    def _generate_analyser_ellipse(self, ellipse_h, ellipse_k, ellipse_phi):
        # generate the analyser as a part of an ideal ellipse
        angles = np.linspace(self.pol_plus, self.pol_minus, num=ELLI_POINTS)
        analyser = np.array(list(
            map(lambda m: self._intersect_on_ellipse(m, h=ellipse_h, k=ellipse_k, phi=ellipse_phi), np.tan(angles))))
        return np.transpose(analyser)

    def _detector_from_analyser(self):
        def an_to_de(an_point):
            line_af = geo.points_to_line(self.foc_point, an_point)
            detector_point = geo.lines_intersect(line1=line_af, line2=self.detector_line_hori)
            if detector_point[0] + self.detector_line_vert[-1] < - instr.ZERO_TOL:
                detector_point = geo.lines_intersect(line1=line_af, line2=self.detector_line_vert)
                if detector_point[1] + self.detector_line_hori[-1] < - instr.ZERO_TOL:
                    raise RuntimeError("Failed to find a detector point.")
            return detector_point

        detector = np.apply_along_axis(func1d=an_to_de, axis=0, arr=self.analyser_points)
        detector_hori_x = detector[0, detector[0, :] > -self.detector_line_vert[-1]]
        detector_vert_y = detector[1, detector[1, :] > -self.detector_line_vert[-1]]
        if detector[0, 0] * detector[0, -1] < 0:
            raise RuntimeError("Detector points overlap.")
        return detector, detector_hori_x, detector_vert_y

    def _analyser2wavenumber(self, an_point, order_parameter=1):
        # gives the wave number from the Bragg's law
        scattering_2theta = self._analyser2twotheta(an_point=an_point)
        return neutron.bragg_angle2wavenumber(twotheta=scattering_2theta, lattice_distance=instr.interplanar_pg002,
                                              order=order_parameter)

    def _analyser2twotheta(self, an_point):
        # gives the scattering angle twotheta from the given analyser segment
        vector_sa = geo.points2vector(self.sample_point, an_point)  # sa = sample_analyser
        vector_af = geo.points2vector(an_point, self.foc_point)  # af = analyser_focus
        return geo.angle_vectors(vector_sa, vector_af)
