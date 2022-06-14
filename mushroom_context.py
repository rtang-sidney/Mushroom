import numpy as np
import instrument_context as instr
import geometry_calculation as geo
import neutron_context as nctx

HEIGHT_INSTR = 3.0  # m, from the highest point of the analyser to the horizontal bank of PSD
ELLI_POINTS = 100  # number of points for the ideal ellipse
PART_UP = "up"
PART_LOW = "low"


class MushroomContext(object):
    def __init__(self):
        self.sa_point = (0.0, 0.0)  # m
        self.foc_point = (0.8, -0.4)

        self.foc_size = 1e-2  # m
        self.wavenumber_f = 1.1e10  # a constant outgoing wavenumber
        self.wavelength_f = nctx.wavenumber2wavelength(self.wavenumber_f)

        self.pol_plus = np.deg2rad(instr.pol_plus_deg)  # the uppermost point of the analyser
        self.pol_minus = self.pol_plus - np.deg2rad(instr.pol_range_deg)  # the nethermost point of the analyser
        self.pol_middle = (self.pol_plus + self.pol_minus) / 2.0
        self.twotheta_an = nctx.bragg_wavenumber2twotheta(wavenumber=self.wavenumber_f,
                                                          lattice_distance=instr.interplanar_pg002)
        self.an_points, self.orients = self._generate_an_segs()
        self.pol_angles = np.arctan2(self.an_points[1, :-1], self.an_points[0, :-1])
        self.dete_bank_line = self._dete_height()
        self.dete_points = np.apply_along_axis(func1d=self._an2dete, axis=0,
                                               arr=self.an_points)  # 1 size larger than that of the point set
        self.filename_geo = "_".join(
            ['Geometry', str(int(instr.pol_plus_deg)), "kf{:.2f}".format(self.wavenumber_f * 1e-10)])
        self.filename_res = "_".join(
            ['Resolution', str(int(instr.pol_plus_deg)), "kf{:.2f}".format(self.wavenumber_f * 1e-10)])

        self.mcstas_rot_rad = np.arctan2(self.orients[1, :], self.orients[0, :])
        self.filename_mcstas = 'Analyser.dat'
        self.arm_sa_an = "arm_sa_an"
        self.arm_sa_reference = "arm_sample_orientation"
        self.component_name_prefix = "pg"
        self.component_ana = "Monochromator_flat"
        self.component_arm = "Arm"
        self.parameter_width_z = "zwidth"
        self.parameter_height_y = "yheight"
        self.parameter_mosaic_horizontal = "mosaich"
        self.parameter_mosaic_vertical = "mosaicv"
        self.parameter_lattice_distance = "DM"
        self.parameter_radius_horizontal_focusing = "RH"
        self.parameter_number_slabs_horizontal = "NH"
        self.parameter_angle_phi = "phi"
        self.component_reference = "arm_sa"
        self.group_name = "analyser"

    def _cal_an_point(self, theta_sa=None, vec_sa=None):
        # if p_a > 0:
        #     twotheta = twotheta_l
        # else:
        #     twotheta = twotheta_h
        if theta_sa is not None:
            pass
        elif vec_sa is not None:
            theta_sa = np.arctan2(vec_sa[1], vec_sa[0])
        else:
            raise ValueError("Either of the two parameters must be given.")

        theta_sf = np.arctan(-self.foc_point[1] / self.foc_point[0])
        twotheta = self.twotheta_an
        theta = theta_sa + theta_sf
        gamma = twotheta - theta
        vec_fs = geo.points2vector(p_start=self.foc_point, p_end=self.sa_point)
        vec_fa = geo.rotation_around_z(rot_angle=-gamma, old_x=vec_fs[0], old_y=vec_fs[1])
        line_sa = geo.point_to_line(slope=np.tan(theta_sa), point=self.sa_point)
        line_fa = geo.vector2line(vector=vec_fa, point=self.foc_point)
        # print(np.rad2deg(phi), np.rad2deg(gamma), line_sa, line_fa)
        an_point = geo.lines_intersect(line1=line_sa, line2=line_fa)
        vec_sa = geo.points2vector(p_start=self.sa_point, p_end=an_point)
        # ang_sa_af = geo.vector2angle(vec_sa, -vec_fa)
        # print(ang_sa_af, twotheta, nctx.bragg_twotheta2wavelength(twotheta, 3.355e-10))
        vec_seg = geo.unit_vector(geo.vector_bisector(vec_sa, -vec_fa))
        return an_point, vec_seg

    def _generate_an_segs(self):
        """
        generate the segments of the analyser and their orientations
        the number of segments is one larger than that of the orientations
        :return: segments and orientations as arrays with sizes 2x(N+1) and 2xN, respectively
        """
        # generates the analyser with a finite segment size
        point_now, orient_now = self._cal_an_point(theta_sa=self.pol_plus)
        if point_now[0] < instr.cryo_radius:
            raise RuntimeError("Not enough space for the cryostat.")
        points = np.copy(point_now).reshape((2, 1))
        orients = np.copy(orient_now).reshape((2, 1))

        while np.arctan2(point_now[1], point_now[0]) > self.pol_minus:
            # the end point is included, since the last point is the outside the threshold
            seg_now = orient_now * instr.an_seg
            seg_end = point_now + seg_now  # update the next point
            vec_sa_now = geo.points2vector(p_start=self.sa_point, p_end=seg_end)
            point_now, orient_now = self._cal_an_point(vec_sa=vec_sa_now)
            points = np.append(points, point_now.reshape((2, 1)), axis=1)
            orients = np.append(orients, orient_now.reshape((2, 1)), axis=1)

        orients = orients[:, :-1]  # the orientation of the endpoint is discharged
        # the number of the analyser points is one larger than that of the segments, hence also one larger than that of
        # the segment orientations
        return points, orients

    def _dete_height(self):
        dete_y = 2 * self.foc_point[1] - self.an_points[1, 0]
        line_dete = geo.point_to_line(slope=0, point=[0, dete_y])
        return line_dete

    def _an2dete(self, an_point):
        # calculate the detector point for an given analyser point
        line_af = geo.points_to_line(point1=an_point, point2=self.foc_point)
        dete_point = geo.lines_intersect(line1=line_af, line2=self.dete_bank_line)
        return dete_point

# class MushroomContext2(object):
#     def __init__(self, distance_sf=0.92, polar_focus_deg=-31.45, detector_height=1.31):
#         self.sample_point = (0.0, 0.0)  # m
#         self.foc_size = 1e-2  # m
#
#         self.pol_plus = np.deg2rad(instr.pol_plus_deg)  # the uppermost point of the analyser
#         self.pol_minus = self.pol_plus - np.deg2rad(instr.pol_range_deg)  # the nethermost point of the analyser
#         self.pol_middle = (self.pol_plus + self.pol_minus) / 2.0
#         self.start_point = (instr.cryo_radius, instr.cryo_radius * np.tan(self.pol_plus))
#
#         self.detector_line_hori = [0.0, 1.0, detector_height]
#         self.detector_line_vert = [1.0, 0.0, -instr.cryo_radius]
#         # [1, 0, -h]: h -> horizontal position (m) of the vertical bank
#
#         self.polar_focus_rad = np.deg2rad(polar_focus_deg)
#         if self.polar_focus_rad < self.pol_minus:
#             self.foc_point = distance_sf * np.array([np.cos(self.polar_focus_rad), np.sin(self.polar_focus_rad)])  # m
#         else:
#             raise RuntimeError("Invalid polar angle of the other focus.")
#         # print("The other focus ({:.2f}, {:.2f}) m".format(*self.foc_point))
#
#         self._elli_a, self._elli_b, self._elli_c, elli_h, elli_k, elli_phi = self._ellipse_parameters()
#
#         self.filename_geo = "_".join(['Geometry', str(int(instr.pol_plus_deg)), "{:.2f}".format(distance_sf)])
#         self.filename_res = "_".join(['Resolution', str(int(instr.pol_plus_deg)), "{:.2f}".format(distance_sf)])
#
#         self.analyser_points, self.analyser_orients = self._generate_analyser_segments()
#
#         self.mcstas_segments = (self.analyser_points[:, 1:] + self.analyser_points[:, :-1]) / 2.0
#         self.mcstas_rotation_rad = np.arctan2(self.analyser_orients[1, :], self.analyser_orients[0, :])
#         self.pol_angles = np.arctan2(self.analyser_points[1, :], self.analyser_points[0, :])
#         self.wavenumbers_out = np.apply_along_axis(func1d=self._analyser2wavenumber, axis=0,
#                                                    arr=self.analyser_points[:, :-1])
#
#         ind_kfmin = np.argmin(self.wavenumbers_out)
#         ind_kfmax = np.argmax(self.wavenumbers_out)
#
#         self.an_2theta = np.apply_along_axis(func1d=self._analyser2twotheta, axis=0, arr=self.analyser_points[:, :-1])
#         np.apply_along_axis(func1d=self._analyser_relocate, axis=0, arr=self.analyser_points[:, :ind_kfmax + 1],
#                             goal_2theta=(self.an_2theta[ind_kfmax] + self.an_2theta[ind_kfmin]) / 2.0)
#         # print(self.wavenumbers)
#         self.pol_middle_index = np.argmin(np.abs(self.pol_angles - self.pol_middle))
#
#         # if the analyser is generated as a part of an ideal ellipse:
#         self.theo_ellipse_points = self._generate_analyser_ellipse(elli_h, elli_k, elli_phi)
#
#         self.detector_points = self._detector_from_analyser()[0]
#         self.dete_hori_x, self.dete_vert_y = self._detector_from_analyser()[1:]
#
#         self.filename_mcstas = 'Analyser_{:.2f}.dat'.format(instr.cryo_radius)
#         self.arm_sa_an = "arm_sa_an"
#         self.arm_sa_reference = "arm_sample_orientation"
#         self.component_name_prefix = "pg_analyser"
#         self.component_ana = "Monochromator_flat"
#         self.component_arm = "Arm"
#         self.parameter_width_z = "zwidth"
#         self.parameter_height_y = "yheight"
#         self.parameter_mosaic_horizontal = "mosaich"
#         self.parameter_mosaic_vertical = "mosaicv"
#         self.parameter_lattice_distance = "DM"
#         self.parameter_radius_horizontal_focusing = "RH"
#         self.parameter_number_slabs_horizontal = "NH"
#         self.parameter_angle_phi = "phi"
#         self.component_reference = "arm_sa"
#         self.group_name = "analyser"
#
#     def _ellipse_parameters(self):
#         a = (geo.points_distance(self.sample_point, self.start_point) + geo.points_distance(
#             self.foc_point, self.start_point)) / 2.0  # ellipse semi-major
#         h, k = np.array(self.foc_point) / 2.0
#         phi = np.arctan(k / h)  # rotational angle of the ellipse, positive sign = anti-clockwise
#         c = abs(h / np.cos(phi))  # linear eccentricity, giving the distance of a focus to the ellipse centre
#         b = np.sqrt(a ** 2 - c ** 2)  # semi-minor axis
#         return a, b, c, h, k, phi
#
#     def _ellipse_points2parameters(self, phi):
#         """
#         calculates the parameters of the general form of an ellipse from focii, in the form of
#         A(x-h)^2 + B(x-h)(y-k) + C(y-k)^2 = 1
#         :return: parameters A, B, C (denoted by aa, bb and cc, respectively), h, k
#         """
#         # parameters of the ellipse after the rotation: aa = A, bb = B, cc = C
#         # Ax^2 + Bxy + Cy^2 = 1
#         aa = np.cos(phi) ** 2 / self._elli_a ** 2 + np.sin(phi) ** 2 / self._elli_b ** 2
#         bb = 2 * np.cos(phi) * np.sin(phi) * (
#                 1. / self._elli_a ** 2 - 1. / self._elli_b ** 2)
#         cc = np.sin(phi) ** 2 / self._elli_a ** 2 + np.cos(phi) ** 2 / self._elli_b ** 2
#         return aa, bb, cc
#
#     def _analyser_segment_orientation(self, analyser_point):
#         """
#         gives the unit vector denoting the orientation of an analyser segment at a given point
#         :param analyser_point: a 2-D list or numpy array
#         :return: a unit vector as a 2-D list
#         """
#         vector_sa = geo.points2vector(p_start=self.sample_point, p_end=analyser_point)
#         vector_af = geo.points2vector(p_start=analyser_point, p_end=self.foc_point)
#         vector_tangential = geo.vector_bisector(vector_sa, vector_af)
#         return geo.unit_vector(vector_tangential)
#
#     def _generate_analyser_segments(self):
#         """
#         generate the segments of the analyser and their orientations
#         the number of segments is one larger than that of the orientations
#         :return: segments and orientations as arrays with sizes 2x(N+1) and 2xN, respectively
#         """
#         # generates the analyser with a finite segment size
#         segment_start_now = np.array(self.start_point)
#         segments_start = np.copy(segment_start_now).reshape((2, 1))
#         orient_now = self._analyser_segment_orientation(segment_start_now)
#         segments_orient = np.copy(orient_now).reshape((2, 1))
#
#         while np.arctan2(segment_start_now[1], segment_start_now[0]) > self.pol_minus:
#             # the end point is included, since the last point is the outside the threshold
#             segment_analyser = orient_now * instr.an_seg
#             segment_start_now += segment_analyser  # update the next point
#             orient_now = self._analyser_segment_orientation(segment_start_now)
#             segments_start = np.append(segments_start, segment_start_now.reshape((2, 1)), axis=1)
#             segments_orient = np.append(segments_orient, orient_now.reshape((2, 1)), axis=1)
#
#         segments_orient = segments_orient[:, :-1]  # the orientation of the endpoint is discharged
#         # the number of the analyser points is one larger than that of the segments, hence also one larger than that of
#         # the segment orientations
#         return segments_start, segments_orient
#
#     def _new_segments_2theta(self, seg_point, direction, mark_now):
#         point_now = seg_point + mark_now * direction
#         vec_sa_now = geo.points2vector(p_start=self.sample_point, p_end=point_now)
#         vec_af_now = geo.points2vector(p_start=point_now, p_end=self.foc_point)
#         twotheta_now = geo.vector2angle(vector1=vec_sa_now, vector2=vec_af_now)
#         return twotheta_now
#
#     def _analyser_relocate(self, seg_point, goal_2theta):
#         vec_as = geo.points2vector(p_start=seg_point, p_end=self.sample_point)
#         vec_af = geo.points2vector(p_start=seg_point, p_end=self.foc_point)
#         bisect_vec = geo.vector_bisector(vec_as, vec_af)
#         mark_array = np.linspace(-10e-3, 10e-3, num=100)
#         angles = np.array(list(
#             map(lambda m: self._new_segments_2theta(seg_point=seg_point, direction=bisect_vec, mark_now=m),
#                 mark_array)))
#         index = np.argmin(abs(angles - goal_2theta))
#         min_ang = abs(angles[index] - goal_2theta)
#         min_mark = mark_array[index]
#         dkf_kf = abs(min_ang / np.tan(goal_2theta))
#         print("Smallest dtheta {:.1e} deg after being relocated by {:.1e} m, dkf/kf={:.1e}".format(np.rad2deg(min_ang),
#                                                                                                    min_mark, dkf_kf))
#
#     def _intersect_on_ellipse(self, m, h, k, phi):
#         # gives the intersect of a line y = mx, with the ellipse described by the parameters (aa, bb, cc, h, k)
#         aa, bb, cc = self._ellipse_points2parameters(phi=phi)
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
#         if abs(aa * (x - h) ** 2 + bb * (x - h) * (y - k) + cc * (y - k) ** 2 - 1) < instr.zero_tol:
#             return np.array([x, y])
#         else:
#             raise RuntimeError("Something wrong when solving the ellipse edge points.")
#
#     def _generate_analyser_ellipse(self, ellipse_h, ellipse_k, ellipse_phi):
#         # generate the analyser as a part of an ideal ellipse
#         angles = np.linspace(self.pol_plus, self.pol_minus, num=ELLI_POINTS)
#         analyser = np.array(list(
#             map(lambda m: self._intersect_on_ellipse(m, h=ellipse_h, k=ellipse_k, phi=ellipse_phi), np.tan(angles))))
#         return np.transpose(analyser)
#
#     def _detector_from_analyser(self):
#         def an_to_de(an_point):
#             line_af = geo.points_to_line(self.foc_point, an_point)
#             detector_point = geo.lines_intersect(line1=line_af, line2=self.detector_line_hori)
#             if detector_point[0] + self.detector_line_vert[-1] < - instr.zero_tol:
#                 detector_point = geo.lines_intersect(line1=line_af, line2=self.detector_line_vert)
#                 if detector_point[1] + self.detector_line_hori[-1] < - instr.zero_tol:
#                     raise RuntimeError("Failed to find a detector point.")
#             return detector_point
#
#         detector = np.apply_along_axis(func1d=an_to_de, axis=0, arr=self.analyser_points)
#         detector_hori_x = detector[0, detector[0, :] > -self.detector_line_vert[-1]]
#         detector_vert_y = detector[1, detector[1, :] > -self.detector_line_vert[-1]]
#         if detector[0, 0] * detector[0, -1] < 0:
#             raise RuntimeError("Detector points overlap.")
#         return detector, detector_hori_x, detector_vert_y
#
#     def _analyser2wavenumber(self, an_point, order_parameter=1):
#         # gives the wave number from the Bragg's law
#         scattering_2theta = self._analyser2twotheta(an_point=an_point)
#         return nctx.bragg_angle2wavenumber(twotheta=scattering_2theta, lattice_distance=instr.interplanar_pg002,
#                                            order=order_parameter)
#
#     def _analyser2twotheta(self, an_point):
#         # gives the scattering angle twotheta from the given analyser segment
#         vector_sa = geo.points2vector(self.sample_point, an_point)  # sa = sample_analyser
#         vector_af = geo.points2vector(an_point, self.foc_point)  # af = analyser_focus
#         return geo.vector2angle(vector_sa, vector_af)
