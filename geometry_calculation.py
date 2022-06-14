import numpy as np

# This code gives universal constants as well geometrical calculations. It is universal and independent of the
# applications.

from global_context import zero_tol

UNIT_VECTOR_X = (1, 0, 0)
UNIT_VECTOR_Y = (0, 1, 0)
UNIT_VECTOR_Z = (0, 0, 1)


def points_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def vector2angle(vector1, vector2):
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
    if abs(y1 - y2) < zero_tol:
        return 0, 1, -y3
    else:
        m = -(x2 - x1) / (y2 - y1)
        return m, -1, y3 - m * x3


def points_to_line(point1, point2):
    def line_through_origin(point):
        # gives the line parameters if the line goes through the origin (0,0)
        if abs(point[0]) < zero_tol:  # when the other point is on the y-axis, too
            return 0.0, 1.0, 0.0
        else:  # finite slope
            return point[1] / point[0], -1.0, 0.0

    if np.linalg.norm(point1) < zero_tol:
        return line_through_origin(point2)
    elif np.linalg.norm(point2) < zero_tol:
        return line_through_origin(point1)
    else:  # if no point is at the origin
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        matrix = np.array([[x1, y1], [x2, y2]])
        array = np.full(2, 1.0)
        a, b = np.linalg.solve(matrix, array)
        return a, b, -1.0


def points2vector(p_start, p_end):
    # gives the vector pointing from the point1 to point2 (direction important)
    return np.array(p_end) - np.array(p_start)


def parameters_to_line(slope, x_inter=None, y_inter=None):
    # gives the line parameters in the form ax+by+c=0 from the form y=ax+b where a is the slope and b is the y intersect
    if y_inter is not None:
        return slope, -1., y_inter
    elif x_inter is not None:
        return slope, -1., -x_inter * slope
    else:
        raise ValueError("Neither x- or y-intercept is given. One of them is necessary.")


def point_to_line(slope, point):
    # gives the line parameters in the form ax+by+c=0 from the form y=ax+b where a is the slope and b is the y intersect
    y_inter = point[1] - slope * point[0]
    return slope, -1., y_inter
    # raise ValueError("Either x- or y-intercept is given. One of them is necessary.")


def vector2line(vector, point):
    vx, vy = vector
    x1, y1 = point
    return vy, -vx, vx * y1 - vy * x1


def line_to_y(x, line_params):
    a, b, c = line_params
    print(a, b, c)
    if abs(b) < zero_tol:
        raise RuntimeError("Given parameters define a line parallel to the y-axis.")
    else:
        return -(a * x + c) / b


def lines_intersect(line1, line2):
    """
    calculates the point where two lines intersect, the lines can be 2D or 3D
    The lines are defined in the form of a*x + b*y + c*z + d = 0 in 3D and without the term c*z in 2D
    :param line1: line1 with parameters (a1, b1, c1, d1)
    :param line2: line2 with parameters (a2, b2, c2, d2)
    :return:
    """
    if len(line1) == len(line2) and len(line1) in [3, 4]:
        matrix = np.array([line1[:-1], line2[:-1]])
        array = -np.array([line1[-1], line2[-1]])
        x1, x2 = np.linalg.solve(matrix, array)
        return np.array([x1, x2])
    else:
        raise RuntimeError("The line parameters provided are not valid. Try again.")


def angle_triangle(a, c, b=None):
    if b is None:
        b = a
    return np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))


def points_to_slope_radian(point1, point2):
    vector12 = points2vector(p_start=point1, p_end=point2)
    if abs(vector12[0]) > zero_tol:
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


def distance_point_line(point, line):
    # gives the distance from a point (x0,y0) to a line ax+by+c=0
    # works also for multiple points with points=(x_values, y_values), where both x and y values are numpy arrays with
    # the same length
    x0, y0 = point
    a, b, c = line
    return np.abs((a * x0 + b * y0 + c) / np.linalg.norm([a, b]))


def data2range(data, number_points=None):
    if number_points is None:
        number_points = data.shape[0]
    return np.linspace(np.min(data), np.max(data), num=number_points)


def rotation_around_z(rot_angle, old_x, old_y):
    # rotate a point in the same coordinate system
    new_x = old_x * np.cos(rot_angle) - old_y * np.sin(rot_angle)
    new_y = old_x * np.sin(rot_angle) + old_y * np.cos(rot_angle)
    return np.array([new_x, new_y])


def dirac_delta_approx(x, x0, resol):
    """
    approximates the delta dirac function with a Gaussian function with expectation value of x0 and with the limit that
    the variance->0, where variance sigma = resol*x0

    :param x: independent variable x
    :param x0: expectation value x0
    :param resol: relative variance, giving the variance sigma = resol * x0
    :return: approximated delta dirac function by Gaussian
    """
    a = abs(resol * x0)
    delta = np.exp(-((x - x0) / a) ** 2 / 2.0) / (a * np.sqrt(2 * np.pi))
    # print(x, x0, resol, delta)
    return delta


def point2line_3d(point_out, line_direction, point_on):
    point_vector = points2vector(point_out, point_on)
    return np.linalg.norm(np.cross(point_vector, line_direction)) / np.linalg.norm(line_direction)


def point2plane(point_out, vec1, vec2, point_on):
    """
    calculates the 3D-distance from a point to a plane, which is defined by two lines
    :param point_out: the point outside the plane
    :param vec1: vector1 defining the plane
    :param vec2: vector2 defining the plane
    :param point_on: point on the plane
    :return:
    """
    vector_oo = points2vector(p_start=point_on, p_end=point_out)
    vector_normal = np.cross(vec1, vec2)
    distance = abs(np.dot(vector_oo, vector_normal))
    return distance


def rotation_3d(vector, rot_axis, angle):
    # The angle is defined by the rotation of the vector anti-clockwise,
    # i.e. the rotation of the coordinate system clockwise
    angle = angle
    vector = np.array(vector)
    rot_axis = np.linalg.norm(np.array(rot_axis))
    if vector.shape[0] != 3:
        raise ValueError("The vector should be 3D, but it is given as {:d}D".format(vector.shape[0]))
    if rot_axis.shape[0] != 3:
        raise ValueError("The rotation axis should be 3D, but it is given as {:d}D".format(rot_axis.shape[0]))
    return rot_axis * np.dot(rot_axis, vector) + np.cos(angle) * np.cross(np.cross(rot_axis, vector),
                                                                          rot_axis) + np.sin(
        angle) * np.cross(rot_axis, vector)


def point_reflect(p0, p1):
    """
    calculates the inversion of point p1 with regarding to point p0
    :param p0: the point through which the inversion takes place
    :param p1: the point to be inverted through p0
    :return: point resulted from the inversion
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = 2 * p0 - p1
    return p2
