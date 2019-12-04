import numpy as np

MASS_NEUTRON = 1.67492749804e-27  # kg
PLANCKS_CONSTANT = 1.0545718e-34  # m2 kg / s
CONVERSION_JOULE_TO_EV = 1.602176634e-19  # J / eV


def wavelength_to_joule(wavelength):
    return 2 * np.pi ** 2 * PLANCKS_CONSTANT ** 2 / (MASS_NEUTRON * wavelength ** 2)


def wavelength_to_eV(wavelength):
    return wavelength_to_joule(wavelength) / CONVERSION_JOULE_TO_EV


def points_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def get_angle(vector1, vector2):
    """
    Returns the angle between vector1 and vector2 in radians

    :param vector1:
    :param vector2:
    :return:
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
