import numpy as np


class InstrumentContext(object):
    def __init__(self):
        self.moasic_pg002 = np.deg2rad(0.4)  # radian, PG crystal mosaic
        self.moasic_analyser = np.deg2rad(0.8)  # radian, analyser mosaic
        self.deltad_d = 6e-4  # relative uncertainty of the lattice distance, given in [paper2]
        self.lattice_distance_pg002 = 3.35e-10  # m, lattice distance d of a PG crystal
        self.analyser_segment = 1e-2  # m, the size of an analyser segment in 1D
        self.distance_ms = 1.0  # m, distance between the monochromator and sample
        self.divergence_initial = np.deg2rad(1.6)  # initial divergence directly from the neutron guide


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


def vector_bisector(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.linalg.norm(vector1) * vector2 + np.linalg.norm(vector2) * vector1


def wavenumber_to_2theta_bragg(instrument: InstrumentContext, wave_number, ordering=1):
    # returns the scattering angle 2theta according to the Bragg's law 2 * d * sin(theta) = n * lambda
    return 2.0 * np.arcsin(ordering * np.pi / (wave_number * instrument.lattice_distance_pg002))
