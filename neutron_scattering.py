import numpy as np

from helper import InstrumentContext


def wavenumber_2theta_bragg(instrument: InstrumentContext, wave_number, ordering=1):
    # returns the scattering angle 2theta according to the Bragg's law 2 * d * sin(theta) = n * lambda
    return 2.0 * np.arcsin(ordering * np.pi / (wave_number * instrument.lattice_distance_pg002))
