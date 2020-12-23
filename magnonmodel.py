import numpy as np
import sys
import neutron_context as neutron
import geometry_calculation as geo
from mushroom_context import MushroomContext

# [Paper1] Kusminskiy2019 - Quantum Magnetism, Spin Waves, and Optical Cavities
# [Paper2] Squires2009 - Introduction to the Theory of Thermal Neutron Scattering
np.set_printoptions(threshold=sys.maxsize, precision=2)


# this file defines the relevant parameters describing the magnon dispersion by itself
# TODO: perhaps one should rewrite it as a class so that one still can change all the parameters externally

class MagnonModel:
    def __init__(self, model_name, latt_const, spin=1, temperature=300, spin_coupling=None, stiff_const=None):
        self.name = model_name
        self.l_const = latt_const
        self.spin = spin
        self.temp = temperature
        if spin_coupling:
            self.spin_coup = spin_coupling
        elif stiff_const:
            self.spin_coup = stiff_const / (self.spin * self.l_const ** 2)
        else:
            raise ValueError("Either spin coupling or stiff constant must be given.")

    def magnon_energy(self, wavevector_transfer):
        # For the formula see [Paper1]
        wavevector_transfer = np.array(wavevector_transfer)
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(wavevector_transfer / reci_const)
        # print(scattering_vector, reci_const, hkl)
        magnon_vector = wavevector_transfer - hkl * reci_const
        return 2 * self.spin_coup * self.spin * (
                3 - np.cos(magnon_vector[0] * self.l_const) - np.cos(magnon_vector[1] * self.l_const) - np.cos(
            magnon_vector[2] * self.l_const))

    def correlation_func(self, qq_vector, hw, resol=0.05):
        # For the formula see [Paper2]
        magnon_hw = self.magnon_energy(wavevector_transfer=qq_vector)
        beta = 1.0 / (neutron.boltzmann * self.temp)
        n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
        energy_loss = geo.dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
        energy_gain = geo.dirac_delta_approx(hw, -magnon_hw, resol) * n_q
        return energy_loss + energy_gain


def magnon_scattered(scattering_de, magnon_de, de_of_e):
    if abs((scattering_de - magnon_de) / magnon_de) < de_of_e or abs(
            (scattering_de + magnon_de) / magnon_de) < de_of_e:
        return scattering_de
    else:
        return np.nan