import numpy as np
import sys
import neutron_context as neutron
import geometry_calculation as geo
from mushroom_context import MushroomContext

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
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(wavevector_transfer / reci_const)
        # print(scattering_vector, reci_const, hkl)
        magnon_vector = wavevector_transfer - hkl * reci_const
        return 2 * self.spin_coup * self.spin * (
                3 - np.cos(magnon_vector[0] * self.l_const) - np.cos(magnon_vector[1] * self.l_const) - np.cos(
            magnon_vector[2] * self.l_const))

    # not used
    def scatt_cross_qxqyde(self, qq_x, qq_y, hw, ki, resol=0.01, qq_z=None, kf=None, mushroom=False):
        if mushroom is True:  # the (Qx,Qy,Qz)-values must satisfy those available in Mushroom
            if abs((hw - (ki ** 2 - kf ** 2) * neutron.planck_constant ** 2 / (2 * neutron.mass_neutron)) / hw) > resol:
                return None
            else:
                pass
        else:  # no restriction from the Mushroom wave vectors
            kf = np.sqrt(ki ** 2 - 2 * neutron.mass_neutron * hw / neutron.planck_constant ** 2)
            pol_angle = np.arccos(np.linalg.norm([qq_x - ki, qq_y]) / kf)
            # it is from -pi/6 to +pi/6, where the cosine function is well-defined
            qq_z = -kf * np.sin(pol_angle)

        # it gives a symmetric pattern iff the ki lays on one reciprocal lattice point
        # print(kf * 1e-10)
        qq_vector = np.array([qq_x, qq_y, qq_z])

        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(qq_vector / reci_const)
        magnon_q = qq_vector - hkl * reci_const
        # it has no symmetry in x-direction, since ki and the reciprocal lattice constant are very likely different
        dd = 2 * self.spin_coup * self.spin * self.l_const ** 2
        magnon_hw = dd * np.linalg.norm(magnon_q) ** 2
        beta = 1.0 / (neutron.boltzmann * self.temp)
        n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
        neutrons_lose_energy = geo.dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
        neutrons_gain_energy = geo.dirac_delta_approx(hw, -magnon_hw, resol) * n_q
        return neutrons_lose_energy + neutrons_gain_energy  # prefactor * debye_waller *

    # not used
    def scatt_cross_kikf(self, ki_vector, kf_vector, resol=0.01):
        # it gives a symmetric pattern iff the ki lays on one reciprocal lattice point
        ki, kf = np.linalg.norm(ki_vector), np.linalg.norm(kf_vector)
        hw = neutron.planck_constant ** 2 * (ki ** 2 - kf ** 2) / (2 * neutron.mass_neutron)
        qq_vector = ki_vector - kf_vector
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(qq_vector / reci_const)
        magnon_q = qq_vector - hkl * reci_const
        # it has no symmetry in x-direction, since ki and the reciprocal lattice constant are very likely different
        dd = 2 * self.spin_coup * self.spin * self.l_const ** 2
        magnon_hw = dd * np.linalg.norm(magnon_q) ** 2
        beta = 1.0 / (neutron.boltzmann * self.temp)
        n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
        # prefactor = (FACTOR_GAMMA * THOMSON_SCATT_LENGTH) ** 2 * kf / ki * (
        #         2 * np.pi / self.l_const) ** 3 * self.spin * (
        #                     1 + kz ** 2)
        # debye_waller = 1
        energy_loss = geo.dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
        energy_gain = geo.dirac_delta_approx(hw, -magnon_hw, resol) * n_q
        return energy_loss + energy_gain
