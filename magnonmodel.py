from helper import PLANCKS_CONSTANT, MASS_NEUTRON, CONVERSION_JOULE_PER_EV, wavenumber_vector, BOLTZMANN, \
    dirac_delta_approx, THOMSON_SCATT_LENGTH, FACTOR_GAMMA
import numpy as np
from geometry_context import GeometryContext
import sys

np.set_printoptions(threshold=sys.maxsize, precision=2)


# this file defines the relevant parameters describing the magnon dispersion by itself
# TODO: perhaps one should rewrite it as a class so that one still can change all the parameters externally

class MagnonModel:
    def __init__(self, latt_const=4.5 * 1e-10, spin_coupling=0.3 * 1e-3 * CONVERSION_JOULE_PER_EV, spin=1,
                 temperature=300, stiff_const=None):
        self.l_const = latt_const
        self.spin_coup = spin_coupling
        self.spin = spin
        self.temp = temperature
        if stiff_const:
            self.stiff_const = stiff_const
        else:
            self.stiff_const = 2 * self.spin_coup * self.spin * self.l_const ** 2

    def if_scattered(self, ki_vector, kf_vector):
        """
        gives back whether the scattering event is allowed or not
        :param ki_vector: m-1, wave-vector of neutrons
        :param kf_vector: m-1, outgoing wave-vector of neutrons
        :return: True (if scattering allowed) or False (if not allowed)
        """
        scattering_vector = ki_vector - kf_vector  # wave-vector transfer
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(scattering_vector / reci_const)
        # print(scattering_vector, reci_const, hkl)
        q_vector = scattering_vector - hkl * reci_const
        de = PLANCKS_CONSTANT ** 2 * (np.linalg.norm(ki_vector) ** 2 - np.linalg.norm(kf_vector) ** 2) / (
                2 * MASS_NEUTRON)
        dd = 2 * self.spin_coup * self.spin * self.l_const ** 2
        if abs(de - dd * np.linalg.norm(q_vector) ** 2) / abs(de) < 5e-2:
            return 1
        else:
            return 0

    def qyqz_to_kf(self, geo_ctx: GeometryContext, de, qy, qz, acute=True):
        ki = geo_ctx.wavenumber_in
        kf = np.sqrt(ki ** 2 - 2 * MASS_NEUTRON * de / PLANCKS_CONSTANT ** 2)
        if abs(qz / kf) <= 1:
            phi = np.arcsin(-qz / kf)
            if abs(qy / (kf * np.cos(phi))) <= 1:
                theta = np.arcsin(-qy / (kf * np.cos(phi)))
                if np.min(geo_ctx.wavenumbers_out) < kf < np.max(
                        geo_ctx.wavenumbers_out):
                    if np.min(geo_ctx.pol_angles) < phi < np.max(
                            geo_ctx.pol_angles):
                        if np.min(geo_ctx.azi_nega) < theta < np.max(geo_ctx.azi_nega) or np.min(
                                geo_ctx.azi_posi) < theta < np.max(geo_ctx.azi_posi):
                            if acute:
                                return wavenumber_vector(kf, theta, phi)
                            else:
                                # TODO: correct this for negative angles
                                return wavenumber_vector(kf, np.pi - theta, phi)
                        else:
                            return np.empty(3)
                    else:
                        return np.empty(3)
                else:
                    return np.empty(3)
            else:
                return np.empty(3)
        else:
            return np.empty(3)

    def magnon_energy(self, wavevector_transfer):
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(wavevector_transfer / reci_const)
        # print(scattering_vector, reci_const, hkl)
        magnon_vector = wavevector_transfer - hkl * reci_const
        return self.stiff_const * np.linalg.norm(magnon_vector) ** 2

    def scatt_cross_qxqyde(self, qq_x, qq_y, hw, ki, resol=0.01, qq_z=None, kf=None, pol_angle=None, mushroom=False):
        if mushroom is True:  # the (Qx,Qy,Qz)-values must satisfy those available in Mushroom
            if abs((hw - (ki ** 2 - kf ** 2) * PLANCKS_CONSTANT ** 2 / (2 * MASS_NEUTRON)) / hw) > resol:
                return None
            else:
                pass
        else:  # no restriction from the Mushroom wave vectors
            kf = np.sqrt(ki ** 2 - 2 * MASS_NEUTRON * hw / PLANCKS_CONSTANT ** 2)
            pol_angle = np.arccos(np.linalg.norm([qq_x - ki, qq_y]) / kf)
            # it is from -pi/6 to +pi/6, where the cosine function is well-defined
            qq_z = -kf * np.sin(pol_angle)

        # it gives a symmetric pattern iff the ki lays on one reciprocal lattice point
        # print(kf * 1e-10)
        qq_vector = np.array([qq_x, qq_y, qq_z])

        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(qq_vector / reci_const)
        kz = qq_vector[-1] / np.linalg.norm(qq_vector)
        magnon_q = qq_vector - hkl * reci_const
        # it has no symmetry in x-direction, since ki and the reciprocal lattice constant are very likely different
        dd = 2 * self.spin_coup * self.spin * self.l_const ** 2
        magnon_hw = dd * np.linalg.norm(magnon_q) ** 2
        beta = 1.0 / (BOLTZMANN * self.temp)
        n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
        prefactor = (FACTOR_GAMMA * THOMSON_SCATT_LENGTH) ** 2 * kf / ki * (
                2 * np.pi / self.l_const) ** 3 * self.spin * (
                            1 + kz ** 2)
        debye_waller = 1
        neutrons_lose_energy = dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
        neutrons_gain_energy = dirac_delta_approx(hw, -magnon_hw, resol) * n_q
        return neutrons_lose_energy + neutrons_gain_energy  # prefactor * debye_waller *

    def scatt_cross_kikf(self, ki_vector, kf_vector, resol=0.01):
        # it gives a symmetric pattern iff the ki lays on one reciprocal lattice point
        ki, kf = np.linalg.norm(ki_vector), np.linalg.norm(kf_vector)
        hw = PLANCKS_CONSTANT ** 2 * (ki ** 2 - kf ** 2) / (2 * MASS_NEUTRON)
        qq_vector = ki_vector - kf_vector
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(qq_vector / reci_const)
        kz = qq_vector[-1] / np.linalg.norm(qq_vector)
        magnon_q = qq_vector - hkl * reci_const
        # it has no symmetry in x-direction, since ki and the reciprocal lattice constant are very likely different
        dd = 2 * self.spin_coup * self.spin * self.l_const ** 2
        magnon_hw = dd * np.linalg.norm(magnon_q) ** 2
        beta = 1.0 / (BOLTZMANN * self.temp)
        n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
        # prefactor = (FACTOR_GAMMA * THOMSON_SCATT_LENGTH) ** 2 * kf / ki * (
        #         2 * np.pi / self.l_const) ** 3 * self.spin * (
        #                     1 + kz ** 2)
        # debye_waller = 1
        energy_loss = dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
        energy_gain = dirac_delta_approx(hw, -magnon_hw, resol) * n_q
        return energy_loss + energy_gain
