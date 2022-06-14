import numpy as np
import sys
import neutron_context as nctx
import geometry_calculation as geo

# from mushroom_context import MushroomContext

# [Paper1] Kusminskiy2019 - Quantum Magnetism, Spin Waves, and Optical Cavities (Page 52)
# [Paper2] Squires2009 - Introduction to the Theory of Thermal Neutron Scattering
# [Ref. 1] http://www.mcstas.org/download/components/samples/Magnon_bcc.comp
from global_context import type_bcc, type_cp

np.set_printoptions(threshold=sys.maxsize, precision=2)


# this file defines the relevant parameters describing the magnon dispersion by itself

class MagnonModel:
    def __init__(self, model_name, latt_const, spin=1, temperature=30, l_type=type_bcc, spin_coupling=None,
                 stiff_const=None):
        self.name = model_name
        self.l_const = latt_const
        self.spin = spin
        self.temp = temperature
        self.l_type = l_type
        if spin_coupling:
            self.spin_coup = spin_coupling
        elif stiff_const:
            self.spin_coup = stiff_const / (self.spin * self.l_const ** 2)
        else:
            raise ValueError("Either spin coupling or stiff constant must be given.")

    def magnon_energy(self, wavevector_transfer):
        # For the formula see [Ref. 1]
        wavevector_transfer = np.array(wavevector_transfer)
        reci_const = 2 * np.pi / self.l_const
        hkl = np.round(wavevector_transfer / reci_const)
        # print(scattering_vector, reci_const, hkl)
        magnon_vector = wavevector_transfer - hkl * reci_const
        return 2 * self.spin_coup * self.spin * (
                4.0 - np.cos(0.5 * self.l_const * (magnon_vector[0] + magnon_vector[1] + magnon_vector[2])) + np.cos(
            0.5 * self.l_const * (magnon_vector[0] + magnon_vector[1] - magnon_vector[2])) + np.cos(
            0.5 * self.l_const * (magnon_vector[0] - magnon_vector[1] + magnon_vector[2])) + np.cos(
            0.5 * self.l_const * (magnon_vector[0] - magnon_vector[1] - magnon_vector[2])))

    def magnon_rlu2energy(self, q_rlu):
        # For the formula see [Ref. 1]
        q_rlu = np.array(q_rlu)
        # q_rlu = (q_rlu - np.round(q_rlu)) * np.pi
        q_rlu *= np.pi
        if self.l_type == type_bcc:
            j10 = 4.0
            j1q = np.cos(q_rlu[0] + q_rlu[1] + q_rlu[2]) + np.cos(q_rlu[0] + q_rlu[1] - q_rlu[2]) + np.cos(
                q_rlu[0] - q_rlu[1] + q_rlu[2]) + np.cos(q_rlu[0] - q_rlu[1] - q_rlu[2])
            # print(j1q, np.cos(q_rlu[0] + q_rlu[1] + q_rlu[2]), np.cos(q_rlu[0] + q_rlu[1] - q_rlu[2]), np.cos(
            #     q_rlu[0] - q_rlu[1] + q_rlu[2]), np.cos(q_rlu[0] - q_rlu[1] - q_rlu[2]), q_rlu / np.pi)
        elif self.l_type == type_cp:
            j10 = 3.0
            j1q = np.cos(q_rlu[0]) + np.cos(q_rlu[1]) + np.cos(q_rlu[2])
        else:
            raise ValueError("Unknown lattice type given".format(type))
        return 2.0 * self.spin_coup * self.spin * (j10 - j1q)

    def corr_func_from_q_hw(self, hw, resol=0.03, q_rlu=None, q_real=None):
        # For the formula see [Paper2]
        # magnon_hw = self.magnon_energy(wavevector_transfer=qq_vector)
        if q_rlu is None:
            if q_real is None:
                raise ValueError("A Q-vector must given either in rlu or real space.")
            else:
                q_rlu = nctx.q2rlu(q_value=q_real, l_const=self.l_const)

        magnon_hw = self.magnon_rlu2energy(q_rlu=q_rlu)
        beta = 1.0 / (nctx.boltzmann * self.temp)
        n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
        if hw * magnon_hw > 0:
            return geo.dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
        else:
            return geo.dirac_delta_approx(hw, -magnon_hw, resol) * n_q


def magnon_scattered(scattering_de, magnon_de, de_of_e):
    if abs((scattering_de - magnon_de) / magnon_de) < de_of_e or abs(
            (scattering_de + magnon_de) / magnon_de) < de_of_e:
        return scattering_de
    else:
        return np.nan
