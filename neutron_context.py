import numpy as np

mass_neutron = 1.67492749804e-27  # kg
planck_constant = 1.0545718e-34  # m2 kg / s, h-bar
conversion_joule_per_ev = 1.602176634e-19  # J / eV
boltzmann = 1.380649e-23  # J / K
factor_gamma = 1.913  # dimensionless constant
thomson_length = 2.818e-15  # m, classical radius of electrons


def wavelength2wavenumber(wavelength):
    return 2.0 * np.pi / wavelength


def wavenumber2wavelength(wavenumber):
    return 2.0 * np.pi / wavenumber


def wavenumber2energy(wavenumber):
    return planck_constant ** 2 * wavenumber ** 2 / (2 * mass_neutron)


def wavelength2energy(wavelength):
    return wavenumber2energy(wavenumber=wavelength2wavenumber(wavelength=wavelength))


def energy2wavenumber(energy):
    return np.sqrt(2 * mass_neutron * energy) / planck_constant


def energy2wavelength(energy):
    return wavenumber2wavelength(wavenumber=energy2wavenumber(energy=energy))


def mev2joule(energy_mev):
    return energy_mev * 1e-3 * conversion_joule_per_ev


def joule2mev(energy_j):
    return energy_j / conversion_joule_per_ev * 1e3


def wavenumber2wavevector(wavenumber, azi_angle, pol_angle):
    return wavenumber * np.array(
        [np.cos(pol_angle) * np.cos(azi_angle), np.cos(pol_angle) * np.sin(azi_angle), np.sin(pol_angle)])


def bragg_wavenumber2angle(wavenumber, lattice_distance, ordering=1):
    # returns the scattering angle 2theta according to the Bragg's law 2 * d * sin(theta) = n * lambda
    return 2.0 * np.arcsin(ordering * np.pi / (wavenumber * lattice_distance))
