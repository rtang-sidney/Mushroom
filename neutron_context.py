import numpy as np

mass_neutron = 1.67492749804e-27  # kg
habr = 1.0545718e-34  # m2 kg / s, h-bar = h / (2*pi)
conversion_joule_per_ev = 1.602176634e-19  # J / eV
boltzmann = 1.380649e-23  # J / K
factor_gamma = 1.913  # dimensionless constant
thomson_length = 2.818e-15  # m, classical radius of electrons
bohr_magneton = 9.274009994e-24  # J / T, Bohr magneton \mu_B


def wavelength2wavenumber(wavelength):
    return 2.0 * np.pi / wavelength


def wavenumber2wavelength(wavenumber):
    return 2.0 * np.pi / wavenumber


def wavenumber2energy(wavenumber):
    return habr ** 2 * wavenumber ** 2 / (2 * mass_neutron)


def wavelength2energy(wavelength):
    return wavenumber2energy(wavenumber=wavelength2wavenumber(wavelength=wavelength))


def energy2wavenumber(energy):
    return np.sqrt(2 * mass_neutron * energy) / habr


def energy2wavelength(energy):
    return wavenumber2wavelength(wavenumber=energy2wavenumber(energy=energy))


def mev2joule(energy_mev):
    return energy_mev * 1e-3 * conversion_joule_per_ev


def velocity2wavenumber(velocity):
    return mass_neutron * velocity / habr


def velocity2wavelength(velocity):
    wavenumber = velocity2wavenumber(velocity)
    return wavenumber2wavelength(wavenumber)


def wavenumber2velocity(wavenumber):
    return habr * wavenumber / mass_neutron


def wavelength2velocity(wavelength):
    return wavenumber2velocity(wavelength2wavenumber(wavelength))


def joule2mev(energy_j):
    return energy_j / conversion_joule_per_ev * 1e3


def wavenumber2wavevector(wavenumber, azi_angle, pol_angle):
    return wavenumber * np.array(
        [np.cos(pol_angle) * np.cos(azi_angle), np.cos(pol_angle) * np.sin(azi_angle), np.sin(pol_angle)])


def bragg_wavenumber2twotheta(wavenumber, lattice_distance, order=1):
    # returns the scattering angle 2theta according to the Bragg's law 2 * d * sin(theta) = n * lambda
    return 2.0 * np.arcsin(order * np.pi / (wavenumber * lattice_distance))


def bragg_twotheta2wavenumber(twotheta, lattice_distance, order=1):
    return order * np.pi / (lattice_distance * np.sin(twotheta / 2.0))


def wavevector_transfer(wavevector_out, wavevector_in):
    wavevector_in = np.array(wavevector_in)
    wavevector_out = np.array(wavevector_out)
    return wavevector_in - wavevector_out


def energy_transfer(wavenumber_in, wavenumber_out):
    return habr ** 2 * (wavenumber_in ** 2 - wavenumber_out ** 2) / (2 * mass_neutron)


def bragg_twotheta2wavelength(twotheta, lattice_distance, order=1):
    return 2.0 * lattice_distance * np.sin(twotheta / 2.0) / float(order)


def q2rlu(q_value, l_const):
    return q_value / (2.0 * np.pi / l_const)


def rlu2q(rlu, l_const):
    return np.array(rlu) * (2 * np.pi / l_const)
