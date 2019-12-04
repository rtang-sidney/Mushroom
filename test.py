import numpy as np

energy_in_meV = 8.0
energy_in_eV = energy_in_meV * 1e-3
energy_in_SI = energy_in_eV * 1.602176634e-19
hbar = 1.0545718e-34  # m2 kg / s
mass = 1.67492749804e-27  # kg
wavelength = np.pi * hbar * np.sqrt(2. / (mass * energy_in_SI))
wavevector = 2 * np.pi / wavelength
print(wavelength * 1e10)
print(wavevector * 1e-10)
