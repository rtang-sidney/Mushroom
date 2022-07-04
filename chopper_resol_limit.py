import numpy as np
import chopper_context as ccxt
import neutron_context as ncxt
import matplotlib.pyplot as plt

# Calculates the resolution limit of the chopper system for removing the frame overlaps and due to mechanical stability
# of the choppers

plt.rcParams.update({'font.size': 20})

wavelengths = np.linspace(ccxt.wavelength_min, ccxt.wavelength_max, num=100)
wavenumbers = ncxt.wavelength2wavenumber(wavelengths)


def limit_frame_overlap(k, d12, d1s, phi1, phi2, n1, n2, tau_max, tau_min):
    return phi1 * ncxt.habr * k * (tau_max - tau_min) / (2 * np.pi * ncxt.mass_neutron * d1s * (
            2 * np.pi / (phi2 * n1 * n2 * d12 - 1.0 / (n1 * d12) - phi1 / (phi2 * n1 * d1s))))


def limit_mechanical(k, phi1, f1, d1s):
    return phi1 * ncxt.habr * k / (2 * np.pi * ncxt.mass_neutron * d1s * f1)


u_rt = limit_frame_overlap(wavenumbers, ccxt.distance12, ccxt.distance1s, np.deg2rad(5), np.deg2rad(10), 6, 1,
                           ccxt.tau_max, ccxt.tau_min)
u_rt_f = limit_mechanical(wavenumbers, np.deg2rad(6), 10e3 / 60.0, ccxt.distance1s)

fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(wavelengths * 1e10, u_rt * 100)
ax.plot(wavelengths * 1e10, u_rt * 100, label="Frame-overlap limit")
ax.plot(wavelengths * 1e10, u_rt_f * 100, label="Mechanical limit")
ax.legend()
ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
ax.set_ylabel(r"Minimal achievable $\frac{\Delta t}{t} * 100\%$")
ax.tick_params(axis="both", top=True, right=True, direction="in")
fig.savefig("Resolution\\ChopperResolution_Limit.png", bbox_inches='tight')
plt.close(fig)
