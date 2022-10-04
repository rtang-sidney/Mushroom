import numpy as np
import chopper_context as cctx
import neutron_context as nctx
import matplotlib.pyplot as plt

# Calculates the resolution limit of the chopper system for removing the frame overlaps and due to mechanical stability
# of the choppers

plt.rcParams.update({'font.size': 20})

wavelengths = np.linspace(cctx.wavelength_min, cctx.wavelength_max, num=100)
wavenumbers = nctx.wavelength2wavenumber(wavelengths)

u_rt = cctx.limit_frame_overlap(wavenumbers, cctx.distance12, cctx.distance1s, np.deg2rad(12), np.deg2rad(9), 5, 1,
                                cctx.tau_max, cctx.tau_min)
u_rt_f = cctx.limit_mechanical(wavenumbers, np.deg2rad(12), 10e3 / 60.0, cctx.distance1s)

fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(wavelengths * 1e10, u_rt * 100)
ax.plot(wavelengths * 1e10, u_rt * 100, label="Frame-overlap limit")
ax.plot(wavelengths * 1e10, u_rt_f * 100, label="Mechanical limit")
ax.legend()
ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
ax.set_ylabel(r"Smallest achievable $\frac{\Delta t}{t} * 100\%$")
ax.tick_params(axis="both", top=True, right=True, direction="in")
fig.savefig("Resolution\\ChopperResolution_Limit.png", bbox_inches='tight')
plt.close(fig)
