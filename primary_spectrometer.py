import matplotlib.pyplot as plt
import numpy as np
from helper import wavelength_to_eV, points_distance, get_angle, vector_bisector, wavenumber_to_2theta_bragg, \
    InstrumentContext, wavelength_to_wavenumber
from geometry_context import GeometryContext

ZERO_TOL = 1e-6

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082

"""

# def get_divergence(sample, analyser_point, focus, sample_size, focus_size):
def divergence_mono(geo_ctx: GeometryContext, instrument: InstrumentContext):
    # ms: monochromator-sample
    divergence_in = instrument.divergence_initial
    divergence_out = geo_ctx.sample_size / instrument.distance_ms
    return divergence_in, divergence_out


def angular_spread_monochromator(geo_ctx: GeometryContext, instrument: InstrumentContext):
    eta = instrument.moasic_analyser  # mosaic
    alpha_i, alpha_f = divergence_mono(geo_ctx=geo_ctx, instrument=instrument)  # incoming and outgoing divergence
    numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
    denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2

    return np.sqrt(numerator / denominator)


def monochromator_twotheta(instrument: InstrumentContext, ki):
    return wavenumber_to_2theta_bragg(wave_number=ki, instrument=instrument)


def get_delta_ki(geo_ctx: GeometryContext, instrument: InstrumentContext, ki):
    # gives the deviation of the wave-number by means of the Bragg's law
    dtheta_analyser = angular_spread_monochromator(geo_ctx=geo_ctx, instrument=instrument)
    twotheta_analyser = monochromator_twotheta(instrument=instrument, ki=ki)
    dki_bragg = ki * np.sqrt(
        np.sum(np.square([instrument.deltad_d, dtheta_analyser / np.tan(twotheta_analyser / 2.0)])))
    return abs(dki_bragg)


geometryctx = GeometryContext(side="same")
instrumentctx = InstrumentContext()

filename = "Resolution_Primary.pdf"

wavelength_incoming = np.linspace(start=3.5, stop=6, num=100) * 1e-10  # m, wavelength
wavenumber_incoming = wavelength_to_wavenumber(wavelength_incoming)
dki = get_delta_ki(geo_ctx=geometryctx, instrument=instrumentctx, ki=wavenumber_incoming)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(wavenumber_incoming * 1e-10, dki * 1e-10)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.set_xlabel(r"Incoming wavenumber $k_i$ (angstrom$^{-1}$)")
ax.set_ylabel(r"$\Delta k_i$ (angstrom$^{-1}$)")
ax.grid()
plt.savefig(filename, bbox_inches='tight')
print("{:s} plotted.".format(filename))
