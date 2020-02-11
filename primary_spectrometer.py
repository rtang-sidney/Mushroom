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


def get_resolution_monochromator(geo_ctx: GeometryContext, instrument: InstrumentContext, ki):
    """
    calculates the resolution of the primary spectrometer with a monochromator
    :param geo_ctx: GeometryContext
    :param instrument: InstrumentContext
    :param ki: incoming wave number
    :return: deviations of the components in the sequence of dki (wave number), dtheta (azimuthal angle), dphi (polar
    angle)
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

    def get_spread_polar(geo_ctx: GeometryContext, instrument: InstrumentContext):
        # the spread of the polar angle is given simply by the divergence in both directions since there is no scattering
        # angles to be taken into consideration in this direction
        return np.sum(np.square([divergence_mono(geo_ctx=geo_ctx, instrument=instrument)]))

    def get_spread_arzimuthal(geo_ctx: GeometryContext, instrument: InstrumentContext):
        return angular_spread_monochromator(geo_ctx=geo_ctx, instrument=instrument)

    dki = get_delta_ki(geo_ctx=geo_ctx, instrument=instrument, ki=ki)
    dtheta = get_spread_arzimuthal(geo_ctx=geo_ctx, instrument=instrument)
    dphi = get_spread_polar(geo_ctx=geo_ctx, instrument=instrument)
    return dki, dtheta, dphi


def get_resolution_components(ki, dki, dtheta, dphi):
    """
    recalculates the resolution and gives the components in x (horizontal), y (vertical) and z (along the ki) directions
    :param ki: wave number
    :param dki: uncertainty of the wave number
    :param dtheta: uncertainty of the azimuthal angle
    :param dphi: uncertainty of the polar angle
    :return: all three components of the Q-resolution in the sequence of x, y, z
    """
    dqx = ki * np.sin(dtheta)
    dqy = ki * np.sin(dphi)
    dqz = dki
    return dqx, dqy, dqz


def plot_resolution(ki, dqx, dqy, dqz, filename):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(ki * 1e-10, dqx * 1e-10, color="blue")
    ax.plot(ki * 1e-10, dqy * 1e-10, color="red")
    ax.plot(ki * 1e-10, dqz * 1e-10, color="gold")
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.legend(("x: horizontal", "y: vertical", r"z: along $k_f$"))
    ax.set_xlabel(r"Incoming wavenumber $k_i$ (angstrom$^{-1}$)")
    ax.set_ylabel(r"$\Delta k_i$ (angstrom$^{-1}$)")
    ax.grid()
    ax.set_title("Q-resolution of the primary spectrometer with a monochromator")
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    ax2.plot(ki * 1e-10, dqz / ki * 1e2, color=colour_ax2)
    ax2.set_ylabel(r"$\dfrac{\Delta k_i}{k_i}$ (%)", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)

    plt.savefig(filename, bbox_inches='tight')
    print("{:s} plotted.".format(filename))


geometryctx = GeometryContext(side="same")
instrumentctx = InstrumentContext()

wavelength_incoming = np.linspace(start=3.5, stop=6, num=100) * 1e-10  # m, wavelength
wavenumber_incoming = wavelength_to_wavenumber(wavelength_incoming)

dki, dtheta, dphi = get_resolution_monochromator(geo_ctx=geometryctx, instrument=instrumentctx, ki=wavenumber_incoming)
dqx, dqy, dqz = get_resolution_components(ki=wavenumber_incoming, dki=dki, dtheta=dtheta, dphi=dphi)
filename = "Resolution_Primary_Monochromator.pdf"
plot_resolution(ki=wavenumber_incoming, dqx=dqx, dqy=dqy, dqz=dqz, filename=filename)
