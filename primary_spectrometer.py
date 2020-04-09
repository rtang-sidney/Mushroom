import matplotlib.pyplot as plt
import numpy as np
from helper import wavenumber_to_2theta_bragg, InstrumentContext, wavelength_to_wavenumber, angle_isoceles

ZERO_TOL = 1e-6
ENERGY_SELECTION_MONOCHROMATOR = "Monochromator"
ENERGY_SELECTION_VELOCITY_SELECTOR = "VelocitySelector"
FILENAME_PREFIX = "Resolution_Primary_"
FILENAME_MONOCHROMATOR = "Resolution_Primary_Monochromator.pdf"
FILENAME_VELOCITY_SELECTOR = "Resolution_Primary_VelocitySelector.pdf"
AXIS_ARZIMUTHAL = "x"
AXIS_POLAR = "y"

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082

"""


def get_filename(energy_selection_type):
    if isinstance(energy_selection_type, str):
        return FILENAME_PREFIX + energy_selection_type + ".pdf"
    else:
        raise RuntimeError("Invalid type of the energy selection variable given.")


def get_resolution_monochromator(instrument: InstrumentContext, ki):
    """
    calculates the resolution of the primary spectrometer with a monochromator
    :param instrument: InstrumentContext
    :param ki: incoming wave number
    :return: deviations of the components in the sequence of dki (wave number), dtheta (azimuthal angle), dphi (polar
    angle)
    """

    def divergence_mono(instrument: InstrumentContext, axis):
        # distance_ms: monochromator-sample distance
        divergence_in = instrument.divergence_initial
        if axis == AXIS_ARZIMUTHAL:
            divergence_out = angle_isoceles(a=instrument.distance_ms, c=instrument.sample_diameter)
        elif axis == AXIS_POLAR:
            divergence_out = angle_isoceles(a=instrument.distance_ms, c=instrument.sample_height)
        else:
            raise RuntimeError("Invalid axis given.")
        return divergence_in, divergence_out

    def angular_spread_monochromator(instrument: InstrumentContext, axis):
        eta = instrument.moasic_analyser  # mosaic
        alpha_i, alpha_f = divergence_mono(instrument=instrument, axis=axis)  # incoming and outgoing divergence
        numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
        denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2

        return np.sqrt(numerator / denominator)

    def monochromator_twotheta(instrument: InstrumentContext, ki):
        return wavenumber_to_2theta_bragg(wave_number=ki, instrument=instrument)

    def get_uncertainty_ki(instrument: InstrumentContext, ki):
        # gives the deviation of the wave-number by means of the Bragg's law
        dtheta_mono = angular_spread_monochromator(instrument=instrument, axis=AXIS_ARZIMUTHAL)
        twotheta_mono = monochromator_twotheta(instrument=instrument, ki=ki)
        dki_bragg = ki * np.sqrt(
            np.sum(np.square([instrument.deltad_d, dtheta_mono / np.tan(twotheta_mono / 2.0)])))
        return abs(dki_bragg)

    def get_spread_polar(instrument: InstrumentContext):
        # the spread of the polar angle is given simply by the divergence in both directions since there is no scattering
        # angles to be taken into consideration in this direction
        return angle_isoceles(instrument.distance_ms, instrument.sample_height)

    def get_spread_arzimuthal(instrument: InstrumentContext):
        return min(np.deg2rad(1.6), angular_spread_monochromator(instrument=instrument, axis=AXIS_POLAR))

    dki = get_uncertainty_ki(instrument=instrument, ki=ki)
    dtheta = get_spread_arzimuthal(instrument=instrument)
    dphi = get_spread_polar(instrument=instrument)
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
    dqx = ki * np.tan(dtheta)
    dqy = ki * np.tan(dphi)
    dqz = dki
    return dqx, dqy, dqz


def plot_resolution(ki, dqx, dqy, dqz, energy_selection):
    filename = get_filename(energy_selection_type=energy_selection)
    plt.rcParams.update({'font.size': 12})
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
    ax.set_title("Q-resolution of the primary spectrometer: {}".format(energy_selection))
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    ax2.plot(ki * 1e-10, dqz / ki * 1e2, '1', color=colour_ax2)
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"$\dfrac{\Delta k_i}{k_i}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)

    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('pdf', 'png'), bbox_inches='tight')
    print("{:s} plotted.".format(filename))


def get_resolution_velocityselector(instrument: InstrumentContext, ki):
    dk_k = 0.1
    dki = ki * dk_k
    dtheta = np.deg2rad(1.0)
    dphi = instrument.divergence_initial
    return dki, dtheta, dphi


instrumentctx = InstrumentContext()

wavelength_incoming = np.linspace(start=3.5, stop=6, num=100) * 1e-10  # m, wavelength
wavenumber_incoming = wavelength_to_wavenumber(wavelength_incoming)

dki, dtheta, dphi = get_resolution_monochromator(instrument=instrumentctx, ki=wavenumber_incoming)
dqx, dqy, dqz = get_resolution_components(ki=wavenumber_incoming, dki=dki, dtheta=dtheta, dphi=dphi)
energy_selection = ENERGY_SELECTION_MONOCHROMATOR
plot_resolution(ki=wavenumber_incoming, dqx=dqx, dqy=dqy, dqz=dqz, energy_selection=energy_selection)

dki, dtheta, dphi = get_resolution_velocityselector(instrument=instrumentctx, ki=wavenumber_incoming)
dqx, dqy, dqz = get_resolution_components(ki=wavenumber_incoming, dki=dki, dtheta=dtheta, dphi=dphi)
energy_selection = ENERGY_SELECTION_VELOCITY_SELECTOR
plot_resolution(ki=wavenumber_incoming, dqx=dqx, dqy=dqy, dqz=dqz, energy_selection=energy_selection)
