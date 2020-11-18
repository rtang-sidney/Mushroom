import matplotlib.pyplot as plt
import numpy as np
import neutron_context as neutron
import instrument_context as instr
import geometry_calculation as geo
from mushroom_context import MushroomContext

ZERO_TOL = 1e-6
ENERGY_CUT_MONOCHROMATOR = "Monochromator"
ENERGY_CUT_VELOCITY_SELECTOR = "VelocitySelector"
FILENAME_PREFIX = "Resolution_Primary_"
FILENAME_MONOCHROMATOR = "Resolution_Primary_Monochromator.pdf"
FILENAME_VELOCITY_SELECTOR = "Resolution_Primary_VelocitySelector.pdf"
AXIS_AZIMUTHAL = "x"
AXIS_POLAR = "y"

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082

"""


def get_filename(energy_selection_type):
    if isinstance(energy_selection_type, str):
        # return FILENAME_PREFIX + energy_selection_type + ".pdf"
        return FILENAME_PREFIX + energy_selection_type + "Test.pdf"
    else:
        raise RuntimeError("Invalid type of the energy selection variable given.")


def get_resolution_monochromator(ki):
    """
    calculates the resolution of the primary spectrometer with a monochromator
    :param 
    :param ki: incoming wavenumber
    :return: deviations of the components in the sequence of dki (wave number), dtheta (azimuthal angle), dphi (polar
    angle)
    """

    def divergence_mono(axis):
        # distance_ms: monochromator-sample distance
        divergence_in = instr.divergence_initial
        if axis == AXIS_AZIMUTHAL:
            divergence_out = geo.angle_triangle(a=instr.distance_ms, c=instr.sample_diameter)
        elif axis == AXIS_POLAR:
            divergence_out = geo.angle_triangle(a=instr.distance_ms, c=instr.sample_height)
        else:
            raise RuntimeError("Invalid axis given.")
        return divergence_in, divergence_out

    def angular_spread_monochromator(axis):
        eta = instr.moasic_analyser  # mosaic
        alpha_i, alpha_f = divergence_mono(axis=axis)  # incoming and outgoing divergence
        numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
        denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2
        return np.sqrt(numerator / denominator)

    def monochromator_twotheta():
        nonlocal ki
        return neutron.bragg_wavenumber2angle(wavenumber=ki, lattice_distance=instr.lattice_distance_pg002)

    def get_uncertainty_ki():
        nonlocal ki
        # gives the deviation of the wave-number by means of the Bragg's law
        dtheta_mono = angular_spread_monochromator(axis=AXIS_AZIMUTHAL)
        twotheta_mono = monochromator_twotheta()
        dki_bragg = ki * np.linalg.norm([instr.deltad_d, dtheta_mono / np.tan(twotheta_mono / 2.0)])
        # print(dtheta_mono, np.rad2deg(twotheta_mono), ki * 1e-10, dki_bragg * 1e-10)
        return abs(dki_bragg)

    def get_spread_polar():
        # the spread of the polar angle is given simply by the divergence in both directions,
        # since there is no scattering angles to be taken into consideration in this direction
        return min(np.deg2rad(1.6), geo.angle_triangle(instr.distance_ms, instr.sample_height))

    def get_spread_azimuthal():
        return min(np.deg2rad(1.6), angular_spread_monochromator(axis=AXIS_AZIMUTHAL))

    dki = get_uncertainty_ki()
    dtheta = get_spread_azimuthal()
    dphi = get_spread_polar()
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
    ax.legend(("x: horizontal", "y: vertical", r"z: along $k_i$"))
    ax.set_xlabel(r"Incoming wavenumber $|k_i|$ ($\AA^{-1}$)")
    ax.set_ylabel(r"Component uncertainties $\Delta k_{i,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    ax.grid()
    ax.set_title("Resolution of the primary spectrometer with {}".format(energy_selection))
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    ax2.plot(ki * 1e-10, dqz / ki * 1e2, '1', color=colour_ax2)
    ax2.legend(["Relative uncertainty"], loc='lower left', bbox_to_anchor=(0, 0.65))
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"Relative uncertainty $\dfrac{|\Delta k_i|}{|k_i|}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)

    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('pdf', 'png'), bbox_inches='tight')
    print("{:s} plotted.".format(filename))


def resolution_v_selector(ki):
    dk_k = 0.1
    dki = ki * dk_k
    dtheta = np.deg2rad(1.0)
    dphi = instr.divergence_initial
    return dki, dtheta, dphi


# kf = GeometryContext(side="same").wavenumbers
# wavelength_incoming = GeometryContext(side="same").wavenumbers * 1e-10  # m, wavelength
wavenumber_in = MushroomContext().wavenumber_in

uncertain_ki, uncertain_theta, uncertain_phi = get_resolution_monochromator(ki=wavenumber_in)
uncertain_qx, uncertain_qy, uncertain_qz = get_resolution_components(ki=wavenumber_in, dki=uncertain_ki,
                                                                     dtheta=uncertain_theta,
                                                                     dphi=uncertain_phi)
plot_resolution(ki=wavenumber_in, dqx=uncertain_qx, dqy=uncertain_qy, dqz=uncertain_qz,
                energy_selection=ENERGY_CUT_MONOCHROMATOR)

uncertain_ki, uncertain_theta, uncertain_phi = resolution_v_selector(ki=wavenumber_in)
uncertain_qx, uncertain_qy, uncertain_qz = get_resolution_components(ki=wavenumber_in, dki=uncertain_ki,
                                                                     dtheta=uncertain_theta,
                                                                     dphi=uncertain_phi)
plot_resolution(ki=wavenumber_in, dqx=uncertain_qx, dqy=uncertain_qy, dqz=uncertain_qz,
                energy_selection=ENERGY_CUT_VELOCITY_SELECTOR)
