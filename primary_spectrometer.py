import matplotlib.pyplot as plt
import numpy as np

import instrument_context
import neutron_context as nctx
import instrument_context as instr
import geometry_calculation as geo
import global_context as glb
from mushroom_context import MushroomContext

plt.rcParams.update({'font.size': 18})

TYPE_MONOCHROMATOR = "Monochromator"
TYPE_VELOCITY_SELECTOR = "VelocitySelector"
TYPE_TOF = "ToF"
TYPES_PRIMARY = [TYPE_MONOCHROMATOR, TYPE_VELOCITY_SELECTOR, TYPE_TOF]

PATH_RESOLUTION = "Resolution\\"
FILENAME_PREFIX = "Resolution_Primary_"

AXIS_AZIMUTHAL = "x"
AXIS_POLAR = "y"

"""
[Paper1]: Demmel2014 http://dx.doi.org/10.1016/j.nima.2014.09.019
[Paper2]: Keller2002 https://doi.org/10.1007/s003390101082

"""


def get_filename(type_primary):
    if type_primary in TYPES_PRIMARY:
        return FILENAME_PREFIX + type_primary + ".pdf"
    else:
        raise RuntimeError("Invalid type of the primary spectrometer.")


def resolution_mono(ki):
    """
    calculates the resolution of the primary spectrometer with a monochromator
    :param 
    :param ki: incoming wavenumber
    :return: deviations of the components in the sequence of dki (wave number), dtheta (azimuthal angle), dphi (polar
    angle)
    """

    def divergence_mono(axis):
        # distance_ms: monochromator-sample distance
        divergence_in = instr.diver_nl
        if axis == AXIS_AZIMUTHAL:
            divergence_out = geo.angle_triangle(a=instr.distance_ms, c=instr.sam_dia)
        elif axis == AXIS_POLAR:
            divergence_out = geo.angle_triangle(a=instr.distance_ms, c=instr.sam_height)
        else:
            raise RuntimeError("Invalid axis given.")
        return divergence_in, divergence_out

    def angular_spread_monochromator(axis):
        eta = instr.moasic_an  # mosaic
        alpha_i, alpha_f = divergence_mono(axis=axis)  # incoming and outgoing divergence
        numerator = alpha_i ** 2 * alpha_f ** 2 + eta ** 2 * alpha_i ** 2 + eta ** 2 * alpha_f ** 2
        denominator = 4 * eta ** 2 + alpha_i ** 2 + alpha_f ** 2
        return np.sqrt(numerator / denominator)

    def monochromator_twotheta():
        nonlocal ki
        return nctx.bragg_wavenumber2twotheta(wavenumber=ki, lattice_distance=instr.interplanar_pg002)

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
        return min(np.deg2rad(1.6), geo.angle_triangle(instr.distance_ms, instr.sam_height))

    def get_spread_azimuthal():
        return min(np.deg2rad(1.6), angular_spread_monochromator(axis=AXIS_AZIMUTHAL))

    dki = get_uncertainty_ki()
    dtheta = get_spread_azimuthal()
    dphi = get_spread_polar()
    return dki, dtheta, dphi


def resolution_tof(ki):
    distance1s = 15
    velocity = nctx.wavenumber2velocity(ki)
    tof = distance1s / velocity
    open_angle = np.deg2rad(2)

    rpm = 10000
    period = 1.0 / (rpm / 60.0)
    open_time = open_angle / (2.0 * np.pi) * period

    rel_resol = open_time / tof

    dki = ki * rel_resol
    dtheta = np.deg2rad(1.0)
    dphi = instr.diver_nl

    # return dki, dtheta, dphi

    return dki, dtheta, dphi


def resolution_primary(ki, comp_type):
    """
    recalculates the resolution and gives the components in x (horizontal), y (vertical) and z (along the ki) directions
    :param ki: wave number
    :param comp_type: type of the component
    :return: all three components of the Q-resolution in the sequence of x, y, z
    """
    if comp_type == TYPE_MONOCHROMATOR:
        uncertain_ki, uncertain_theta, uncertain_phi = resolution_mono(ki=ki)
    elif comp_type == TYPE_VELOCITY_SELECTOR:
        uncertain_ki, uncertain_theta, uncertain_phi = resolution_v_selector(ki=ki)
    elif comp_type == TYPE_TOF:
        uncertain_ki, uncertain_theta, uncertain_phi = resolution_tof(ki=ki)
    else:
        raise ValueError("Unknown component type.")
    dqx = ki * np.tan(uncertain_theta)
    dqy = ki * np.tan(uncertain_phi)
    dqz = uncertain_ki
    # dqz = ki * (1 - np.cos(uncertain_theta) * np.cos(uncertain_phi))
    dq_q = uncertain_ki / ki
    return dqx, dqy, dqz, dq_q


def plot_resolution(ki, dqx, dqy, dqz, dq_relative, comp_type):
    if comp_type not in TYPES_PRIMARY:
        raise ValueError("Wrong type of the primary spectrometer is given.")
    else:
        pass
    filename = get_filename(type_primary=comp_type)
    filename = "".join([PATH_RESOLUTION, filename])
    fig, ax = plt.subplots()
    ax.plot(ki * 1e-10, dqx * 1e-10, color="blue", label="x: horizontal")
    ax.plot(ki * 1e-10, dqy * 1e-10, color="red", label="y: vertical")
    ax.plot(ki * 1e-10, dqz * 1e-10, color="darkgoldenrod", label="z: along $k_i$")
    # print(dqz * 1e-10)

    ax.tick_params(axis="both", direction="in")
    # ax.legend(labelcolor=["blue", "red", "darkgoldenrod"], loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.5)
    ax.set_xlabel(r"Wavenumber $k_i$ ($\AA^{-1}$)")
    ax.set_ylabel(r"Uncertainty $\Delta k_{i,\alpha}$ ($\AA^{-1}$), $\alpha=x,y,z$")
    # ax.grid()
    ax.set_title("Primary spectrometer - {:s}".format(comp_type))
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    # ax2.plot(ki * 1e-10, dqz / ki * 1e2, color=colour_ax2)
    ax2.plot(ki * 1e-10, dq_relative * 1e2, color=colour_ax2)
    # ax2.legend(["Relative uncertainty"], loc='upper right', bbox_to_anchor=(1, 0.5), labelcolor=colour_ax2,
    #            framealpha=0.5)
    ax2.tick_params(axis="y", direction="in", color=colour_ax2)
    ax2.set_ylabel(r"$\dfrac{\Delta k_i}{k_i}$ * 100%", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)

    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('pdf', 'png'), bbox_inches='tight')
    print("{:s} plotted.".format(filename))


def resolution_v_selector(ki):
    dk_k = 0.1
    dki = ki * dk_k
    dtheta = np.deg2rad(1.0)
    dphi = instr.diver_nl
    return dki, dtheta, dphi


# # kf = GeometryContext(side="same").wavenumbers
# # wavelength_incoming = GeometryContext(side="same").wavenumbers * 1e-10  # m, wavelength
# wavenumber_in = MushroomContext().wavenumbers_out  # use the same possible outgoing wavenumbers
wavenumber_in = np.linspace(instrument_context.wavenumber_in_min, instrument_context.wavenumber_in_max, num=100)

uncertain_qx, uncertain_qy, uncertain_qz, uncertain_relative = resolution_primary(ki=wavenumber_in,
                                                                                  comp_type=TYPE_MONOCHROMATOR)
plot_resolution(ki=wavenumber_in, dqx=uncertain_qx, dqy=uncertain_qy, dqz=uncertain_qz, dq_relative=uncertain_relative,
                comp_type=TYPE_MONOCHROMATOR)

uncertain_qx, uncertain_qy, uncertain_qz, uncertain_relative = resolution_primary(ki=wavenumber_in,
                                                                                  comp_type=TYPE_VELOCITY_SELECTOR)
plot_resolution(ki=wavenumber_in, dqx=uncertain_qx, dqy=uncertain_qy, dqz=uncertain_qz, dq_relative=uncertain_relative,
                comp_type=TYPE_VELOCITY_SELECTOR)

wavelength_in = np.linspace(1e-10, 10e-10, num=100)
wavenumber_in = nctx.wavelength2wavenumber(wavelength_in)
uncertain_qx, uncertain_qy, uncertain_qz, uncertain_relative = resolution_primary(ki=wavenumber_in, comp_type=TYPE_TOF)
plot_resolution(ki=wavenumber_in, dqx=uncertain_qx, dqy=uncertain_qy, dqz=uncertain_qz, dq_relative=uncertain_relative,
                comp_type=TYPE_TOF)
