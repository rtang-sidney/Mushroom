import numpy as np

import neutron_context as nctx

# Gives all the context for the chopper system of Mushroom spectrometer
# Written by R. Tang (Ran.Tang@frm2.tum.de)

def wavelength2inversevelocity(wavelength):
    velocity = nctx.wavelength2velocity(wavelength)
    return 1.0 / velocity


def inversevelocity2wavelength(inverse_velocity):
    return nctx.velocity2wavelength(1.0 / inverse_velocity)


def rpm2hz(rpm):
    return rpm / 60.0


def hz2rpm(hz):
    return hz * 60.0


def rpm2period(rpm):
    # calculates rotation period of a chopper from revolutions pro minute (RPM)
    # for choppers with multiple openings one must divide the total period by the number of openings
    return 1.0 / rpm2hz(rpm)


def open_angle2time(open_angle, period_rot):
    # gives the open time of a slit with the given open angle
    return open_angle / (2.0 * np.pi) * period_rot


def opentime2angle(open_time, frequency):
    return open_time * 2 * np.pi * frequency


distance12 = 5  # distance chopper 1 & 2
distance1s = 15  # distance chopper 1 & sample
wavelength_min = 1e-10
wavelength_max = 10e-10
tau_min = wavelength2inversevelocity(wavelength_min)
tau_max = wavelength2inversevelocity(wavelength_max)
mechanical_limit_rpm = 1e4
