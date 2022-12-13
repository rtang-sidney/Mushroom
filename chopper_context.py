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


def tau_parallelogram(n1, n2, repetition_t1, repetition_t2, w1, w2):
    tau_p1 = tau_min + (n2 * repetition_t2 - n1 * repetition_t1 - w1 - w2) / distance12
    tau_p2 = tau_min + (n2 * repetition_t2 - n1 * repetition_t1 + w1 - w2) / distance12
    tau_p3 = tau_min + (n2 * repetition_t2 - n1 * repetition_t1 - w1 + w2) / distance12
    tau_p4 = tau_min + (n2 * repetition_t2 - n1 * repetition_t1 + w1 + w2) / distance12
    return tau_p1, tau_p2, tau_p3, tau_p4


def t_parallelogram(tau_p1, tau_p2, tau_p3, tau_p4, n1, repetition_t1, w1):
    ts_p1 = tau_p1 * distance1s + repetition_t1 * n1 + w1
    ts_p2 = tau_p2 * distance1s + repetition_t1 * n1 - w1
    ts_p3 = tau_p3 * distance1s + repetition_t1 * n1 + w1
    ts_p4 = tau_p4 * distance1s + repetition_t1 * n1 - w1
    return ts_p1, ts_p2, ts_p3, ts_p4


def remove_overlap(repetition_t1, repetition_t2, w1, w2, n2_1d_len=100, parallelogram=False):
    n2_1d = np.arange(n2_1d_len)
    n1_1d = np.arange(int(repetition_t2 / repetition_t1 * n2_1d_len))
    n1_2d, n2_2d = np.meshgrid(n1_1d, n2_1d)
    # print(n1_1d.shape, n2_1d.shape, repetition_t2 / repetition_t1)

    n1n2 = n2_2d * repetition_t2 - n1_2d * repetition_t1
    tau_pass = tau_min + n1n2 / distance12
    # print(tau_pass.shape)

    boundary_low = tau_pass >= tau_min
    boundary_high = tau_pass <= tau_max
    pass_index = np.logical_and(boundary_low, boundary_high)

    n1_pass, n2_pass = n1_2d[pass_index], n2_2d[pass_index]
    tau_pass = tau_pass[pass_index]

    tau_p1, tau_p2, tau_p3, tau_p4 = tau_parallelogram(n1_pass, n2_pass, repetition_t1, repetition_t2, w1, w2)
    ts_p1, ts_p2, ts_p3, ts_p4 = t_parallelogram(tau_p1, tau_p2, tau_p3, tau_p4, n1_pass, repetition_t1, w1)

    if_overlap = check_overlap(ts_p4, ts_p1)
    # print("Overlap: {}".format(if_overlap))
    if parallelogram is True:
        return n1_pass, n2_pass, tau_pass, tau_p1, tau_p2, tau_p3, tau_p4, ts_p1, ts_p2, ts_p3, ts_p4, if_overlap
    else:
        return n1_pass, n2_pass, if_overlap


def check_overlap(ts_p4, ts_p1):
    """
    Check if two neutron packets overlap with each other
    :param ts_p4: the time of the slowest neutrons with the largest wavelength of a packet
    :param ts_p1:the time of the fastest neutrons with the smallest wavelength of a packet
    :return: boolean, True if neutron packages overlap, False otherwise
    """
    if ts_p4.shape[0] == 0:
        raise RuntimeError("No neutrons accepted.")
    t = 0
    for i in range(ts_p4.shape[0]):
        for j in range(ts_p4.shape[0]):
            if i != j and ts_p1[i] < ts_p4[j] and ts_p1[i] > ts_p1[j]:
                # print(i, ts_p1[i], ts_p4[i], j, ts_p1[j], ts_p4[j])
                t += 1
    if t == 0:
        return False
    else:
        return True


def transmission_rate(n_combi, n2, open_t1, open_t2, repetition_t2):
    # n_combi: the number of combinations that are transmitted
    # n2: the total number of n2-values
    rate = n_combi * 0.5 * open_t1 * open_t2 / (distance12 * n2 * repetition_t2 * (tau_max - tau_min))
    # print(rate)
    return rate


def limit_frame_overlap(k, d12, d1s, phi1, phi2, n1, n2, tau_max, tau_min):
    return phi1 * nctx.habr * k * (tau_max - tau_min) / (2 * np.pi * nctx.mass_neutron * d1s * (
            2 * np.pi / (phi2 * n1 * n2 * d12 - 1.0 / (n1 * d12) - phi1 / (phi2 * n1 * d1s))))


def limit_mechanical(k, phi1, f1, d1s):
    return phi1 * nctx.habr * k / (2 * np.pi * nctx.mass_neutron * d1s * f1)


distance12 = 5  # distance chopper 1 & 2
distance1s = 15  # distance chopper 1 & sample
wavelength_min = 1e-10
wavelength_max = 10e-10
tau_min = wavelength2inversevelocity(wavelength_min)
tau_max = wavelength2inversevelocity(wavelength_max)
mechanical_limit_rpm = 10000
open_limit = 5  # limit of the number of openings in a chopper

