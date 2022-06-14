import numpy as np
import matplotlib.pyplot as plt
import neutron_context as nctx
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import chopper_context as cctx
from chopper_context import wavelength2inversevelocity, inversevelocity2wavelength, rpm2hz, rpm2period, open_angle2time, \
    opentime2angle, distance12, distance1s, wavelength_min, wavelength_max, tau_min, tau_max

plt.rcParams.update({'font.size': 20})

# This code calculates the chopper parameters to achieve the maximal transmission rate of the two-chopper system for
# the ToF-Mushroom spectrometer with given limits for some of the parameters. At the end, the time when neutrons
# arrive at the sample position is plotted against the neutron wavelengths. This plot is saved in the "TOF" folder if
# there is no frame overlap among neibouring pulses.
# Reference(s): [Paper1] 10.1016/S0168-9002(03)01731-5


CHOPPER_LIMIT = 10000  # rotation limit of a chopper in RMP, mainly due to mechanical stability
OPEN_LIMIT = 5  # limit of the number of openings in a chopper

WAVELENGTH_REF = nctx.wavenumber2wavelength(10e10)  # the reference wavelength to calculate the resolution

TOF_REFERENCE = distance1s / nctx.wavelength2velocity(WAVELENGTH_REF)
RESOLUTION_REF = 2e-2  # the resolution in percentage to be achieved at the reference wavelength
# Using the in-plane resolution at the secondary spectrometer 0.9589406504527286e-2

OPEN_T1_THEORY = RESOLUTION_REF * TOF_REFERENCE
RPM1 = CHOPPER_LIMIT
OPEN1 = OPEN_LIMIT
OPEN2 = 1


def check_overlap(ts_p4, ts_p1):
    """
    Check if two neutron packets overlap with each other
    :param ts_p4: the time of the slowest neutrons with the largest wavelength of a packet
    :param ts_p1:the time of the fastest neutrons with the smallest wavelength of a packet
    :return: boolean, True if neutron packages overlap, False otherwise
    """
    if ts_p4.shape[0] == 0:
        raise RuntimeError("Now neutrons accepted.")
    t = 0
    for i in range(ts_p4.shape[0]):
        for j in range(ts_p4.shape[0]):
            if i != j and ts_p1[i] < ts_p4[j] and ts_p1[i] > ts_p1[j]:
                print(i, ts_p1[i], ts_p4[i], j, ts_p1[j], ts_p4[j])
                t += 1
    if t == 0:
        return False
    else:
        return True


def transmission_rate(n_combi, n2, open_t1, open_t2, tau_min, tau_max, repetition_t2):
    # n_combi: the number of combinations that are transmitted
    # n2: the total number of n2-values
    return n_combi * 0.5 * open_t1 * open_t2 / (distance12 * n2 * repetition_t2 * (tau_max - tau_min))


tau_ref = wavelength2inversevelocity(WAVELENGTH_REF)

repetition_t1 = rpm2period(RPM1) / float(OPEN1)
open_angle1 = int(np.rad2deg(opentime2angle(OPEN_T1_THEORY, frequency=rpm2hz(RPM1))))  # open angle as an integer
open_angle1 = np.deg2rad(open_angle1)
open_t1 = open_angle2time(open_angle=open_angle1, period_rot=repetition_t1 * OPEN1)
repetition_t2 = (distance1s - distance12) * (repetition_t1 / distance12 + tau_max - tau_min)
open_t2 = (repetition_t1 - open_t1) * (1 - distance12 / distance1s)
freq2 = 1.0 / (repetition_t2 * OPEN2)
rpm2 = int(np.floor(freq2 * 60.0))
open_angle2 = int(np.rad2deg(open_t2 * rpm2 * 2.0 * np.pi / 60.0))
open_angle2 = np.deg2rad(open_angle2)
repetition_t2 = rpm2period(rpm=rpm2) / float(OPEN2)
open_t2 = open_angle2time(open_angle=open_angle2, period_rot=repetition_t2 * OPEN2)

print("bl, self_minus, self_plus, bh:", open_t1 * (1 - cctx.distance12 / cctx.distance1s), repetition_t2 - open_t2,
      repetition_t2 + open_t2, (repetition_t1 - open_t1) * (1 - distance12 / distance1s))

w1, w2 = open_t1 / 2.0, open_t2 / 2.0  # the parameters as defined in [Paper1]

print(
    "Chopper1 {:d} RPM, openings {:d} x {:.0f}°; Chopper2 {:d} RPM, openings {:d} x {:.0f}°".format(
        RPM1, OPEN1, np.rad2deg(open_angle1), rpm2, OPEN2, np.rad2deg(open_angle2)))
print("Maximal possible transmission rate", open_t1 * open_t2 / (2.0 * repetition_t1 * repetition_t2))

RPMS = [RPM1, rpm2]
OPENINGS = [OPEN1, OPEN2]
open_angles = [open_angle1, open_angle2]

wavelengths = np.linspace(wavelength_min, wavelength_max, 100)

print(open_t2 * distance1s / distance12 + open_t1 * (distance1s / distance12 - 1))
print(repetition_t1 * (distance1s / distance12 - 1), repetition_t2 - (tau_max - tau_min) * (distance1s - distance12))

n2_1d = np.arange(10)
n1_1d = np.arange(int(repetition_t2 / repetition_t1 * n2_1d.shape[0]))
n1_2d, n2_2d = np.meshgrid(n1_1d, n2_1d)
print(n1_1d.shape, n2_1d.shape, repetition_t2 / repetition_t1)

n1n2 = n2_2d * repetition_t2 - n1_2d * repetition_t1
tau_accepted = tau_min + n1n2 / distance12
print(tau_accepted.shape)

boundary_low = tau_accepted >= tau_min
boundary_high = tau_accepted <= tau_max
interest = np.logical_and(boundary_low, boundary_high)

n1_interest, n2_interest = n1_2d[interest], n2_2d[interest]
tau_accepted = tau_accepted[interest]

tau_p1 = tau_min + (n2_interest * repetition_t2 - n1_interest * repetition_t1 - w1 - w2) / distance12
tau_p2 = tau_min + (n2_interest * repetition_t2 - n1_interest * repetition_t1 + w1 - w2) / distance12
tau_p3 = tau_min + (n2_interest * repetition_t2 - n1_interest * repetition_t1 - w1 + w2) / distance12
tau_p4 = tau_min + (n2_interest * repetition_t2 - n1_interest * repetition_t1 + w1 + w2) / distance12

ts_p1 = tau_p1 * distance1s + repetition_t1 * n1_interest + w1
ts_p2 = tau_p2 * distance1s + repetition_t1 * n1_interest - w1
ts_p3 = tau_p3 * distance1s + repetition_t1 * n1_interest + w1
ts_p4 = tau_p4 * distance1s + repetition_t1 * n1_interest - w1

if_overlap = check_overlap(ts_p4, ts_p1)
print("Overlap: {}".format(if_overlap))

index_ref = np.argmin(np.abs(tau_accepted - tau_ref))
print("For ki = 1.1 AA-1, n1 = {}, n2 = {}".format(n1_interest[index_ref], n2_interest[index_ref]))

lambda_p1 = inversevelocity2wavelength(tau_p1)
lambda_p2 = inversevelocity2wavelength(tau_p2)
lambda_p3 = inversevelocity2wavelength(tau_p3)
lambda_p4 = inversevelocity2wavelength(tau_p4)

fig, ax = plt.subplots(figsize=(10, 6))
color = next(ax._get_lines.prop_cycler)['color']
for i in range(tau_accepted.shape[0]):
    parallelogram = Polygon(np.array(
        [[lambda_p1[i] * 1e10, ts_p1[i]], [lambda_p2[i] * 1e10, ts_p2[i]],
         [lambda_p4[i] * 1e10, ts_p4[i]], [lambda_p3[i] * 1e10, ts_p3[i]]]), color=color)
    ax.add_patch(parallelogram)
ax.plot()

ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
ax.set_ylabel("Time at sample (s)")
ax.tick_params(axis="both", top=True, right=True, direction="in")

ax2 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(1.10, 0, 0.3, 0.4), bbox_transform=ax.transAxes,
                 loc="lower left")
for i in [1, 2]:
    parallelogram = Polygon(np.array(
        [[lambda_p1[i] * 1e10, ts_p1[i]], [lambda_p2[i] * 1e10, ts_p2[i]], [lambda_p4[i] * 1e10, ts_p4[i]],
         [lambda_p3[i] * 1e10, ts_p3[i]]]))
    ax2.add_patch(parallelogram)
    ax2.plot()
ax2.hlines(y=(ts_p4[2] + ts_p1[1]) / 2.0, xmin=lambda_p1[2] * 1e10, xmax=lambda_p4[1] * 1e10, linestyles="dashed")
# ax2.set_xlabel(r"Wavelength ($\AA$)")
# ax2.set_ylabel("Time at sample (s)")
ax2.tick_params(axis="both", top=True, right=True, direction="in")

# ax.indicate_inset_zoom(ax2, edgecolor="black")
rectpatch, connects = ax.indicate_inset_zoom(ax2, edgecolor="black")
connects[0].set_visible(True)
connects[1].set_visible(True)
connects[2].set_visible(False)
connects[3].set_visible(False)

if if_overlap is False:
    rate = transmission_rate(n1_interest.shape[0], n2_1d.shape[0], open_t1, open_t2, tau_min, tau_max, repetition_t2)

    ax.text(1.01, 1, "Chopper 1\n{:d} RPM\n{:d} x {:.0f}°\n\n".format(RPM1, OPEN1, np.rad2deg(
        open_angle1)) + "Chopper 2\n{:d} RPM\n{:d} x {:.0f}°\n\n".format(rpm2, OPEN2, np.rad2deg(
        open_angle2)), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    print("Transmission rate: {:.2f}%".format(rate * 1e2))
    print("Filename:",
          "TOF//{:d}_{:d}x{:.0f}_{:d}_{:d}x{:.0f}_{:d}m.png".format(RPM1, OPEN1, np.rad2deg(open_angle1), rpm2, OPEN2,
                                                                    np.rad2deg(open_angle2), distance12))
    fig.savefig(
        "TOF//{:d}_{:d}x{:.0f}_{:d}_{:d}x{:.0f}_{:d}m.png".format(RPM1, OPEN1, np.rad2deg(open_angle1), rpm2, OPEN2,
                                                                  np.rad2deg(open_angle2), distance12),
        bbox_inches='tight')
# plt.show()
plt.close(fig)
