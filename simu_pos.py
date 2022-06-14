import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
import geometry_calculation as geo
import neutron_context as nctx
import global_context as glb
import instrument_context as instr
from magnonmodel import MagnonModel
from matplotlib import colors

plt.rcParams.update({'font.size': 18})


# This code is probably not in use any longer


def one_kf(ki, kf, sa_rot, an_azi, an_pol):
    qx = ki - kf * np.cos(an_pol) * np.cos(an_azi)
    qy = -kf * np.cos(an_pol) * np.sin(an_azi)
    qz = -kf * np.sin(an_pol)
    qx, qy = geo.rotation_around_z(rot_angle=-sa_rot, old_x=qx, old_y=qy)
    qx = nctx.q2rlu(q_value=qx, l_const=latt_const)
    qy = nctx.q2rlu(q_value=qy, l_const=latt_const)
    qz = nctx.q2rlu(q_value=qz, l_const=latt_const)
    hw = nctx.wavenumber2energy(ki) - nctx.wavenumber2energy(kf)
    corr_func = magnonmdl.corr_func_from_q_hw(hw=hw, q_rlu=np.array([qx, qy, qz]))
    inten = kf / ki * corr_func
    print(inten, qx, qy, qz, nctx.joule2mev(hw))
    return inten


def pos2inten(sa_deg, an_deg, an_pol):
    sa_rot = np.deg2rad(sa_deg)
    an_azi = np.deg2rad(an_deg)
    kf_range = nctx.wavelength2wavenumber(lf_range)
    inten_all = np.zeros(0)
    for kf in kf_range:
        inten = one_kf(ki, kf, sa_rot, an_azi, an_pol)
        inten_all = np.append(inten_all, inten)
    fig, ax = plt.subplots()
    inten_all = np.where(inten_all > 1, inten_all, 1)
    ax.plot(lf_range * 1e10, inten_all, "o-")
    ax.set_xlabel("Wavelength (AA)")
    ax.set_ylabel("Intensity")
    ax.tick_params(axis="both", direction="in")
    ax.set_yscale("log")
    plt.show()
    # fig.savefig("Wavelength_sa{:d}_an{:d}.png".format(int(sa_deg), int(an_deg)))
    # plt.close(fig)
    print("Position sa{:d} an{:d}".format(int(sa_deg), int(an_deg)))


latt_const = 4.5 * 1e-10
magnonmdl = MagnonModel(model_name=glb.magnon_default, latt_const=latt_const, spin_coupling=nctx.mev2joule(0.3))
mush_ctx = MushroomContext()
ki = 1.6e10
lf_range = np.linspace(2, 8, num=70) * 1e-10
file_param = glb.path_mcstas + "ki1.6_k-1.00_l-0.12.dat"
params = np.loadtxt(file_param, delimiter=",").transpose()

for i in range(params.shape[1]):
    if i + 2 == 11:
        sa_deg = params[1, i]
        an_deg = params[2, i]
        an_y = params[3, i]
        an_x = params[4, i]
        an_pol = np.arctan2(an_y, an_x)
        pos2inten(sa_deg, an_deg, an_pol)
