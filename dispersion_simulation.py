import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
import geometry_calculation as geo
import neutron_context as nctx
import global_context as glb
import instrument_context as instr
from magnonmodel import MagnonModel

plt.rcParams.update({'font.size': 18})

# Simulates the dispersion relation along within a given hkl-plane

HKL_VALUES = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 2, 1),
              (2, 2, 1)]  # (0, 0, 1),
POINT_INTEREST = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def q_mushroom_check(q_vector, kf, hw, resol=0.025e10):
    """
    calculates the polar and azimuthal angles and check if they are available on Mushroom
    :param q_vector: 3D wavevector transfer
    :param kf: scalar, outgoing wvelength
    :return: polar, azimuthal angles and the incoming wavelength if available, None otherwise
    """
    sin_pol = -q_vector[-1] / kf
    if np.sin(instr.pol_minus_rad) <= sin_pol <= np.sin(instr.pol_plus_rad):
        pol_ang = np.arcsin(sin_pol)
        sin_azi = -q_vector[1] / (kf * np.cos(pol_ang))
        boundary_closer = min(np.sin(instr.azi_min_rad), np.sin(instr.azi_max_rad))
        if -1 <= sin_azi <= -boundary_closer or boundary_closer < sin_azi < 1:
            ki_from_hw = nctx.energy2wavenumber(hw + nctx.wavenumber2energy(kf))
            azi_ang = np.arcsin(sin_azi)
            ki = q_vector[0] + kf * np.cos(pol_ang) * np.cos(azi_ang)
            if instr.wavenumber_in_min <= ki <= instr.wavenumber_in_max:
                if abs(ki_from_hw - ki) < resol:
                    print(ki, pol_ang, azi_ang)
                    return True, ki
            else:
                return False, None
        else:
            return False, None
    else:
        return False, None


def qrlu2corr(qrlu, hw, latt):
    q_vector = nctx.rlu2q(qrlu, latt)
    for steps in range(instr.sa_rot_steps):
        q_vector = geo.rotation_3d(angle=instr.sa_rot_stepsize, rot_axis=(0, 0, 1), vector=q_vector)
        for order in [1, 2]:
            kf = mush_ctx.wavenumber_f * order
            print(q_vector, kf, hw)
            scattered, ki = q_mushroom_check(q_vector, kf, hw)
            if scattered is True:
                corr_func = magnonmdl.corr_func_from_q_hw(hw=nctx.wavenumber2energy(ki) - nctx.wavenumber2energy(kf),
                                                          q_rlu=qrlu)
                return corr_func
            else:
                pass
    return None


def xi2corr(xi1, xi2, hkl1, hkl2, hw):
    hkl1 = np.array(hkl1)
    hkl2 = np.array(hkl2)
    hkl = xi1 * hkl1 + xi2 * hkl2
    corr_func = qrlu2corr(qrlu=hkl, hw=hw, latt=latt_const)
    return corr_func


def plot_plane(hkl1, hkl2, hw):
    xi1 = np.linspace(-4, 4, num=200)
    xi2 = np.linspace(-4, 4, num=200)
    xi1, xi2 = np.meshgrid(xi1, xi2)
    corr_2d = np.array(list(
        map(lambda i: np.array(list(map(lambda j: xi2corr(xi1[i, j], xi2[i, j], hkl1, hkl2, hw), range(xi1.shape[1])))),
            range(xi1.shape[0]))))
    fig, ax = plt.subplots()
    ax.contourf(xi1, xi2, corr_2d)
    ax.axis("equal")
    ax.set_xlabel("({}, {}, {}) (rlu)".format(*hkl1))
    ax.set_ylabel("({}, {}, {}) (rlu)".format(*hkl2))
    ax.set_title("hw={:.2f} meV".format(nctx.joule2mev(hw)))
    fig.savefig(
        "{}Dispersion_hw{:.1f}_({:d}{:d}{:d})_({:d}{:d}{:d})".format(glb.path_performance, nctx.joule2mev(hw), *hkl1,
                                                                     *hkl2))
    plt.close(fig)


def qx2qy(qx, qz, ki, kf):
    """
    calculates the polar and azimuthal angles and check if they are available on Mushroom
    :param q_vector: 3D wavevector transfer
    :param kf: scalar, outgoing wvelength
    :return: polar, azimuthal angles and the incoming wavelength if available, None otherwise
    """
    sin_pol = -qz / kf
    if np.sin(instr.pol_minus_rad) <= sin_pol <= np.sin(instr.pol_plus_rad):
        pol_ang = np.arcsin(sin_pol)
        cos_azi = (ki - qx) / (kf * np.cos(pol_ang))
        boundary = max(np.cos(instr.azi_min_rad), np.cos(instr.azi_max_rad))
        if abs(cos_azi) <= boundary:
            azi_ang = np.arccos(cos_azi)
            qy = -kf * np.cos(pol_ang) * np.sin(azi_ang)
            return qy
        else:
            return None
    return None


def qxqz2corr(qx, qz, kf, hw):
    ki = nctx.energy2wavenumber(hw + nctx.wavenumber2energy(kf))
    qy = qx2qy(nctx.rlu2q(qx, latt_const), nctx.rlu2q(qz, latt_const), ki, kf)
    if qy:
        qy = nctx.q2rlu(qy, latt_const)
        qrlu = np.array([qx, qy, qz])
        corr_func = magnonmdl.corr_func_from_q_hw(hw=hw, q_rlu=qrlu)
        return qrlu, corr_func
    else:
        return None, None


def plot_qxqy(hkl1, hkl2, hw):
    qx = np.linspace(-3, 3, num=200)
    qz = 0
    for order in [1, 2]:
        kf = mush_ctx.wavenumber_f * order


latt_const = 4.5 * 1e-10
magnonmdl = MagnonModel(model_name=glb.magnon_default, latt_const=latt_const, spin_coupling=nctx.mev2joule(0.3))
mush_ctx = MushroomContext()

hkl1 = HKL_VALUES[0]
hkl2 = HKL_VALUES[1]
hw = nctx.mev2joule(2.0)
# print(hkl1, hkl2)
plot_plane(hkl1, hkl2, hw=hw)
