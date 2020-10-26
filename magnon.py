from helper import PLANCKS_CONSTANT, MASS_NEUTRON, CONVERSION_JOULE_PER_EV, dispersion_signal, data2range, \
    wavenumber_vector, ZERO_TOL, BOLTZMANN, dirac_delta_approx, THOMSON_SCATT_LENGTH, FACTOR_GAMMA
import numpy as np
from geometry_context import GeometryContext
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize, precision=2)


def if_scattered(ki_vector, kf_vector, latt_const=5 * 1e-10, j=0.1 * 1e-3 * CONVERSION_JOULE_PER_EV, s=1):
    """
    gives back whether the scattering event is allowed or not
    :param ki_vector: m-1, wave-vector of neutrons
    :param kf_vector: m-1, outgoing wave-vector of neutrons
    :param latt_const: m, lattice constant of a cubic structure
    :param j: interaction constant between the nearest two spins
    :param s: spin
    :return: True (if scattering allowed) or False (if not allowed)
    """
    scattering_vector = ki_vector - kf_vector  # wave-vector transfer
    reci_const = 2 * np.pi / latt_const
    hkl = np.round(scattering_vector / reci_const)
    # print(scattering_vector, reci_const, hkl)
    q_vector = scattering_vector - hkl * reci_const
    de = PLANCKS_CONSTANT ** 2 * (np.linalg.norm(ki_vector) ** 2 - np.linalg.norm(kf_vector) ** 2) / (2 * MASS_NEUTRON)
    dd = 2 * j * s * latt_const ** 2
    # print(scattering_vector * 1e-10, reci_const * 1e-10, hkl, q_vector * 1e-10, de * 1e3 / CONVERSION_JOULE_PER_EV,
    #       dd * np.linalg.norm(q_vector) ** 2 * 1e3 / CONVERSION_JOULE_PER_EV)
    if abs(de - dd * np.linalg.norm(q_vector) ** 2) / abs(de) < 5e-2:
        return 1
    else:
        return 0


def qyqz_to_kf(geo_ctx: GeometryContext, de, qy, qz, acute=True):
    ki = geo_ctx.wavenumber_in
    kf = np.sqrt(ki ** 2 - 2 * MASS_NEUTRON * de / PLANCKS_CONSTANT ** 2)
    if abs(qz / kf) <= 1:
        phi = np.arcsin(-qz / kf)
        if abs(qy / (kf * np.cos(phi))) <= 1:
            theta = np.arcsin(-qy / (kf * np.cos(phi)))
            if np.min(geo_ctx.wavenumbers_out) < kf < np.max(
                    geo_ctx.wavenumbers_out):
                if np.min(geo_ctx.pol_angles) < phi < np.max(
                        geo_ctx.pol_angles):
                    if np.min(geo_ctx.azi_nega) < theta < np.max(geo_ctx.azi_nega) or np.min(
                            geo_ctx.azi_posi) < theta < np.max(geo_ctx.azi_posi):
                        if acute:
                            return wavenumber_vector(kf, theta, phi)
                        else:
                            # TODO: correct this for negative angles
                            return wavenumber_vector(kf, np.pi - theta, phi)
                    else:
                        # print("theta", np.rad2deg(theta))
                        return np.empty(3)
                else:
                    # print("phi", np.rad2deg(phi))
                    return np.empty(3)
            else:
                # print("kf", kf * 1e-10)
                return np.empty(3)
        else:
            return np.empty(3)
    else:
        return np.empty(3)


def magnon_energy(wavevector_transfer, latt_const=10 * 1e-10, j=0.1 * 1e-3 * CONVERSION_JOULE_PER_EV, s=1):
    reci_const = 2 * np.pi / latt_const
    hkl = np.round(wavevector_transfer / reci_const)
    # print(scattering_vector, reci_const, hkl)
    magnon_vector = wavevector_transfer - hkl * reci_const
    dd = 2 * j * s * latt_const ** 2
    return dd * np.linalg.norm(magnon_vector) ** 2


def scatt_cross_qxqyde(qq_x, qq_y, hw, ki, temperature=300, latt_const=10e-10, j=0.5e-3 * CONVERSION_JOULE_PER_EV,
                       spin=0.5, resol=0.01, qq_z=None, kf=None, pol_angle=None, mushroom=False):
    if mushroom is True:  # the (Qx,Qy,Qz)-values must satisfy those available in Mushroom
        if abs((hw - (ki ** 2 - kf ** 2) * PLANCKS_CONSTANT ** 2 / (2 * MASS_NEUTRON)) / hw) > resol:
            return None
        else:
            pass
    else:  # no restriction from the Mushroom wave vectors
        kf = np.sqrt(ki ** 2 - 2 * MASS_NEUTRON * hw / PLANCKS_CONSTANT ** 2)
        pol_angle = np.arccos(
            np.linalg.norm(
                [qq_x - ki, qq_y]) / kf)  # it is from -pi/6 to +pi/6, where the cosine function is well-defined
        qq_z = -kf * np.sin(pol_angle)

    # it gives a symmetric pattern iff the ki lays on one reciprocal lattice point
    # print(kf * 1e-10)
    qq_vector = np.array([qq_x, qq_y, qq_z])

    reci_const = 2 * np.pi / latt_const
    hkl = np.round(qq_vector / reci_const)
    kz = qq_vector[-1] / np.linalg.norm(qq_vector)
    magnon_q = qq_vector - hkl * reci_const
    # it has no symmetry in x-direction, since ki and the reciprocal lattice constant are very likely different
    dd = 2 * j * spin * latt_const ** 2
    magnon_hw = dd * np.linalg.norm(magnon_q) ** 2
    beta = 1.0 / (BOLTZMANN * temperature)
    n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
    prefactor = (FACTOR_GAMMA * THOMSON_SCATT_LENGTH) ** 2 * kf / ki * (2 * np.pi / latt_const) ** 3 * spin * (
            1 + kz ** 2)
    debye_waller = 1
    neutrons_lose_energy = dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
    neutrons_gain_energy = dirac_delta_approx(hw, -magnon_hw, resol) * n_q
    return (neutrons_lose_energy + neutrons_gain_energy)  # prefactor * debye_waller *


def scatt_cross_kikf(ki_vector, kf_vector, temperature=300, latt_const=10e-10, j=0.5e-3 * CONVERSION_JOULE_PER_EV,
                     spin=0.5, resol=0.01):
    # it gives a symmetric pattern iff the ki lays on one reciprocal lattice point
    ki, kf = np.linalg.norm(ki_vector), np.linalg.norm(kf_vector)
    hw = PLANCKS_CONSTANT ** 2 * (ki ** 2 - kf ** 2) / (2 * MASS_NEUTRON)
    qq_vector = ki_vector - kf_vector
    reci_const = 2 * np.pi / latt_const
    hkl = np.round(qq_vector / reci_const)
    kz = qq_vector[-1] / np.linalg.norm(qq_vector)
    magnon_q = qq_vector - hkl * reci_const
    # it has no symmetry in x-direction, since ki and the reciprocal lattice constant are very likely different
    dd = 2 * j * spin * latt_const ** 2
    magnon_hw = dd * np.linalg.norm(magnon_q) ** 2
    beta = 1.0 / (BOLTZMANN * temperature)
    n_q = 1.0 / (np.exp(magnon_hw * beta) - 1)
    prefactor = (FACTOR_GAMMA * THOMSON_SCATT_LENGTH) ** 2 * kf / ki * (2 * np.pi / latt_const) ** 3 * spin * (
            1 + kz ** 2)
    debye_waller = 1
    energy_loss = dirac_delta_approx(hw, magnon_hw, resol) * (n_q + 1)
    energy_gain = dirac_delta_approx(hw, -magnon_hw, resol) * n_q
    return (energy_loss + energy_gain)

# # This code is for the real geometry of Mushroom
# geometryctx = GeometryContext()
# ki_vector = np.array([geometryctx.wavenumber_in, 0, 0])
# pol_1d = geometryctx.polar_angles
# kf_1d = geometryctx.wavenumbers
# data_qx = np.array(list(map(lambda j: np.array(list(
#     map(lambda i: geometryctx.vector_transfer(index_pol=j, index_azi=i)[0], range(geometryctx.azi_angles.shape[0])))),
#                             range(pol_1d.shape[0]))))
# data_qy = np.array(list(map(lambda j: np.array(list(
#     map(lambda i: geometryctx.vector_transfer(index_pol=j, index_azi=i)[1], range(geometryctx.azi_angles.shape[0])))),
#                             range(pol_1d.shape[0]))))
# data_qz = np.array(list(map(lambda j: np.array(list(
#     map(lambda i: geometryctx.vector_transfer(index_pol=j, index_azi=i)[2], range(geometryctx.azi_angles.shape[0])))),
#                             range(pol_1d.shape[0]))))
# data_de = np.array(list(map(lambda j: PLANCKS_CONSTANT ** 2 * (
#         geometryctx.wavenumber_in ** 2 - geometryctx.wavenumbers[j] ** 2) / (2 * MASS_NEUTRON),
#                             range(pol_1d.shape[0]))))
# # print(data_de)
# all_peaks = np.array(list(map(lambda j: np.array(list(
#     map(lambda azi: if_scattered(ki_vector=ki_vector, kf_vector=kf2vector(kf=kf_1d[j], azi=azi, pol=pol_1d[j])),
#         geometryctx.azi_angles))), range(pol_1d.shape[0]))))
#
# # plot Qi-E (i=x,y,z) for a generally inelastic scattering
# # plt.figure()
#
#
# plot_qx = data2range(data_qx)
# plot_qy = data2range(data_qy)
# plot_qz = data2range(data_qz)
# plot_e = data2range(data_de)
# print(scan_qx.shape, scan_qy.shape, scan_qz.shape, scan_e.shape)
# scan_qy, scan_qz = np.meshgrid(scan_qy, scan_qz)
# # scan_inten = np.empty_like(scan_qy)
# e_transfer = 2 * 1e-3 * CONVERSION_JOULE_PER_EV

# fig, axs = plt.subplots(1, 3, sharey="all")
# for e_transfer in scan_e:
#     scan_inten = np.array(
#         list(map(lambda i: np.array(list(map(lambda j: if_scattered(ki_vector, qyqz_to_kf(geometryctx, de=e_transfer,
#                                                                                           qy=scan_qy[i, j],
#                                                                                           qz=scan_qz[i, j],
#                                                                                           acute=True)),
#                                              range(scan_qy.shape[1])))), range(scan_qy.shape[0]))))
#
#     plt.figure()
#     plt.contourf(scan_qy * 1e-10, scan_qz * 1e-10, scan_inten)
#     plt.axis("equal")
#     plt.colorbar()
#     plt.savefig("MagnonDispersion_dE{:.1f}meV.pdf".format(e_transfer * 1e3 / CONVERSION_JOULE_PER_EV))
#     plt.close()

# Plot Qx-Qy for magnon at a given dE
# de = 1 * 1e-3 * CONVERSION_JOULE_PER_EV
# e_index = np.argmin(abs(de - data_de))
# data_qx = data_qx[e_index, :]
# data_qy = data_qy[e_index, :]
# all_peaks = all_peaks[e_index, :]
# fig, ax = plt.subplots()
# qx_2d, qy_2d = np.meshgrid(plot_qx, plot_qy)
# peak_2d = dispersion2intensity(range_x=plot_qx, range_y=plot_qy, data_x=data_qx, data_y=data_qy,
#                                intensity=all_peaks)
# cnt = ax.scatter(x=qx_2d * 1e-10, y=qy_2d * 1e-10, c=peak_2d)  # , cmap="coolwarm"
# fig.colorbar(cnt, ax=ax)
# # print(peak_2d)
# # cnt = ax.contourf(wavevector_transfer * 1e-10, energy_transfer * 1e3, intensities_total, cmap="coolwarm")
# plt.show()

# # Plot Qi-dE (i = x, y, z) for magnon
# fig, axs = plt.subplots(1, 3, sharey="all")
# plot_x = [plot_qx, plot_qy, plot_qz]
# data_x = [data_qx, data_qy, data_qz]
# for i in range(axs.ravel().shape[0]):
#     q_2d, e_2d = np.meshgrid(plot_x[i], plot_e)
#     peak_2d = dispersion2intensity(range_x=plot_x[i], range_y=plot_e, data_x=data_x[i], data_y=data_de,
#                                    intensity_2d=all_peaks)
#     print(q_2d.shape, e_2d.shape, peak_2d.shape)
#     cnt = axs[i].scatter(x=q_2d * 1e-10, y=e_2d * 1e3 / CONVERSION_JOULE_PER_EV, c=peak_2d)  # , cmap="coolwarm"
#     fig.colorbar(cnt, ax=axs[i])
#     # print(peak_2d)
#     # cnt = ax.contourf(wavevector_transfer * 1e-10, energy_transfer * 1e3, intensities_total, cmap="coolwarm")
# plt.show()

# # This code is a scan along qy and dE
# qy_1d = np.linspace(-3, 3, num=100) * 1e10
# # qz_1d = np.linspace(0.5, 2, num=1000) * 1e10
# de_1d = np.linspace(-5, 5, num=100) * 1e-3 * CONVERSION_JOULE_PER_EV
# qy_2d, de_2d = np.meshgrid(qy_1d, de_1d)
# inten_2d = np.empty_like(qy_2d)
#
# for i in range(qy_2d.shape[0]):
#     for j in range(qy_2d.shape[1]):
#         ki = MASS_NEUTRON * de_2d[i, j] / (PLANCKS_CONSTANT ** 2 * qy_2d[i, j]) + qy_2d[i, j] / 2.0
#         kf = MASS_NEUTRON * de_2d[i, j] / (PLANCKS_CONSTANT ** 2 * qy_2d[i, j]) - qy_2d[i, j] / 2.0
#         inten_2d[i, j] = if_scattered(ki_vector=np.array([0, ki, 0]), kf_vector=np.array([0, kf, 0]))
#
# plt.figure()
# plt.contourf(qy_2d * 1e-10, de_2d / CONVERSION_JOULE_PER_EV * 1e3, inten_2d)
# plt.colorbar()
# plt.show()
#
# # This code is a scan along qy and qz with kiz=0
# qx_1d = np.linspace(-3, 3, num=1001) * 1e10
# qy_1d = np.linspace(-3, 3, num=999) * 1e10
# # de = 0.7 * 1e-3 * CONVERSION_JOULE_PER_EV
# qx_2d, qy_2d = np.meshgrid(qx_1d, qy_1d)
# hw = 1.5e-3 * CONVERSION_JOULE_PER_EV
# # de_2d = np.array(list(map(lambda j: np.array(list(
# #     map(lambda i: magnon_energy(wavevector_transfer=np.array([qx_2d[i, j], qy_2d[i, j], 0])), range(qx_1d.shape[0])))),
# #                           range(qy_1d.shape[0]))))
#
# cross_section_2d = np.array(list(map(lambda i: np.array(list(
#     map(lambda j: scatt_crosssection(qq_x=qx_2d[i, j], qq_y=qy_2d[i, j], hw=hw, ki=1.5e10), range(qx_2d.shape[1])))),
#                                      range(qx_2d.shape[0]))))

# # inten_2d = np.empty_like(qy_2d)
# #
# # for i in range(qy_2d.shape[0]):
# #     for j in range(qy_2d.shape[1]):
# #         if abs(qy_2d[i, j]) < ZERO_TOL:
# #             kiy = 0
# #             kfy = 0
# #             kfz = np.sqrt(2 * MASS_NEUTRON * de) / PLANCKS_CONSTANT
# #         else:
# #             kiy = MASS_NEUTRON * de / (PLANCKS_CONSTANT ** 2 * qy_2d[i, j]) + (qz_2d[i, j] ** 2 + qy_2d[i, j] ** 2) / (
# #                     2.0 * qy_2d[i, j])
# #             kfy = MASS_NEUTRON * de / (PLANCKS_CONSTANT ** 2 * qy_2d[i, j]) + (qz_2d[i, j] ** 2 - qy_2d[i, j] ** 2) / (
# #                     2.0 * qy_2d[i, j])
# #             kfz = -qz_2d[i, j]
# #         inten_2d[i, j] = if_scattered(ki_vector=np.array([0, kiy, 0]), kf_vector=np.array([0, kfy, kfz]))
# #

# fig, ax = plt.subplots()
# cnt = ax.contourf(qx_2d * 1e-10, qy_2d * 1e-10, de_2d * 1e3 / CONVERSION_JOULE_PER_EV)
# cbar = fig.colorbar(cnt, ax=ax)
# cbar.set_label(r"$\hbar\omega_{magnon}=D|\vec{k}_i-\vec{k}_f-\vec{\tau}|^2$ (meV)")
# ax.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
# ax.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# ax.set_title(r"Magnon dispersion on ($Q_x$, $Q_y$) plane")
# ax.axis("equal")
# plt.tight_layout()
# plt.savefig("Magnon_a5A_J0.1meV.png")
# plt.close(fig)

# fig, ax = plt.subplots()
# cnt = ax.contourf(qx_2d * 1e-10, qy_2d * 1e-10, cross_section_2d)
# cbar = fig.colorbar(cnt, ax=ax)
# # cbar.set_label(r"$\hbar\omega_{magnon}=D|\vec{k}_i-\vec{k}_f-\vec{\tau}|^2$ (meV)")
# # ax.set_xlabel(r"$Q_x=k_{i,x}-k_{f,x}$ ($\AA^{-1}$)")
# # ax.set_ylabel(r"$Q_y=k_{i,y}-k_{f,y}$ ($\AA^{-1}$)")
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# # ax.set_title(r"Magnon dispersion on ($Q_x$, $Q_y$) plane")
# ax.axis("equal")
# plt.tight_layout()
# plt.show()
# # plt.savefig("Magnon_a5A_J0.1meV.png")
# # plt.close(fig)
