import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
import geometry_calculation as geo
import neutron_context as neutron
import format_context as fmt

plt.rcParams.update({'font.size': 18})


def mushroom_wavevector_transfer(geo_ctx: MushroomContext, ki):
    ki_vector = (ki, 0, 0)
    azi_angles, kf_norms = np.meshgrid(geo_ctx.azi_angles, geo_ctx.wavenumbers_out)
    polar_angles = np.meshgrid(geo_ctx.azi_angles, geo_ctx.pol_angles)[1]
    azi_angles = azi_angles.flatten()
    kf_norms = kf_norms.flatten()
    polar_angles = polar_angles.flatten()
    kf_vectors = np.empty((3, kf_norms.shape[0]))
    kf_vectors[0, :] = kf_norms * np.cos(polar_angles) * np.cos(azi_angles)
    kf_vectors[1, :] = kf_norms * np.cos(polar_angles) * np.sin(azi_angles)
    kf_vectors[2, :] = kf_norms * np.sin(polar_angles)
    mushroom_q_vectors = np.apply_along_axis(neutron.wavevector_transfer, axis=0, arr=kf_vectors,
                                             wavevector_in=ki_vector)
    return mushroom_q_vectors


def mushroom_e_transfer(geo_ctx: MushroomContext, ki):
    #  values of all the qx, qy, qz in each sample rotation
    mushroom_hw = np.array(list(map(lambda i: np.array(list(map(
        lambda j: neutron.planck_constant ** 2 * (ki ** 2 - geo_ctx.wavenumbers_out[i] ** 2) / (
                2 * neutron.mass_neutron), range(geo_ctx.azi_angles.shape[0])))),
                                    range(geo_ctx.pol_angles.shape[0]))))
    return mushroom_hw.flatten()


def dispersion_mushroom(geo_ctx: MushroomContext):
    # separate the process of calculations and plotting
    def calc_per_ki(ki_now):
        """
        Do the calculations for each value of the incoming wavenumber ki
        It gives only the values from Mushroom, independent of the sample
        :param ki_now: incoming wavenumber in SI unit
        :return: wavevector transfers, energy transfers from Mushroom, energy transfers from magnon
        """
        nonlocal geo_ctx
        print("Step: ki={:.3f} AA-1, rotation step {:d}".format(ki_now * 1e-10, 0))
        q_vectors_per_ki = mushroom_wavevector_transfer(geo_ctx=geo_ctx, ki=ki_now)
        q_vectors_now = q_vectors_per_ki
        hw_mush_per_ki = mushroom_e_transfer(geo_ctx=geo_ctx, ki=ki_now)

        if fmt.rotation_steps > 0:
            for r in range(1, fmt.rotation_steps + 1):
                print("Step: ki={:.3f} AA-1, rotation step {:d}".format(ki_now * 1e-10, r))
                q_vectors_now = np.apply_along_axis(geo.rotation_3d, axis=0, arr=q_vectors_now,
                                                    rot_axis=fmt.sample_rot_axis, angle=-fmt.rotation_stepsize)
                q_vectors_per_ki = np.append(q_vectors_per_ki, q_vectors_now, axis=1)
                hw_mush_per_ki = np.append(hw_mush_per_ki, mushroom_e_transfer(geo_ctx=geo_ctx, ki=ki_now))
        # hw_mag_per_ki = energy_transfer(geo_ctx=geo_ctx, q_vector=q_vectors_per_ki, calc_term=TERM_MAGNON)
        # return q_vectors_per_ki, hw_mush_per_ki, hw_mag_per_ki
        return q_vectors_per_ki, hw_mush_per_ki

    for k, ki in enumerate(fmt.wavenumbers_in):
        # q_vetors, hw_mush, hw_mag = calc_per_ki(ki_now=ki)
        q_vetors, hw_mush = calc_per_ki(ki_now=ki)
        file = open(fmt.filenames_ki[k], "w+")
        file.write("# Q_x, Q_y, Q_z, hw_Mushroom, hw_Magnon\n")
        for i in range(q_vetors.shape[1]):
            file.write("{:e}, {:e}, {:e}, {:e}\n".format(*q_vetors[:, i], hw_mush[i]))
            # file.write("{:e}, {:e}, {:e}, {:e}, {:e}\n".format(*q_vetors[:, i], hw_mush[i], hw_mag[i]))
        file.close()
        print("File written in: {:s}".format(fmt.filenames_ki[k]))


geometryctx = MushroomContext()

dispersion_mushroom(geo_ctx=geometryctx)
