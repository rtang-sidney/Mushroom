import matplotlib.pyplot as plt
import numpy as np
from mushroom_context import MushroomContext
import geometry_calculation as geo
import neutron_context as nctx
import global_context as glb
import instrument_context as instr

plt.rcParams.update({'font.size': 18})


def mushroom_wavevector_transfer(geo_ctx: MushroomContext, ki, order):
    ki_vector = (ki, 0, 0)
    azi_angles, polar_angles = np.meshgrid(instr.azi_rad, geo_ctx.pol_angles)
    azi_angles = azi_angles.flatten()
    polar_angles = polar_angles.flatten()
    kf_vectors = np.empty((3, polar_angles.shape[0]))
    kf_vectors[0, :] = geo_ctx.wavenumber_f * np.cos(polar_angles) * np.cos(azi_angles)
    kf_vectors[1, :] = geo_ctx.wavenumber_f * np.cos(polar_angles) * np.sin(azi_angles)
    kf_vectors[2, :] = geo_ctx.wavenumber_f * np.sin(polar_angles)
    kf_vectors *= order  # diffraction order
    mushroom_q_vectors = np.apply_along_axis(nctx.wavevector_transfer, axis=0, arr=kf_vectors, wavevector_in=ki_vector)
    return mushroom_q_vectors


# def mushroom_e_transfer(geo_ctx: MushroomContext, ki):
#     #  values of all the qx, qy, qz in each sample rotation
#     # mushroom_hw = np.array(list(map(lambda i: np.array(list(
#     #     map(lambda j: neutron.wavenumber2energy(ki) - neutron.wavenumber2energy(geo_ctx.wavenumbers_out[i]),
#     #         range(geo_ctx.azi_angles.shape[0])))), range(geo_ctx.pol_angles.shape[0]))))
#     mushroom_hw = nctx.wavenumber2energy(ki) - nctx.wavenumber2energy(geo_ctx.wavenumber_f)
#     mushroom_hw = np.repeat(mushroom_hw, instr.azi_rad.shape[0])
#     return mushroom_hw


def dispersion_mushroom(geo_ctx: MushroomContext):
    def ki2qvector(ki_now, order):
        """
        Do the calculations for each value of the incoming wavenumber ki
        It gives only the values from Mushroom, independent of the sample
        :param ki_now: incoming wavenumber in SI unit
        :return: wavevector transfers, energy transfers from Mushroom, energy transfers from magnon
        """
        nonlocal geo_ctx
        print("Step: ki={:.3f} AA-1, order={:d}, rotation step {:d}".format(ki_now * 1e-10, order, 0))
        q_vectors_per_ki = mushroom_wavevector_transfer(geo_ctx=geo_ctx, ki=ki_now, order=order)
        q_vectors_now = q_vectors_per_ki

        if glb.rotation_steps > 0:
            for r in range(1, glb.rotation_steps + 1):
                print("Step: ki={:.3f} AA-1, order={:d}, rotation step {:d}".format(ki_now * 1e-10, order, r))
                q_vectors_now = np.apply_along_axis(geo.rotation_3d, axis=0, arr=q_vectors_now,
                                                    rot_axis=glb.sample_rot_axis, angle=-glb.rotation_stepsize)
                q_vectors_per_ki = np.append(q_vectors_per_ki, q_vectors_now, axis=1)
        # hw_mag_per_ki = energy_transfer(geo_ctx=geo_ctx, q_vector=q_vectors_per_ki, calc_term=TERM_MAGNON)
        # return q_vectors_per_ki, hw_mush_per_ki, hw_mag_per_ki
        return q_vectors_per_ki

    for ki in instr.wavenumbers_in:
        for order in glb.pg_orders:
            q_vectors = ki2qvector(ki_now=ki, order=order)
            hw_mush = nctx.wavenumber2energy(ki) - nctx.wavenumber2energy(geo_ctx.wavenumber_f * order)
            glb.write_q_hw(prefix=glb.prefix_mush, ki=ki, q_vectors=q_vectors, mush_hw=hw_mush, order=order)


geometryctx = MushroomContext()

dispersion_mushroom(geo_ctx=geometryctx)
