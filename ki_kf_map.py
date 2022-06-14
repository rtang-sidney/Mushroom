import numpy as np
import matplotlib.pyplot as plt
import neutron_context as neutron
import instrument_context as instr

# Calculate the kf as a function of ki when the magnon model parameters are given

hw_max = neutron.mev2joule(-5.0)
ki = np.linspace(1.2, 1.6, num=100) * 1e10
twotheta = neutron.bragg_wavenumber2twotheta(wavenumber=ki, lattice_distance=instr.interplanar_pg002)
kf = neutron.energy2wavenumber(energy=neutron.wavenumber2energy(np.min(ki)) - hw_max)
print(kf * 1e-10)
# kf = 1.9e10
hw = neutron.wavenumber2energy(ki) - neutron.wavenumber2energy(kf)
plt.figure()
# plt.plot(ki * 1e-10, np.rad2deg(twotheta))
plt.plot(ki * 1e-10, neutron.joule2mev(hw))
plt.show()


# kf_mid = kf[np.argmin(abs(ki * 1e-10 - 1.5))]
# twotheta_an_mid = neutron.bragg_wavenumber2angle(kf_mid, instr.interplanar_pg002)
# print(np.rad2deg(twotheta_an_mid))

# angle_mid = 90
# angle_range = 30
# twotheta = np.deg2rad(np.linspace(angle_mid - angle_range / 2.0, angle_mid + angle_range / 2.0, num=100))
# ki = neutron.bragg_angle2wavenumber(twotheta, instr.interplanar_pg002)
#
# plt.figure()
# plt.plot(np.rad2deg(twotheta), ki * 1e-10)
# # plt.plot(ki * 1e-10, kf * 1e-10)
# plt.show()
