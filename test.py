import numpy as np
import geometry_calculation as geo
import matplotlib.pyplot as plt

marks = np.linspace(-3, 3, num=100)
a = 4.5e-10
reci_const = 2 * np.pi / a
wavevector_transfer=np.array([0.5 + 2*marks,0.5 + marks,0.5 + marks])* 1e10
hkl = np.round(wavevector_transfer / reci_const)
# print(scattering_vector, reci_const, hkl)
magnon_vector = wavevector_transfer - hkl * reci_const
energy = 3-np.cos(magnon_vector[0] * a) - np.cos(magnon_vector[1] * a) - np.cos(magnon_vector[2] * a)

plt.figure()
plt.plot(marks, energy)
plt.show()
