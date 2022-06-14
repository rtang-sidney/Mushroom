import numpy as np

extension_pdf = "pdf"
extension_png = "png"
extension_dat = "dat"

path_performance = "Performance/"
path_resolution = "Resolution/"
path_geometry = "Geometry/"

q_unit_real = r"$\AA^{-1}$"  # the real unit of Q-vector
q_unit_rlu = "r.l.u."  # reciprocal lattice unit
hw_label = r"$\hbar\omega$ (meV)"

axis_x = "x"
axis_y = "y"
axis_z = "z"
axes = [axis_x, axis_y, axis_z]

colour_x = "blue"
colour_y = "red"
colour_z = "darkgoldenrod"
rotation_stepsize = np.deg2rad(1)  # sample rotation step size
rotation_steps = 90
magnon_default = "Default"
sample_rot_axis = (0, 0, 1)
points_interest = [(0, 1, 0), (0, 0, 1)]  # (0, 0, 0),(1, 0, 0) ,

# The range of the incoming wavenumbers is the same as at MIRA II
wavenumber_in_min = 1.2e10
wavenumber_in_max = 1.6e10
wavenumbers_in = np.linspace(wavenumber_in_min, wavenumber_in_max, num=5)

filenames_ki = []
for ki in wavenumbers_in:
    name = "Mushroom_ki{:.1f}_Rot({:d}{:d}{:d}){:d}".format(ki * 1e-10, *sample_rot_axis, rotation_steps)
    name = ".".join([name, extension_dat])
    filenames_ki.append(name)
