import numpy as np
import instrument_context as instr

# This code gives the global coefficients and methods for other code files regarding the Mushroom project. It will
# not execute any commands and should be called by other code files.

extension_pdf = "pdf"
extension_png = "png"
extension_dat = "dat"

path_performance = "Performance/"
path_resolution = "Resolution/"
path_geometry = "Geometry/"
path_mcstas = "McStas/"

q_unit_real = r"$\AA^{-1}$"  # the real unit of Q-vector
q_unit_rlu = "r.l.u."  # reciprocal lattice unit
e_joule = "joule"
e_mev = "meV"
hw_label = r"$\hbar\omega$ ({:s})".format(e_mev)

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
l_interest = 1
animation_frames = 100

prefix_mush = "Mushroom"
prefix_mcstas = "McStas"

comment_symbol = "#"
term_hw = "hw"

pg_orders = [1, 2]


def fname_write_dispersion(prefix, ki, order, path):
    name = "{:s}{:.3f}_PG00{:d}".format(prefix, ki * 1e-10, int(2 * order))
    name = "".join([path, name])
    name = ".".join([name, extension_dat])
    return name


def fname_lineplot(hkl, point_interest, extension):
    dim_info = "[{:d}{:d}{:d}]({:d},{:d},{:d})".format(*hkl, *point_interest)
    filename = "Correlation_ki[{:.1f},{:.1f}]_{:s}.{:s}".format(instr.wavenumber_in_min * 1e-10,
                                                                instr.wavenumber_in_max * 1e-10, dim_info,
                                                                extension)
    return "".join([path_performance, filename])


def write_q_hw(prefix, ki, q_vectors, mush_hw, order):
    filename = fname_write_dispersion(prefix, ki, order, path=path_performance)
    file = open(filename, "w+")
    file.write("# {:s} (joule) {:e}\n".format(term_hw, mush_hw))
    file.write("# Q_x, Q_y, Q_z (m^{-1})\n")
    for i in range(q_vectors.shape[1]):
        file.write("{:e}, {:e}, {:e}\n".format(*q_vectors[:, i]))
        # file.write("{:e}, {:e}, {:e}, {:e}, {:e}\n".format(*q_vetors[:, i], hw_mush[i], hw_mag[i]))
    file.close()
    print("File written in: {:s}".format(filename))


def write_dispersion(prefix, ki, q_vectors, intensities, order):
    filename = fname_write_dispersion(prefix, ki, order, path=path_mcstas)
    file = open(filename, "w+")
    file.write("# Q_x, Q_y, Q_z (m^{-1}), Intensity\n")
    for i in range(q_vectors.shape[1]):
        file.write("{:e}, {:e}, {:e}, {:e}\n".format(*q_vectors[:, i], intensities[i]))
        # file.write("{:e}, {:e}, {:e}, {:e}, {:e}\n".format(*q_vetors[:, i], hw_mush[i], hw_mag[i]))
    file.close()
    print("File written in: {:s}".format(filename))


mcstas_ana_angles = np.linspace(start=5 + 7, stop=170 - 8, num=11).astype(int)
type_bcc = "bcc"
type_cp = "cP"
zero_tol = 1e-6