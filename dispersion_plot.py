import matplotlib.pyplot as plt
import numpy as np
from magnonmodel import MagnonModel
import geometry_calculation as geo
import neutron_context as neutron
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker, colors
import format_context as fmt

plt.rcParams.update({'font.size': 18})

HKL_VALUES = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 2, 1),
              (2, 2, 1)]


def hkl2name(hkl):
    if hkl not in HKL_VALUES:
        raise ValueError("Invalid hkl value given, {} does not exist.".format(hkl))
    return "{:d},{:d},{:d}".format(*hkl)


def dispersion_plot(plot_term, extension, q_unit):
    # def energy_transfer(qx_per_rot, qy_per_rot, qz_per_rot):
    def plot_q(q_value):
        nonlocal q_unit
        if q_unit == fmt.q_unit_real:
            return q_value * 1e-10
        else:
            return q_value / (2 * np.pi / magnonmdl.l_const)

    def rlu2q(rlu):
        return np.array(rlu) * (2 * np.pi / magnonmdl.l_const)

    def line_through_points(line_direction, points, point):
        resol = 0.025e10  # unit m^⁻1
        distances = np.apply_along_axis(geo.point2line_3d, axis=0, arr=points, line_direction=line_direction,
                                        point_on=point)
        print(np.min(distances) * 1e-10, np.max(distances) * 1e-10)
        other_points = distances < resol
        return other_points

    def do_plot():
        nonlocal fig, plot_term, dim_info, extension
        filename = "{:s}_ki[{:.1f},{:.1f}]_{:s}.{:s}".format(plot_term, fmt.wavenumber_in_min * 1e-10,
                                                             fmt.wavenumber_in_max * 1e-10, dim_info, extension)
        filename = "".join([fmt.path_performance, filename])
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print("Plot saved: {:s}".format(filename))

    def hkl_plot(hkl_element):
        hkl_element = int(round(hkl_element))
        if hkl_element == 0:
            return ""
        else:
            if hkl_element == 1:
                return r"$+\xi$".format(hkl_element)
            else:
                return r"$+ {:d}\xi$".format(hkl_element)

    def points2xi(point, points, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        for x, i in enumerate(hkl):
            x = int(round(x))
            if x != 0:
                return (points[i, :] - point[i]) / float(x)

    def xi2points(point, hkl, xi):
        if isinstance(xi, float):
            point = np.array(point)
            hkl = np.array(hkl)
            return xi * hkl + point
        else:
            raise ValueError("xi should be a float")

    def mark_energy(mark, point, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        return magnonmdl.magnon_energy(wavevector_transfer=np.add(point, mark * hkl))

    def mark2qvector(mark, point, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        return np.add(point, mark * hkl)

    def mark2correlation(mark_q, mark_hw):
        q_vector = mark2qvector(mark=mark_q, point=point_interest, hkl=hkl)
        correlation = magnonmdl.correlation_func(qq_vector=q_vector, hw=mark_hw)
        return correlation

    data = np.loadtxt(fname=fmt.filenames_ki[0], delimiter=",")
    collect_q_vetors, collect_hw_mush = data[:, :3], data[:, 3]
    collect_q_vetors = np.transpose(collect_q_vetors)
    for k in range(len(fmt.wavenumbers_in))[1:]:
        data = np.loadtxt(fname=fmt.filenames_ki[k], delimiter=",")
        q_vetors, hw_mush = data[:, :3], data[:, 3]
        q_vetors = np.transpose(q_vetors)
        collect_q_vetors = np.append(collect_q_vetors, q_vetors, axis=1)
        collect_hw_mush = np.append(collect_hw_mush, hw_mush)

    # calculate all the data for the maximal times of rotation
    for point_interest in fmt.points_interest:
        for hkl in HKL_VALUES:
            hkl_name = hkl2name(hkl)
            print("Step: hkl {:s}, point {:d}{:d}{:d}".format(hkl_name, *point_interest))
            # index_point, index_points = line_through_points(hkl_value, points_finite_hw)
            index_points = line_through_points(hkl, collect_q_vetors, point=rlu2q(point_interest))
            if np.count_nonzero(index_points) > 0:
                points_near_line = collect_q_vetors[:, index_points]
                mush_hw = collect_hw_mush[index_points]
                mush_xi = points2xi(point=rlu2q(point_interest), points=points_near_line, hkl=hkl)
                max_distance = np.max(
                    np.apply_along_axis(func1d=geo.points_distance, axis=0, arr=points_near_line,
                                        point2=rlu2q(point_interest)))
                points_marks = np.linspace(-max_distance, max_distance, num=100)

                xlabel = r"({:d}{:s}, {:d}{:s}, {:d}{:s}) ({:s})".format(point_interest[0], hkl_plot(hkl[0]),
                                                                         point_interest[1], hkl_plot(hkl[1]),
                                                                         point_interest[2], hkl_plot(hkl[2]),
                                                                         q_unit)

                if plot_term == fmt.plot_correlation:
                    energy_marks = np.linspace(-np.max(abs(mush_hw)), np.max(abs(mush_hw)), num=100)
                    mark_grids, energy_grids = np.meshgrid(points_marks, energy_marks)
                    correlation_grids = np.array(list(map(lambda mi: np.array(list(
                        map(lambda mj: mark2correlation(mark_q=mark_grids[mi, mj], mark_hw=energy_grids[mi, mj]),
                            range(mark_grids.shape[1])))), range(mark_grids.shape[0]))))
                    clevels = np.round(np.geomspace(1, np.max(correlation_grids), num=10))

                    correlation_mush = np.array(list(
                        map(lambda m: magnonmdl.correlation_func(qq_vector=points_near_line[:, m], hw=mush_hw[m]),
                            range(mush_xi.shape[0]))))
                    correlation_mush = np.where(correlation_mush > 1, correlation_mush, 1)

                    # 2 subplots: ax1 theoretical magnon model, ax2 Mushroom
                    fig, [ax1, ax2] = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(20, 10))
                    cnt1 = ax1.contourf(plot_q(mark_grids), neutron.joule2mev(energy_grids), correlation_grids,
                                        locator=ticker.LogLocator(), levels=clevels)
                    cnt2 = ax2.scatter(plot_q(mush_xi), neutron.joule2mev(mush_hw), c=correlation_mush,
                                       norm=colors.LogNorm(vmin=np.min(clevels), vmax=np.max(clevels)))
                    fig.subplots_adjust(right=0.75)
                    cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
                    cbar = fig.colorbar(cnt2, cax=cbar_ax)
                    cbar.set_label("Correlation function")
                    cbar.set_label(r"$S(q,\omega)$")
                    # cbar2.set_label(r"$S(q,\omega)$")
                    ax1.tick_params(axis="both", direction="in")
                    ax2.tick_params(axis="both", direction="in")
                    ax1.set_xlabel(xlabel)
                    ax2.set_xlabel(xlabel)
                    ax1.set_ylabel(fmt.hw_label)
                else:
                    raise ValueError("Cannot plot this: {:s}".format(plot_term))
                dim_info = "[{:s}]({:d},{:d},{:d})".format(hkl, *point_interest)
                do_plot()
            else:
                print("Direction {:s} does not have any points".format(hkl))
                continue


magnonmdl = MagnonModel(model_name=fmt.magnon_default, latt_const=4.5 * 1e-10, spin_coupling=neutron.mev2joule(0.3))

dispersion_plot(plot_term=fmt.plot_correlation, extension=fmt.extension_png, q_unit=fmt.q_unit_rlu)
