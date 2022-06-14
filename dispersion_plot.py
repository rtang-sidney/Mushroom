import matplotlib.pyplot as plt
import numpy as np
import re
import instrument_context as instr
from magnonmodel import MagnonModel
import geometry_calculation as geo
import neutron_context as nctx
from matplotlib import ticker, colors
import global_context as glb
import os

plt.rcParams.update({'font.size': 18})

# Plots the dispersion relation along one line in Q-space
# Probably not in use at the moment

CORRELATION_FUNCTION = "CorrelationFunction"

HKL_VALUES = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 2, 1),
              (2, 2, 1)]  # (0, 0, 1),
POINT_INTEREST = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

PATTERN_HW = re.compile(r"hw \s*\((\S*)\)\s*([-+]?[0-9]*.[0-9]*e[-+]?[0-9]*)")


def latest_dispersion_file():
    time_lists = []
    for ki in instr.wavenumbers_in:
        # for each ki value there is only one value of hw!
        for order in glb.pg_orders:
            fname = glb.fname_write_dispersion(prefix=glb.prefix_mush, ki=ki, order=order, path=glb.path_performance)

            time_lists.append(os.stat(fname).st_mtime)
    print(time_lists)
    time_lists = np.array(time_lists)
    return np.max(time_lists)


def hkl2name(hkl):
    if hkl not in HKL_VALUES:
        raise ValueError("Invalid hkl value given, {} does not exist.".format(hkl))
    return "{:d},{:d},{:d}".format(*hkl)


def plot_line(point_interest, hkl, extension):
    """
    to do a 1D plot along the line, which is defined by the direction and a point on it
    The x-axis is hkl (rlu), y is hw, and heatmap gives the correlation function
    One takes out the data according to the line of interest each time and repeat with the files
    :param point_interest: the point through which the line goes
    :param hkl: the direction defining the line direction
    :param extension: file extension
    :return: nothing
    """

    def line_through_points(line_direction, points, point, resol=0.01):
        # if not given, resolution is 0.025 AA^-1, which is about 3%
        distances = np.apply_along_axis(geo.point2line_3d, axis=0, arr=points, line_direction=line_direction,
                                        point_on=point)
        print(np.min(distances), np.max(distances))
        accepted_index = distances < resol
        return accepted_index

    def qrluxi(point, points, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        for i, x in enumerate(hkl):
            x = int(round(x))
            if x != 0:
                return (points[i, :] - point[i]) / float(x)

    def xi2qvector(mark, point, hkl):
        point = np.array(point)
        hkl = np.array(hkl)
        return np.add(point, mark * hkl)

    def file2hw(fname):
        f = open(file=fname).readlines()
        hw_value = None
        for line in f:
            if line.startswith(glb.comment_symbol):
                line = line[2:]
                if line.startswith(glb.term_hw):
                    hw_unit, hw_value = re.search(pattern=PATTERN_HW, string=line).groups()
                    hw_value = float(hw_value)
                    if hw_unit == glb.e_mev:
                        hw_value = nctx.mev2joule(hw_value)
        if hw_value:
            return hw_value
        else:
            raise RuntimeError("Failed to get the hw value from the file.")

    def data_collect_select(point_interest, hkl, ki, order):
        fname = glb.fname_write_dispersion(prefix=glb.prefix_mush, ki=ki, order=order, path=glb.path_performance)
        hw_value = file2hw(fname=fname)
        q_vectors = np.loadtxt(fname=fname, delimiter=",")
        q_vectors = np.transpose(q_vectors)
        qrlu = nctx.q2rlu(q_vectors, l_const=latt_const)
        select_index = line_through_points(line_direction=hkl, point=point_interest, points=qrlu)
        select_q = qrlu[:, select_index]
        return select_q, hw_value

    def hkl_plot(hkl_element):
        hkl_element = int(round(hkl_element))
        if hkl_element == 0:
            return ""
        else:
            if hkl_element == 1:
                return r"$+\xi$".format(hkl_element)
            else:
                return r"$+ {:d}\xi$".format(hkl_element)

    def xi2correlation(xi, hw, p_interest, hkl):
        q_vector = xi2qvector(mark=xi, point=p_interest, hkl=hkl)
        correlation = magnonmdl.corr_func_from_q_hw(q_rlu=q_vector, hw=hw)
        return correlation

    filename = glb.fname_lineplot(hkl=hkl, point_interest=point_interest, extension=extension)

    # If the magnon model is changed, all plots have to be done again.
    # Otherwise continue with the unfinished part only.
    if model_changed is False:
        if os.path.exists(filename):
            time_plot = os.stat(filename).st_mtime
            if time_plot > latest_modify:
                print("This plot exists already {}".format(filename))
                return

    # calculate all the data for the maximal times of rotation
    hkl_name = hkl2name(hkl)
    print("Step: hkl {:s}, point {:d}{:d}{:d}".format(hkl_name, *point_interest))
    xlabel = r"({:d}{:s}, {:d}{:s}, {:d}{:s}) ({:s})".format(point_interest[0], hkl_plot(hkl[0]),
                                                             point_interest[1], hkl_plot(hkl[1]),
                                                             point_interest[2], hkl_plot(hkl[2]),
                                                             "rlu")
    mush_qrlu = None
    mush_xi = None
    mush_hw = None
    for ki in instr.wavenumbers_in:
        # for each ki value there is only one value of hw!
        for order in glb.pg_orders:
            ki_qrlu, ki_hw = data_collect_select(point_interest=point_interest, hkl=hkl, ki=ki, order=order)
            ki_xi = qrluxi(point=point_interest, points=ki_qrlu, hkl=hkl)
            ki_hw = np.repeat(ki_hw, ki_qrlu.shape[1])
            if ki < instr.wavenumbers_in[1]:
                mush_qrlu = ki_qrlu
                mush_xi = ki_xi
                mush_hw = ki_hw
            else:
                mush_qrlu = np.append(mush_qrlu, ki_qrlu, axis=1)
                mush_xi = np.append(mush_xi, ki_xi)
                mush_hw = np.append(mush_hw, ki_hw)

    if mush_xi is None or mush_xi.shape[0] == 0:
        fig = plt.figure()
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return

    model_xi = np.linspace(np.min(mush_xi), np.max(mush_xi), num=100)
    model_hw = np.linspace(-np.max(abs(mush_hw)), np.max(abs(mush_hw)), num=100)
    model_xi_grid, model_hw_grid = np.meshgrid(model_xi, model_hw)
    model_corr = np.array(list(map(lambda mi: np.array(list(map(
        lambda mj: xi2correlation(xi=model_xi_grid[mi, mj], hw=model_hw_grid[mi, mj],
                                  p_interest=point_interest, hkl=hkl), range(model_xi_grid.shape[1])))),
                                   range(model_xi_grid.shape[0]))))
    clevels = np.round(np.geomspace(1, np.max(model_corr), num=10))

    mush_corr = np.array(list(
        map(lambda m: magnonmdl.corr_func_from_q_hw(q_rlu=mush_qrlu[:, m], hw=mush_hw[m]),
            range(mush_hw.shape[0]))))
    mush_corr = np.where(mush_corr > 1, mush_corr, 1)

    # 2 subplots: ax1 theoretical magnon model, ax2 Mushroom
    fig, [ax1, ax2] = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(20, 10))
    cnt1 = ax1.contourf(model_xi_grid, nctx.joule2mev(model_hw_grid), model_corr,
                        locator=ticker.LogLocator(), levels=clevels)
    cnt2 = ax2.scatter(mush_xi, nctx.joule2mev(mush_hw), c=mush_corr,
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
    ax1.set_ylabel(glb.hw_label)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


latt_const = 4.5 * 1e-10
magnonmdl = MagnonModel(model_name=glb.magnon_default, latt_const=latt_const, spin_coupling=nctx.mev2joule(0.3))
latest_modify = latest_dispersion_file()
model_changed = True

for point_interest in POINT_INTEREST:
    for hkl in HKL_VALUES:
        plot_line(hkl=hkl, point_interest=point_interest, extension=glb.extension_png)
