import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths

plt.rcParams.update({'font.size': 18})

# plots the wavelength distribution of the two instrument configurations

FOLDER1 = "test_20210408_165802"
FOLDER2 = "test_20210408_171342"
FILENAME = "lambda.dat"
CASE1 = "Mushroom"
CASE2 = "TAS"
CASES = [CASE1, CASE2]
FOLDERS = [FOLDER1, FOLDER2]


def get_wavelength_signals(fname):
    signals = np.loadtxt(fname=fname, comments="#")
    wavelengths = signals[:, 0] * 1e-10
    intentisities = signals[:, 1]
    return wavelengths, intentisities


fig, ax = plt.subplots()
for i, f in enumerate(FOLDERS):
    fname = "/".join([f, FILENAME])
    wavel, inten = get_wavelength_signals(fname=fname)
    peaks, _ = find_peaks(inten)
    results_half = peak_widths(inten, peaks, rel_height=0.5)[0]
    print(peaks, results_half)
    ax.plot(wavel * 1e10, inten, label=CASES[i])
ax.set_xlabel(r"$\lambda$ ($\AA$)")
ax.set_ylabel("Intensity")
ax.set_xlim(3, 4)
ax.legend()
ax.tick_params(axis="both", direction="in")
plot_file = "CaseComparison"
fig.savefig(plot_file, bbox_inches='tight')
plt.close(fig)
print("Plot saved: {:s}".format(plot_file))
