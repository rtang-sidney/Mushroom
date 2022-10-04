import numpy as np

import chopper_context as cctx
import neutron_context as nctx

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})


# Given a reference resolution of ToF and the respective wavelength or wavenumber, the code checks if this is achievable
# and if so, gives back the corresponding frequencies (both in hz and rpm possible) of the choppers

class ChopperInfo:
    def __init__(self, ref_resol_tof, open1, open2, angle1_deg, angle2_deg, ref_wavelength=None, ref_wavenumber=None):
        """
        :param ref_resol_tof: the relative uncertainty of ToF, which is half of that for the energy
        :param ref_wavelength: wavelength where the time resolution is defined
        :param ref_wavenumber: wavenumber where the time resolution is defined
        """
        if ref_wavelength:
            if 1e-10 <= ref_wavelength <= 10e-10:
                self.velocity = nctx.wavelength2velocity(ref_wavelength)
            else:
                raise ValueError("Wavelength outside the limit between 1 and 10 Ansgtrom.")
        elif ref_wavenumber:
            if 1e-10 <= nctx.wavenumber2wavelength(ref_wavenumber) <= 10e-10:
                self.velocity = nctx.wavenumber2velocity(ref_wavenumber)
            else:
                raise ValueError("Wavelength outside the limit between 1 and 10 Ansgtrom.")
        else:
            raise ValueError("Cannot calculate the reference velocity since neither wavelength or wavenumber is given.")

        self.angle1_rad = np.deg2rad(angle1_deg)
        self.angle2_rad = np.deg2rad(angle2_deg)
        resol_limit = self._uncertainty_limit_mechanical()
        if ref_resol_tof < resol_limit:
            raise RuntimeError(
                "The target resolution {:.2f} % at {:.21f} AA greater than {:.2f} % from mechanical limit.".format(
                    ref_resol_tof * 1e2, nctx.velocity2wavelength(self.velocity) * 1e10, resol_limit * 1e2))
        else:
            self.time_resol = ref_resol_tof
        self.open1 = open1
        self.open2 = open2

        self.chopper1_hz = self.angle1_rad * self.velocity / (2.0 * np.pi * cctx.distance1s * ref_resol_tof)

        self.chopper1_rpm = np.ceil(cctx.hz2rpm(self.chopper1_hz))
        self.repetition_t1 = cctx.rpm2period(self.chopper1_rpm) / float(open1)
        self.open_t1 = self.angle1_rad / (2.0 * np.pi * self.chopper1_hz)
        self.open_t2 = (self.repetition_t1 - self.open_t1) * (1 - cctx.distance12 / cctx.distance1s)
        chopper2_hz = self.angle2_rad / (2 * np.pi * self.open_t2)
        self.chopper2_rpm = np.floor(cctx.hz2rpm(chopper2_hz))
        self.repetition_t2 = (cctx.distance1s - cctx.distance12) * (
                self.repetition_t1 / cctx.distance12 + cctx.tau_max - cctx.tau_min)
        chopper2_hz2 = 1.0 / (open2 * self.repetition_t2)
        self.chopper2_rpm2 = np.ceil(cctx.hz2rpm(chopper2_hz2))

    def _uncertainty_limit_mechanical(self):
        return self.angle1_rad * self.velocity / (2 * np.pi * cctx.rpm2hz(cctx.mechanical_limit_rpm) * cctx.distance1s)


MODE_WAVELENGTH = "Wavelength"
MODE_RESOLUTION = "Resolution"
MODES = [MODE_WAVELENGTH, MODE_RESOLUTION]


def parameter_transimission(n2_1d_len, mode, open1, open2, angle1_deg, angle2_deg):
    def calculation_step(resolution, wavelength):
        chopper12 = ChopperInfo(ref_resol_tof=resolution, open1=open1, open2=open2, angle1_deg=angle1_deg,
                                angle2_deg=angle2_deg, ref_wavelength=wavelength)
        chopper1_rpms.append(chopper12.chopper1_rpm)
        chopper2_rpms.append(chopper12.chopper2_rpm2)
        repetition_t1s.append(chopper12.repetition_t1)
        repetition_t2s.append(chopper12.repetition_t2)
        w1s.append(chopper12.open_t1 / 2.0)
        w2s.append(chopper12.open_t2 / 2.0)
        n1_pass, n2_pass, if_overlap = cctx.remove_overlap(chopper12.repetition_t1, chopper12.repetition_t2,
                                                           chopper12.open_t1 / 2.0, chopper12.open_t2 / 2.0,
                                                           n2_1d_len=n2_1d_len, parallelogram=False)
        print(n1_pass.shape)
        rate = cctx.transmission_rate(n1_pass.shape[0], n2_1d_len, chopper12.open_t1, chopper12.open_t2,
                                      chopper12.repetition_t2)
        rates.append(rate)

        rate_limit = chopper12.open_t1 * chopper12.open_t2 / (2.0 * chopper12.repetition_t1 * chopper12.repetition_t2)
        rates_limit.append(rate_limit)

    chopper1_rpms = []
    chopper2_rpms = []
    repetition_t1s = []
    repetition_t2s = []
    w1s = []
    w2s = []
    rates = []
    rates_limit = []
    label_c1 = "Chopper 1"  # , {:d} x {:d}°".format(open1, angle1_deg)
    label_c2 = "Chopper 2"  # , {:d} x {:d}°".format(open2, angle2_deg)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"

    if mode == MODE_WAVELENGTH:
        resolution = np.ceil(
            cctx.limit_mechanical(nctx.wavelength2wavenumber(cctx.wavelength_min), np.deg2rad(angle1_deg),
                                  cctx.rpm2hz(cctx.mechanical_limit_rpm), cctx.distance1s) * 1e2) * 1e-2
        wavelengths = np.linspace(cctx.wavelength_min, cctx.wavelength_max, 100)
        for wavelength in wavelengths:
            calculation_step(resolution, wavelength)
        ax.plot(wavelengths * 1e10, chopper1_rpms, label=label_c1)
        ax.plot(wavelengths * 1e10, chopper2_rpms, label=label_c2)
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        ax.set_title("Time resolution {:.0f} %".format(resolution * 1e2))
        ax2.plot(wavelengths * 1e10, np.array(rates) * 1e2, ".", color=colour_ax2)
        ax2.plot(wavelengths * 1e10, np.array(rates_limit) * 1e2, color=colour_ax2)
    elif mode == MODE_RESOLUTION:
        wavenumber = 1.1e10
        wavelength = nctx.wavenumber2wavelength(wavenumber)
        resol_limit = cctx.limit_mechanical(wavenumber, np.deg2rad(angle1_deg), cctx.rpm2hz(cctx.mechanical_limit_rpm),
                                            cctx.distance1s)
        resolutions = np.linspace(resol_limit, 10e-2, num=100)
        for resolution in resolutions:
            calculation_step(resolution, wavelength)
        ax.plot(resolutions * 1e2, chopper1_rpms, label=label_c1)
        ax.plot(resolutions * 1e2, chopper2_rpms, label=label_c2)
        ax.set_xlabel("Time resolution (%)")
        ax.set_title("Reference wavelength {:.1f}".format(wavelength * 1e10) + r" $\mathrm{\AA}$")
        ax2.plot(resolutions * 1e2, np.array(rates) * 1e2, ".", color=colour_ax2)
        ax2.plot(resolutions * 1e2, np.array(rates_limit) * 1e2, color=colour_ax2)

    # ax.legend()
    # ax2.legend()

    ax.set_ylabel("Chopper frequency (RPM)")
    ax.tick_params(axis="both", direction="in")
    # ax2.set_yscale("log")
    ax2.set_ylabel("Transmission rate (%)", color=colour_ax2)
    ax2.tick_params(axis='y', direction="in", color=colour_ax2, labelcolor=colour_ax2)
    fig.savefig("TOF//Chopper_Parameter_vs_{:s}.png".format(mode), bbox_inches='tight')
    plt.close(fig)


for mode in MODES:
    parameter_transimission(n2_1d_len=1000, mode=mode, open1=5, open2=1, angle1_deg=12, angle2_deg=9)
