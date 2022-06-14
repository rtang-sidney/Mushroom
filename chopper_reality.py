import numpy as np

import chopper_context as cctx
import neutron_context as nctx


# Given a reference resolution of ToF and the respective wavelength or wavenumber, the code checks if this is achievable
# and if so, gives back the corresponding frequencies (both in hz and rpm possible) of the choppers

class ChopperInfo:
    def __init__(self, ref_resol_tof, open1=5, open2=1, angle1_deg=6, angle2_deg=10, ref_wavelength=None,
                 ref_wavenumber=None):
        """
        :param ref_resol_tof: the relative uncertainty of ToF, which is half of that for the energy
        :param ref_wavelength: wavelength where the time resolution is defined
        :param ref_wavenumber: wavenumber where the time resolution is defined
        """
        if ref_wavelength:
            self.velocity = nctx.wavelength2velocity(ref_wavelength)
        elif ref_wavenumber:
            self.velocity = nctx.wavenumber2velocity(ref_wavenumber)
        else:
            raise ValueError("Cannot calculate the reference velocity since neither wavelength or wavenumber is given.")

        self.angle1_rad = np.deg2rad(angle1_deg)
        self.angle2_rad = np.deg2rad(angle2_deg)
        resol_limit = self._uncertainty_limit_mechanical()
        if ref_resol_tof < resol_limit:
            raise RuntimeError(
                "The target resolution {:.1f} % greater than {:.1f} % from mechanical limit.".format(
                    ref_resol_tof * 1e2, resol_limit * 1e2))
        else:
            self.time_resol = ref_resol_tof
        self.open1 = open1
        self.open2 = open2

        self.chopper1_hz = self.angle1_rad * self.velocity / (2.0 * np.pi * cctx.distance1s * ref_resol_tof)

        self.chopper1_rpm = np.ceil(cctx.hz2rpm(self.chopper1_hz))
        repetition_t1 = cctx.rpm2period(self.chopper1_rpm) / float(open1)
        open_t1 = self.angle1_rad / (2.0 * np.pi * self.chopper1_hz)
        open_t2 = (repetition_t1 - open_t1) * (1 - cctx.distance12 / cctx.distance1s)
        chopper2_hz = self.angle2_rad / (2 * np.pi * open_t2)
        self.chopper2_rpm = np.floor(cctx.hz2rpm(chopper2_hz))
        repetition_t2 = (cctx.distance1s - cctx.distance12) * (
                repetition_t1 / cctx.distance12 + cctx.tau_max - cctx.tau_min)
        chopper2_hz2 = 1.0 / (open2 * repetition_t2)
        self.chopper2_rpm2 = np.ceil(cctx.hz2rpm(chopper2_hz2))

    def _uncertainty_limit_mechanical(self):
        return self.angle1_rad * self.velocity / (2 * np.pi * cctx.rpm2hz(cctx.mechanical_limit_rpm) * cctx.distance1s)


chopper12 = ChopperInfo(2.7e-2, ref_wavelength=1e-10)
print(chopper12.chopper1_rpm, chopper12.chopper2_rpm, chopper12.chopper2_rpm2)
f1_threshold = (2 * np.pi / np.deg2rad(10) / 1.0 / 5.0 / cctx.distance1s - 1 / 5.0 / cctx.distance12 - np.deg2rad(
    6) / np.deg2rad(10) / 1.0 / cctx.distance1s) / (cctx.tau_max - cctx.tau_min)
print(cctx.hz2rpm(f1_threshold))
