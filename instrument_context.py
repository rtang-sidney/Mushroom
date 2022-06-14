import numpy as np
import neutron_context as nctx

# This code gives all parameters about the Mushroom spectrometer with a monochromator

sa_point = (0.0, 0.0)
cryo_radius = 0.25  # m, reserving the space for a cryostat
moasic_pg002 = np.deg2rad(0.4)  # radian, PG crystal mosaic
moasic_an = np.deg2rad(0.4)  # radian, analyser mosaic
deltad_d = 6e-4  # relative uncertainty of the lattice distance, given in [paper2]
interplanar_pg002 = 3.355e-10  # m, lattice distance d of a PG crystal
an_seg = 1e-2  # m, the size of an analyser segment in 1D
distance_ms = 1.75  # m, monochromator-sample distance
diver_nl = np.deg2rad(1.6)  # initial divergence directly from the neutron guide
sam_dia = 0.5e-2  # m
sam_height = 0.5e-2  # m
detector_resolution = 1e-2  # m, the positional resolution of the position-sensitive detectors

pol_plus_deg = 60  # degree, largest polar angle of the analyser
# pol_plus_deg = 15  # degree, largest polar angle of the analyser
pol_range_deg = 65  # the range of the solar angles covered by the analyser
pol_minus_deg = pol_plus_deg - pol_range_deg  # degree, smallest (largest negative) polar angle
pol_plus_rad = np.deg2rad(pol_plus_deg)
pol_minus_rad = np.deg2rad(pol_minus_deg)

azi_min_deg = 10
azi_max_deg = 170
azi_num = azi_max_deg - azi_min_deg + 1
azi_deg = np.append(np.linspace(-azi_max_deg, -azi_min_deg, num=azi_num),
                    np.linspace(azi_min_deg, azi_max_deg, num=azi_num))
azi_min_rad = np.deg2rad(azi_min_deg)
azi_max_rad = np.deg2rad(azi_max_deg)
azi_rad = np.deg2rad(azi_deg)

sa_rot_steps = 90
sa_rot_stepsize = np.deg2rad(1.0)

radius_sec_spec = 1.5  # m, radial space for the secondary spectrometer
dist_monos = 2.0  # m, largest distance between the two monochromators
twotheta_mono = np.arcsin(radius_sec_spec / dist_monos)
wavenumber_in_max = nctx.bragg_twotheta2wavenumber(twotheta=twotheta_mono, lattice_distance=interplanar_pg002)
wavenumber_in_min = nctx.bragg_twotheta2wavenumber(twotheta=np.pi - twotheta_mono, lattice_distance=interplanar_pg002)
ki_number = 20
# wavenumber_in_max = wavenumber_in_min + 0.025e10 * ki_number
wavenumbers_in = np.linspace(wavenumber_in_min, wavenumber_in_max, num=ki_number)
# print(wavenumbers_in * 1e-10)
# print(np.rad2deg(twotheta_mono), wavenumber_in_min * 1e-10, wavenumber_in_max * 1e-10)
