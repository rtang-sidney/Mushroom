import numpy as np

ZERO_TOL = 1e-6

moasic_pg002 = np.deg2rad(0.4)  # radian, PG crystal mosaic
moasic_analyser = np.deg2rad(0.4)  # radian, analyser mosaic
deltad_d = 6e-4  # relative uncertainty of the lattice distance, given in [paper2]
lattice_distance_pg002 = 3.35e-10  # m, lattice distance d of a PG crystal
an_seg = 1e-2  # m, the size of an analyser segment in 1D
distance_ms = 1.0  # m, monochromator-sample distance
divergence_initial = np.deg2rad(1.6)  # initial divergence directly from the neutron guide
sample_diameter = 1e-2  # m
sample_height = 1e-2  # m
detector_resolution = 1e-2  # m, the positional resolution of the position-sensitive detectors
