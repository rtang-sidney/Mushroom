import numpy as np
from helper import points_distance, InstrumentContext, angle_vectors, points_to_line, lines_intersect
from geometry_context import GeometryContext
import matplotlib.pyplot as plt

KYRO_RADIUS = 0.25  # m, radius of the gap reserved for the cyrostat
angle_plus_deg = 30  # degree, largest polar angle of the analyser
angle_minus_deg = angle_plus_deg - 60  # degree, smallest (largest negative) polar angle
ANGLE_PLUS_RAD = np.deg2rad(angle_plus_deg)
ANGLE_MINUS_RAD = np.deg2rad(angle_minus_deg)

AF_DISTANCE_VERT = 0.1  # m, vertical distance between the focus and the lowest segment of the analyser
KF_RATE = 0.5  # kf_min/kf_max, should be 0.5 if one wants to cover a kf range without a gap in between
DETECTOR_OUTER_RADIUS = 3  # m, the largest horizontal point of detector on one side
DETECTOR_HEIGHT = 3  # m, the largest vertical distance between sample and detector

POINT_SAMPLE = np.array([0, 0])
POINT_START = KYRO_RADIUS * np.array([1, np.tan(ANGLE_PLUS_RAD)])
distance_sf = np.linspace(KYRO_RADIUS, 3, num=100)
angle_focus = np.deg2rad(np.linspace(angle_minus_deg - 1, -90, num=100))

plt.rcParams.update({'font.size': 12})

PLOT_FORMAT = "pdf"

def endpoint_kf(sf, angle):
    point_focus = sf * np.array([np.cos(angle), np.sin(angle)])
    point_end = get_endpoint(sf=sf, angle=angle)
    # print(point_end, point_focus)
    vector_sa = point_end
    vector_af = point_focus - point_end
    analyser_twotheta = angle_vectors(vector1=vector_sa, vector2=vector_af)
    lattice_distance = 3.55 * 1e-10
    wavenumber = np.pi / (lattice_distance * np.sin(analyser_twotheta / 2.0))
    return wavenumber


def get_endpoint(sf, angle):
    point_focus = get_focus_point(sf, angle)
    distance_sa = points_distance(point1=POINT_SAMPLE, point2=POINT_START)
    distance_af = points_distance(point1=POINT_START, point2=point_focus)
    ellipse_semimajor = (distance_sa + distance_af) / 2.0

    x0, y0 = point_focus
    a = ellipse_semimajor
    x = (4 * a ** 2 - x0 ** 2 - y0 ** 2) / (4 * a / np.cos(ANGLE_MINUS_RAD) - 2 * (x0 + y0 * np.tan(ANGLE_MINUS_RAD)))
    point_end = x * np.array([1, np.tan(ANGLE_MINUS_RAD)])
    return point_end


def focus2kf(sf, angle):
    point_focus = get_focus_point(sf, angle)
    distance_sa = points_distance(point1=POINT_SAMPLE, point2=POINT_START)
    distance_af = points_distance(point1=POINT_START, point2=point_focus)
    ellipse_semimajor = (distance_sa + distance_af) / 2.0
    ellipse_lin_eccen = points_distance(point1=POINT_SAMPLE, point2=point_focus) / 2.0
    angle_triangle = np.arccos(
        (2 * ellipse_semimajor ** 2 - (2 * ellipse_lin_eccen) ** 2) / (2 * ellipse_semimajor ** 2))
    analyser_twotheta = np.pi - angle_triangle
    lattice_distance = 3.55 * 1e-10
    wavenumber = np.pi / (lattice_distance * np.sin(analyser_twotheta / 2.0))
    return wavenumber


def get_focus_point(sf, angle):
    point_focus = sf * np.array([np.cos(angle), np.sin(angle)])
    return point_focus


def farest_detector(sf, angle):
    point_focus = get_focus_point(sf, angle)
    detector_line = [0, 1, -2 * point_focus[-1]]
    line_af = points_to_line(point1=POINT_START, point2=point_focus)
    point_detector = lines_intersect(line1=line_af, line2=detector_line)
    return point_detector


sf_2d, angle_2d = np.meshgrid(distance_sf, angle_focus)
kfmax_2d = np.empty_like(sf_2d)
kfmin_2d = np.empty_like(sf_2d)
kf_min_max_2d = np.empty_like(sf_2d)
detector_hori_size = np.empty_like(sf_2d)
lowest_analyser = np.empty_like(sf_2d)
focus_vertical = np.empty_like(sf_2d)
for i in range(sf_2d.shape[0]):
    for j in range(sf_2d.shape[1]):
        kfmax_2d[i, j] = focus2kf(sf=sf_2d[i, j], angle=angle_2d[i, j])
        kfmin_2d[i, j] = endpoint_kf(sf=sf_2d[i, j], angle=angle_2d[i, j])
        kf_min_max_2d[i, j] = kfmin_2d[i, j] / kfmax_2d[i, j]
        detector_hori_size[i, j], detector_height = farest_detector(sf=sf_2d[i, j], angle=angle_2d[i, j])
        lowest_analyser[i, j] = get_endpoint(sf=sf_2d[i, j], angle=angle_2d[i, j])[-1]
        focus_vertical[i, j] = get_focus_point(sf=sf_2d[i, j], angle=angle_2d[i, j])[-1]
        if lowest_analyser[i, j] - focus_vertical[i, j] > AF_DISTANCE_VERT and kf_min_max_2d[i, j] < KF_RATE and \
                detector_hori_size[i, j] < DETECTOR_OUTER_RADIUS and abs(detector_height) < DETECTOR_HEIGHT:
            print(
                "angle_plus = {:.2f} degree, distance_sf = {:.2f} m, focus_angle = {:.2f} degree, detector_height = {:.2f} m, kfmax = {:.2f} AA-1, kfmin = {:.2f} AA-1".format(
                    angle_plus_deg, sf_2d[i, j], np.rad2deg(angle_2d[i, j]), detector_height, kfmax_2d[i, j] * 1e-10,
                                                                                              kfmin_2d[i, j] * 1e-10))
            
fig, axs = plt.subplots(1, 2, sharey="all")
cnt = axs[0].contourf(sf_2d, np.rad2deg(angle_2d), kfmax_2d * 1e-10)
cbar = fig.colorbar(cnt, ax=axs[0])
axs[0].set_xlabel("Distance sample-focus (m)")
axs[0].set_ylabel("Polar angle of the focus (degree)")
axs[0].tick_params(axis="x", direction="in")
axs[0].tick_params(axis="y", direction="in")
cbar.set_label(r"$k_{f,max}$ ($\AA^{-1}$)")
# axs[0].set_xlabel("Distance between sample and the other focus (m)")
# axs[0].set_ylabel("Polar angle of the other focus (degree)")
# axs[0].set_title(r"Maximum of $k_f$ covered by the analyser ($\AA^{-1}$)")
# axs[0].plot(1.8, -45, ".", color="black")
# axs[0].text(1.25, -47, r"$k_{f,max}=$" + "{:.2f}".format(1e-10 * focus2kf(1.8, np.deg2rad(-45))) + r"$\AA$")

cnt = axs[1].contourf(sf_2d, np.rad2deg(angle_2d), kfmin_2d * 1e-10)
cbar = fig.colorbar(cnt, ax=axs[1])
axs[1].set_xlabel("Distance sample-focus (m)")
axs[1].tick_params(axis="x", direction="in")
axs[1].tick_params(axis="y", direction="in")
# axs[1].set_ylabel("Polar angle of the focus (degree)")
cbar.set_label(r"$k_{f,min}$ ($\AA^{-1}$)")
# axs[1].set_xlabel("Distance between sample and the other focus (m)")
# axs[1].set_ylabel("Polar angle of the other focus (degree)")
# axs[1].set_title(r"Minimum of $k_f$ covered by the analyser ($\AA^{-1}$)")
# axs[1].plot(1.8, -45, ".", color="black")
# axs[1].text(1.25, -47, r"$k_{f,min}=$" + "{:.2f}".format(1e-10 * endpoint_kf(1.8, np.deg2rad(-45))) + r"$\AA$")
plt.tight_layout()
plt.savefig("kfmax_geometry.{:s}".format(PLOT_FORMAT))
plt.close(fig)

fig, axs = plt.subplots(1, 2, sharey="all")
cnt = axs[0].contourf(sf_2d, np.rad2deg(angle_2d), kf_min_max_2d)
cbar = fig.colorbar(cnt, ax=axs[0])
axs[0].set_xlabel("Distance sample-focus (m)")
axs[0].set_ylabel("Polar angle of the focus (degree)")
cbar.set_label(r"$\frac{k_{f,min}}{k_{f,max}}$")

cnt = axs[1].contourf(sf_2d, np.rad2deg(angle_2d), lowest_analyser - focus_vertical)
cbar = fig.colorbar(cnt, ax=axs[1])
axs[1].set_xlabel("Distance sample-focus (m)")
# axs[1].set_ylabel("Polar angle of the focus (degree)")
cbar.set_label(r"$H_{analyser}-y_{focus}$")
# axs[1].set_xlabel("Distance between sample and the other focus (m)")
# axs[1].set_ylabel("Polar angle of the other focus (degree)")
# axs[1].set_title(r"Minimum of $k_f$ covered by the analyser ($\AA^{-1}$)")
# axs[1].plot(1.5, -45, ".", color="black")
# axs[1].text(1.25, -47, r"$k_{f,min}=$" + "{:.2f}".format(1e-10 * endpoint_kf(1.5, np.deg2rad(-45))) + r"$\AA$")
plt.tight_layout()
plt.savefig("kf_min_max.{:s}".format(PLOT_FORMAT))
plt.close(fig)

fig, ax = plt.subplots()
cnt = ax.contourf(sf_2d, np.rad2deg(angle_2d), detector_hori_size)
cbar = fig.colorbar(cnt, ax=ax)
ax.set_xlabel("Distance sample-focus (m)")
ax.set_ylabel("Polar angle of the focus (degree)")
cbar.set_label("Horizontal size of the detector (m)")
# axs[1].set_xlabel("Distance between sample and the other focus (m)")
# axs[1].set_ylabel("Polar angle of the other focus (degree)")
# axs[1].set_title(r"Minimum of $k_f$ covered by the analyser ($\AA^{-1}$)")
# axs[1].plot(1.5, -45, ".", color="black")
# axs[1].text(1.25, -47, r"$k_{f,min}=$" + "{:.2f}".format(1e-10 * endpoint_kf(1.5, np.deg2rad(-45))) + r"$\AA$")
plt.tight_layout()
plt.savefig("detector_size.{:s}".format(PLOT_FORMAT))
plt.close(fig)
