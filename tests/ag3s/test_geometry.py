"""Primitive-fitting tests.

Every fit is checked for **containment**, not for tightness. A primitive that is slightly too big
causes a detour; one that is slightly too small causes a collision, and only one of those is
recoverable. Tightness is checked too, but only as a relative comparison between types — a capsule
should beat a sphere on something elongated, or there is no reason for it to exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.stages.geometry import (
    PRIORITY,
    contains,
    fit_box,
    fit_capsule,
    fit_ellipsoid,
    fit_ellipsoid_mvee,
    fit_primitive,
    fit_sphere,
    rms_radius,
    to_spheres,
)
from AG3S.ag3s.stages.reconstruction import reconstruct
from AG3S.ag3s.types import PrimitiveType
from tests.ag3s.fixtures import LABEL_BOX, LABEL_CYLINDER, LABEL_SPHERE, make_scene

CFG = AG3SConfig()
ALL_FITTERS = (fit_sphere, fit_capsule, fit_box, fit_ellipsoid)


@pytest.fixture(scope="module")
def clusters():
    """Real points off the fixture, one array per object, as candidate generation would see them."""
    scene = make_scene()
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=CFG.pointcloud,
    )
    labels = scene.labels[cloud.uv[:, 1], cloud.uv[:, 0]]
    return {
        label: cloud.points[labels == label]
        for label in (LABEL_SPHERE, LABEL_BOX, LABEL_CYLINDER)
    }


def _elongated(n=400, seed=0):
    """A rod along a tilted axis — the case a capsule exists for."""
    rng = np.random.default_rng(seed)
    axis = np.array([1.0, 1.0, 0.5])
    axis /= np.linalg.norm(axis)
    t = rng.uniform(-0.25, 0.25, n)
    radial = rng.normal(0.0, 0.006, (n, 3))
    return np.array([0.6, 0.0, 0.9]) + t[:, None] * axis + radial


# -------------------------------------------------------------------------- containment


@pytest.mark.parametrize("fitter", ALL_FITTERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("label", [LABEL_SPHERE, LABEL_BOX, LABEL_CYLINDER])
def test_every_fit_contains_its_cluster(clusters, fitter, label):
    """The invariant. Under-approximating an obstacle is the error that causes a collision."""
    points = clusters[label]
    primitive = fitter(points)
    inside = contains(primitive, points)
    assert inside.all(), f"{fitter.__name__} left {int((~inside).sum())}/{len(points)} points outside"


@pytest.mark.parametrize("fitter", ALL_FITTERS, ids=lambda f: f.__name__)
def test_every_fit_contains_an_elongated_cluster(fitter):
    points = _elongated()
    assert contains(fitter(points), points).all()


def test_fits_are_deterministic(clusters):
    for fitter in ALL_FITTERS:
        a, b = fitter(clusters[LABEL_BOX]), fitter(clusters[LABEL_BOX])
        assert np.array_equal(a.center, b.center)
        assert np.array_equal(a.dimensions, b.dimensions)


# ------------------------------------------------------------------------------ tightness


def test_a_capsule_beats_a_sphere_on_an_elongated_cluster():
    """Otherwise there would be no reason to have the type."""
    points = _elongated()
    sphere = fit_sphere(points)
    capsule = fit_capsule(points)
    assert capsule.dimensions[0] < 0.5 * sphere.dimensions[0]
    # And its sphere chain is genuinely tighter than the one big sphere it replaces.
    chain = to_spheres(capsule)
    assert len(chain) > 1
    assert max(r for _, r in chain) < sphere.dimensions[0]


def test_a_sphere_fit_is_near_exact_on_a_sphere(clusters):
    """The fixture's target is an analytic 4 cm sphere seen from one side."""
    points = clusters[LABEL_SPHERE]
    primitive = fit_sphere(points)
    assert 0.02 < primitive.dimensions[0] < 0.05


def test_a_box_fit_recovers_the_true_half_extents(clusters):
    """The fixture box is 4.5 x 4.5 x 5.0 cm; a single view sees three faces of it."""
    primitive = fit_box(clusters[LABEL_BOX])
    assert primitive.type is PrimitiveType.BOX
    assert np.all(primitive.dimensions < 0.08)
    assert np.all(primitive.dimensions > 0.005)


def test_mvee_is_no_looser_than_the_covariance_ellipsoid(clusters):
    """The optional reuse of `Baseline.knows_vla`'s Khachiyan solver, when it is importable."""
    points = clusters[LABEL_CYLINDER]
    covariance = fit_ellipsoid(points)
    mvee = fit_ellipsoid_mvee(points)
    assert contains(mvee, points).all()
    assert float(np.prod(mvee.dimensions)) <= float(np.prod(covariance.dimensions)) * 1.05


# ------------------------------------------------------------------------------ orientation


@pytest.mark.parametrize("fitter", [fit_capsule, fit_box, fit_ellipsoid], ids=lambda f: f.__name__)
def test_orientation_is_a_proper_rotation(fitter):
    """A reflection is a valid eigenbasis but not a valid pose; TO would mirror the obstacle."""
    primitive = fitter(_elongated())
    R = primitive.orientation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(R)) == pytest.approx(1.0)


def test_the_capsule_axis_is_the_long_direction():
    """`Primitive` documents the segment as lying along local +z, and `to_spheres` relies on it."""
    axis = np.array([1.0, 1.0, 0.5])
    axis /= np.linalg.norm(axis)
    primitive = fit_capsule(_elongated())
    alignment = abs(float(primitive.orientation[:, 2] @ axis))
    assert alignment > 0.99, f"capsule axis off by {np.degrees(np.arccos(alignment)):.1f} deg"


# -------------------------------------------------------------------------- fallback chain


def test_priority_order_matches_the_spec():
    assert PRIORITY == (
        PrimitiveType.SPHERE, PrimitiveType.CAPSULE, PrimitiveType.BOX, PrimitiveType.ELLIPSOID
    )


@pytest.mark.parametrize("requested", list(PrimitiveType))
def test_a_two_point_cluster_degrades_instead_of_raising(requested):
    """A candidate that produces no shape produces no constraint, and that is a collision."""
    points = np.array([[0.5, 0.0, 0.9], [0.52, 0.0, 0.9]])
    primitive = fit_primitive(points, requested, min_radius=0.005)
    assert np.all(np.isfinite(primitive.center)) and np.all(primitive.dimensions > 0)
    assert contains(primitive, points).all()


@pytest.mark.parametrize("requested", list(PrimitiveType))
def test_fit_primitive_honours_the_request_when_it_can(clusters, requested):
    primitive = fit_primitive(clusters[LABEL_BOX], requested)
    assert primitive.type is requested
    assert contains(primitive, clusters[LABEL_BOX]).all()


def test_fit_primitive_rejects_an_empty_cluster():
    with pytest.raises(ValueError, match="empty cluster"):
        fit_primitive(np.zeros((0, 3)))


def test_min_radius_floors_a_degenerate_cluster():
    """A zero-radius sphere is a constraint nothing can push against."""
    coincident = np.tile([0.5, 0.0, 0.9], (5, 1))
    primitive = fit_sphere(coincident, min_radius=0.01)
    assert primitive.dimensions[0] == pytest.approx(0.01)


# ------------------------------------------------------------------- constraint spheres


def test_to_spheres_is_the_identity_for_a_sphere(clusters):
    primitive = fit_sphere(clusters[LABEL_SPHERE])
    chain = to_spheres(primitive)
    assert len(chain) == 1
    assert np.allclose(chain[0][0], primitive.center)
    assert chain[0][1] == pytest.approx(primitive.dimensions[0])


def test_spacing_alone_would_not_cover_the_capsule():
    """The correction that motivates inflating the chain at all.

    "Centres at most one radius apart, therefore covered" is false. At centre spacing `s`, the
    capsule-surface point midway between two spheres is sqrt(r^2 + (s/2)^2) from the nearest centre,
    so even at `s = r` a plain chain leaves 11.8% of the radius outside — an under-approximation.
    Each sphere is inflated to exactly that distance, which is why the radii come back larger than
    the capsule's own.
    """
    primitive = fit_capsule(_elongated())
    radius = float(primitive.dimensions[0])
    needed = int(np.ceil(2.0 * primitive.dimensions[1] / radius)) + 1
    chain = to_spheres(primitive, max_spheres=needed)

    centres = np.stack([c for c, _ in chain])
    spacing = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).max())
    assert spacing <= radius + 1e-9  # the plain spacing rule still holds ...
    expected = float(np.hypot(radius, spacing / 2.0))
    assert all(r == pytest.approx(expected, rel=1e-6) for _, r in chain)  # ... but radii are inflated
    assert 1.0 < expected / radius <= np.sqrt(1.25) + 1e-9


@pytest.mark.parametrize("fitter", [fit_box, fit_ellipsoid], ids=lambda f: f.__name__)
def test_box_and_ellipsoid_fall_back_to_a_bounding_sphere(clusters, fitter):
    """Documented limitation: the sphere-vs-sphere constraint form cannot express them tightly yet.

    Their pose and dimensions still travel to TO intact, so a future optimizer with box/ellipsoid
    distance support can use them without AG3S changing.
    """
    points = clusters[LABEL_BOX]
    primitive = fitter(points)
    chain = to_spheres(primitive)
    assert len(chain) == 1
    covered = np.linalg.norm(points - chain[0][0], axis=1) <= chain[0][1] + 1e-9
    assert covered.all()


@pytest.mark.parametrize("max_spheres", [2, 3, 5, 8, 24])
def test_a_capped_capsule_chain_still_covers_the_cluster(max_spheres):
    """The cap must not open gaps.

    With `max_spheres` binding, the centres sit further apart than one radius; each sphere is
    inflated to sqrt(r^2 + (spacing/2)^2) so the union still contains the capsule. Getting this
    wrong is an under-approximation, which is the error direction that causes a collision.
    """
    points = _elongated()
    primitive = fit_capsule(points)
    chain = to_spheres(primitive, max_spheres=max_spheres)
    assert len(chain) <= max_spheres

    covered = np.zeros(len(points), bool)
    for centre, radius in chain:
        covered |= np.linalg.norm(points - centre, axis=1) <= radius + 1e-9
    assert covered.all(), f"max_spheres={max_spheres} left {int((~covered).sum())} points uncovered"

    # And the midpoint between consecutive centres — the deepest gap — is covered too.
    centres = np.stack([c for c, _ in chain])
    if len(chain) > 1:
        midpoints = (centres[:-1] + centres[1:]) / 2.0
        radii = np.array([r for _, r in chain])
        reach = np.linalg.norm(midpoints - centres[:-1], axis=1) - radii[:-1]
        assert reach.max() <= 1e-9


# ------------------------------------------------------------------------------- helpers


def test_rms_radius_is_not_driven_by_a_single_outlier():
    """Why compactness uses RMS: the max would rank a clean object below a noisy one."""
    blob = np.random.default_rng(0).normal(0.0, 0.01, (500, 3))
    with_outlier = np.vstack([blob, [[1.0, 0.0, 0.0]]])
    assert rms_radius(with_outlier) < 3.0 * rms_radius(blob)
    assert np.linalg.norm(with_outlier, axis=1).max() > 30.0 * rms_radius(blob)


def test_rms_radius_of_a_single_point_is_zero():
    assert rms_radius(np.array([[1.0, 2.0, 3.0]])) == 0.0
