import numpy as np

from experiments.projection_cost_ablation.backbones import build_variants, deterministic_queries, load_config
from experiments.projection_cost_ablation.geometry import validate_geometry
from experiments.projection_cost_ablation.timing import PreparedProjection, time_compilation, time_projection


def test_projection_cost_parameter_counts_and_geometry():
    config = load_config()
    first, spatial_first = build_variants(config, 101)
    second, spatial_second = build_variants(config, 101)
    assert {key: item.parameter_count for key, item in first.items()} == config["expected_parameter_counts"]
    np.testing.assert_array_equal(spatial_first["centers"], spatial_second["centers"])
    np.testing.assert_array_equal(spatial_first["raw_diag"], spatial_second["raw_diag"])
    np.testing.assert_array_equal(spatial_first["offdiag"], spatial_second["offdiag"])
    np.testing.assert_array_equal(spatial_first["desired_radii"], spatial_second["desired_radii"])
    report = validate_geometry(
        first, spatial_first["desired_radii"], config["target"],
        tolerance=config["geometry_radius_tolerance"], grid_resolution=config["membership_grid_resolution"],
    )
    assert all(row["active_ellipsoids"] == 128 for row in report["checks"])
    assert all(row["membership_mismatches"] == 0 for row in report["checks"])


def test_projection_cost_small_timing_smoke():
    config = load_config()
    variants, _ = build_variants(config, 101)
    queries = deterministic_queries(config, 101)[:1]
    prepared = {
        key: PreparedProjection(variant.frozen, config["target"], queries)
        for key, variant in variants.items()
    }
    for key, variant in variants.items():
        rows = time_compilation(variant.frozen, config["target"], repetitions=1)
        assert rows[0]["elapsed_seconds"] > 0
        assert rows[0]["active_ellipsoids"] == 128
        assert prepared[key].cached_eigendecompositions == 128
    rows, summaries = time_projection(prepared, repetitions=1, warmups=0)
    assert len(rows) == 5
    assert all(summary["mean_seconds"] > 0 for summary in summaries.values())
