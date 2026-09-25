"""Run one PREMAP2 input-splitting job for a frozen Q64 CPWL MLP.

This deliberately invokes PREMAP2's public Python wrapper and only supplies
paths/configuration from this isolated study.  It never imports or changes the
It uses only the pinned external PREMAP2 checkout and the local adapter.
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
import pickle
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY / "external" / "Premap2"
PREMAP_SOURCE = REPO / "PreimageApproxForNNs" / "src"
REPO_SOURCE = REPO / "PreimageApproxForNNs"
WRAPPER_SOURCE = REPO / "src"
ONNX = STUDY / "adapter" / "cpwl_q64.onnx"
VNNLIB = STUDY / "results" / "adapter" / "property_g0p4.vnnlib"
OUTPUT_ROOT = Path(os.environ.get("TRIO_PREMAP_OUTPUT", STUDY / "premap2"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-budget", type=int, default=512)
    parser.add_argument("--tag", required=True, help="result subdirectory")
    parser.add_argument("--onnx-path", type=Path, default=ONNX)
    parser.add_argument("--vnnlib-path", type=Path, default=VNNLIB)
    args = parser.parse_args()
    mode = "under"
    output = STUDY / "results" / args.tag / mode
    onnx_path = args.onnx_path.resolve()
    vnnlib_path = args.vnnlib_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not onnx_path.is_file() or not vnnlib_path.is_file():
        raise FileNotFoundError("Run scripts/test_setup.py before this pilot.")

    # This mirrors the supported source-tree use on Windows, where the
    # upstream POSIX symlinks cannot be represented as import directories.
    sys.path[:0] = [str(WRAPPER_SOURCE), str(PREMAP_SOURCE), str(REPO_SOURCE)]
    from premap2.wrapper import premap

    def released_loader_bridge(_config: object) -> None:
        """Bridge two released-loader compatibility omissions without patching it.

        PREMAP2 v0.3.0 evaluates its default ONNX-loader string in a namespace
        where that loader is not imported.  Its loader also passes the newer
        optional ``quirks`` keyword to the exact older onnx2pytorch revision
        pinned in PREMAP2's lockfile.  The bridge retains PREMAP2's published
        parser/converter/algorithm but omits that unsupported *optional* empty
        keyword.  No released PREMAP2 source file is changed.
        """
        import preimage_main  # type: ignore
        import onnx
        import torch
        from onnx2pytorch import ConvertModel
        from read_vnnlib import read_vnnlib  # type: ignore

        def released_compat_onnx_loader(file_root: str, onnx_path: str, vnnlib_path: str):
            vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
            onnx_model = onnx.load(os.path.join(file_root, onnx_path))
            inputs = [node for node in onnx_model.graph.input if node.name not in {item.name for item in onnx_model.graph.initializer}]
            input_shape = tuple(dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim[1:])
            model = ConvertModel(onnx_model, experimental=True)
            model.eval().to(dtype=torch.get_default_dtype())
            return model, (-1, *input_shape), vnnlib

        preimage_main.released_compat_onnx_loader = released_compat_onnx_loader

        # The released ``demo`` fixture encodes box bounds as integer tensors.
        # PREMAP2's own coverage sampler then feeds those tensors to Uniform,
        # which only accepts floating bounds.  Supply the identical [-1,1]^2
        # fixture in float form for this continuous domain adapter.
        import preimage_model_utils  # type: ignore
        import preimage_batch_approx_input_split  # type: ignore
        import test_polyhedron_util  # type: ignore

        original_load_input_bounds = preimage_model_utils.load_input_bounds

        def continuous_demo_bounds(dataset: str, truth_label: int, quant: bool, trans: bool):
            if dataset == "demo":
                return (
                    torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                    torch.tensor([truth_label], dtype=torch.long),
                    torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                    torch.tensor([[-1.0, -1.0]], dtype=torch.float32),
                    None,
                )
            return original_load_input_bounds(dataset, truth_label, quant, trans)

        preimage_model_utils.load_input_bounds = continuous_demo_bounds
        test_polyhedron_util.load_input_bounds = continuous_demo_bounds

        def compat_load_model(*_args, **_kwargs):
            # Used only by PREMAP2's Monte-Carlo coverage reporter.  It must
            # evaluate the same converted frozen ONNX model as the verifier.
            return released_compat_onnx_loader("", str(onnx_path), str(vnnlib_path))[0]

        test_polyhedron_util.load_model = compat_load_model

        def compatible_over_tightness(sample_res, preimg_idx):
            # Upstream mixes NumPy affine products with Torch biases.  Convert
            # either representation explicitly before the released statistic.
            array = sample_res.detach().cpu().numpy() if hasattr(sample_res, "detach") else __import__("numpy").asarray(sample_res)
            if preimg_idx is not None:
                array = array[:, preimg_idx]
            reduced = __import__("numpy").min(array, axis=0)
            return float(__import__("numpy").mean(1.0 / (1.0 + __import__("numpy").exp(-reduced))))

        test_polyhedron_util.calc_over_tightness = compatible_over_tightness

        # PREMAP2's coverage *heuristic* samples every split cell, including
        # zero-width cells generated by repeated bisection.  Torch rejects a
        # Uniform distribution with equal endpoints.  Preserve the real bound
        # computation and use the closest representable upper endpoint only
        # for that auxiliary sampling call.
        import auto_LiRPA.perturbations as perturbations  # type: ignore

        original_uniform = perturbations.Uniform

        def degenerate_safe_uniform(low, high, *args, **kwargs):
            adjusted_high = torch.where(
                high <= low,
                torch.nextafter(low, torch.full_like(low, float("inf"))),
                high,
            )
            return original_uniform(low, adjusted_high, *args, **kwargs)

        perturbations.Uniform = degenerate_safe_uniform
        # ``test_polyhedron_util`` imports the class by value, so update that
        # released helper's local reference as well.  This remains confined to
        # the auxiliary coverage sampler used by the pilot process.
        test_polyhedron_util.Uniform = degenerate_safe_uniform

        original_get_preimage_info = preimage_batch_approx_input_split.get_preimage_info

        def capture_preimage_info(domains, bound_under, bound_over):
            result = original_get_preimage_info(domains, bound_under, bound_over)
            # The released main routine intentionally discards its final
            # polyhedral representation.  Capture exactly that returned
            # detached artifact in the study wrapper without touching PREMAP2.
            capture_dir = output / "captured_representation"
            capture_dir.mkdir(parents=True, exist_ok=True)
            with (capture_dir / "final_preimage.pkl").open("wb") as handle:
                pickle.dump(result, handle)
            return result

        preimage_batch_approx_input_split.get_preimage_info = capture_preimage_info

    started = time.time()
    paths = premap(
        premap_path=str(PREMAP_SOURCE),
        post_config=released_loader_bridge,
        onnx_path=str(onnx_path),
        vnnlib_path=str(vnnlib_path),
        onnx_loader="released_compat_onnx_loader",
        # PREMAP2's released coverage reporter names its full [-1,1]^2 box
        # fixture "demo".  This exactly matches our normalized rectangle.
        dataset="demo",
        label=0,
        result_dir=str(output),
        under_approx=True,
        over_approx=False,
        enable_input_split=True,
        branching_method="preimg",
        timeout=300.0,
        # PREMAP2's released under-approximation input-splitting loop does not
        # consult ``timeout``; this explicit, recorded cap bounds the pilot.
        branch_budget=args.branch_budget,
        device="cpu",
        complete_verifier="bab",
        seed=101,
        # Keep the returned artifacts explicit; no dataset adapter is used.
        start=0,
        end=1,
        silent=False,
    )
    summary = {
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "released_premap2_repo": "https://github.com/Aggrathon/Premap2",
        "released_premap2_commit": "cdb0f412f82ca7df3a7d72fcdf567a4a952c0e64",
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnnlib_path),
        "returned_paths": [str(p) for p in paths],
        "configuration": {
            "under_approx": True,
            "over_approx": False,
            "enable_input_split": True,
            "timeout_seconds": 300.0,
            "branch_budget": args.branch_budget,
            "domain_normalized": [[-1.0, 1.0], [-1.0, 1.0]],
            "threshold": 0.4,
        },
    }
    summary["mode"] = mode
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
