"""Compare sam3-mlx text-prompt segmentation with upstream PyTorch SAM 3 end to end.

Each medical fixture image is segmented with its ``target`` from
``tests/fixtures/medical/sources.json`` as the text prompt, at the default
1008-pixel resolution and 0.5 threshold. The MLX side runs the published
``mlx-community/sam3-image`` weights; the upstream side runs
``facebookresearch/sam3`` at the pinned commit with Meta's ``sam3.pt`` on the
CPU, in a separate Python given by ``--upstream-python``, because PyTorch and the
upstream package are not dependencies of this project. Upstream's ViT MLP always
uses a BF16 fused CUDA op; the reference replaces it with the same MLP in FP32,
matching sam3-mlx's default precision, resizes in FP32 as sam3-mlx's tested
preprocessing does, and makes tensors requested on CUDA on the CPU. That Python needs torch
and torchvision as pinned here, plus timm, iopath, ftfy, regex, einops,
setuptools<81, numpy<2, scikit-image, pycocotools, psutil and huggingface_hub;
``--upstream-source`` is a checkout of the upstream repository. ``sam3.pt`` is
gated: request access to ``facebook/sam3`` on Hugging Face first.

Reported per image: raw per-query differences before thresholding (presence-
weighted score, normalized box and mask logits on the 288-pixel grid) and the
final detections (count, scores, pixel boxes, full-resolution mask IoU).
``--record tests/fixtures/upstream_text_prompts.json`` writes the upstream
detections as the reference for ``tests/image/test_real_weights.py``.

Usage::

    python scripts/compare_upstream_image.py \\
        --upstream-python /path/to/oracle/bin/python --upstream-source /path/to/sam3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "medical"
UPSTREAM_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"
SAM3_PT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
MLX_WEIGHTS_REVISION = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
RESOLUTION = 1008
THRESHOLD = 0.5


def _cases() -> list[tuple[str, str]]:
    """Return ``(file, text prompt)`` for every medical fixture."""
    images = json.loads((FIXTURES / "sources.json").read_text())["images"]
    return [(image["file"], image["target"]) for image in images]


def _run_upstream(source: Path, output: Path) -> None:
    """Segment every case with upstream SAM 3 on the CPU and save the arrays to ``output``."""
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    # sam3.model.edt holds a Triton kernel for video tracking and is imported
    # when the package loads. Triton has no macOS build and image grounding never
    # calls it, so the module is replaced before import.
    edt = types.ModuleType("sam3.model.edt")
    setattr(edt, "edt_triton", None)
    sys.modules["sam3.model.edt"] = edt
    sys.path.insert(0, str(source))
    from torch.overrides import TorchFunctionMode

    import sam3.model.vitdet as upstream_vitdet
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Upstream's ViT MLP always runs fc1 + GELU through a BF16 fused CUDA op
    # (sam3.perflib.fused.addmm_act); Meta's examples run the whole model under
    # BF16 autocast. sam3-mlx defaults to FP32, so the reference uses the same
    # MLP in FP32: fc1, exact GELU, then fc2. Nothing else in upstream changes.
    def _fp32_addmm_act(activation, linear, mat1):
        return activation()(linear(mat1))

    setattr(upstream_vitdet, "addmm_act", _fp32_addmm_act)

    class _CpuForCuda(TorchFunctionMode):
        """Keep upstream on the CPU: map ``device="cuda"`` there and skip pinning.

        Upstream builds its position-encoding and box-coordinate caches with a
        hard-coded CUDA device and pins box scales for transfer; neither changes
        any value.
        """

        def __torch_function__(self, func, types, args=(), kwargs=None):
            # Pinned host memory only speeds CUDA copies; on macOS it targets MPS.
            if func is torch.Tensor.pin_memory:
                return args[0]
            kwargs = dict(kwargs or {})
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return func(*args, **kwargs)

    checkpoint = hf_hub_download("facebook/sam3", "sam3.pt", revision=SAM3_PT_REVISION)
    arrays: dict[str, np.ndarray] = {}
    with _CpuForCuda(), torch.inference_mode():
        model = build_sam3_image_model(
            device="cpu", checkpoint_path=checkpoint, load_from_HF=False
        )
        processor = Sam3Processor(
            model, resolution=RESOLUTION, device="cpu", confidence_threshold=THRESHOLD
        )
        # TorchVision 0.28 resizes uint8 images with a different native kernel on
        # ARM; the FP32 resize rounded back to uint8 is the preprocessing contract
        # sam3-mlx matches pixel for pixel (tests/_resize_reference.py). Upstream
        # outputs move visibly with one-level pixel changes, so both sides must
        # start from the same pixels.
        from torchvision.transforms import v2

        def _fp32_resize(image: torch.Tensor) -> torch.Tensor:
            resized = v2.functional.resize(image.to(torch.float32), [RESOLUTION, RESOLUTION])
            return resized.round().to(torch.uint8)

        processor.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Lambda(_fp32_resize),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        for index, (name, prompt) in enumerate(_cases()):
            image = Image.open(FIXTURES / name).convert("RGB")
            state = processor.set_text_prompt(prompt, processor.set_image(image))
            raw = model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )
            arrays[f"{index}_logits"] = raw["pred_logits"][0, :, 0].numpy()
            arrays[f"{index}_presence"] = raw["presence_logit_dec"][0].numpy()
            arrays[f"{index}_boxes"] = raw["pred_boxes"][0].numpy()
            arrays[f"{index}_masks"] = raw["pred_masks"][0].numpy()
            arrays[f"{index}_final_scores"] = state["scores"].numpy()
            arrays[f"{index}_final_boxes"] = state["boxes"].numpy()
            arrays[f"{index}_final_masks"] = state["masks"][:, 0].numpy()
            print(f"upstream {name}: {len(state['scores'])} detections", flush=True)
    np.savez(output, **arrays)


def _run_mlx() -> dict[str, np.ndarray]:
    """Segment every case with sam3-mlx and return the same arrays as the upstream side."""
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from sam3_mlx import build_sam3_image_model
    from sam3_mlx.mlx_runtime import to_numpy
    from sam3_mlx.model.sam3_image_processor import Sam3Processor

    checkpoint = hf_hub_download(
        "mlx-community/sam3-image", "model.safetensors", revision=MLX_WEIGHTS_REVISION
    )
    model = build_sam3_image_model(checkpoint_path=checkpoint)
    processor = Sam3Processor(model, resolution=RESOLUTION, confidence_threshold=THRESHOLD)
    arrays: dict[str, np.ndarray] = {}
    for index, (name, prompt) in enumerate(_cases()):
        image = Image.open(FIXTURES / name).convert("RGB")
        state = processor.set_text_prompt(prompt, processor.set_image(image))
        raw = processor.predict_raw(state)
        arrays[f"{index}_logits"] = to_numpy(raw["pred_logits"][0, :, 0])
        arrays[f"{index}_presence"] = to_numpy(raw["presence_logit_dec"][0])
        arrays[f"{index}_boxes"] = to_numpy(raw["pred_boxes"][0])
        arrays[f"{index}_masks"] = to_numpy(raw["pred_masks"][0])
        arrays[f"{index}_final_scores"] = to_numpy(state["scores"])
        arrays[f"{index}_final_boxes"] = to_numpy(state["boxes"])
        arrays[f"{index}_final_masks"] = to_numpy(state["masks"])[:, 0]
    return arrays


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values.astype(np.float64)))


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(np.logical_and(first, second).sum() / union)


def _compare(mlx: dict[str, np.ndarray], upstream: dict[str, np.ndarray]) -> bool:
    """Print per-image differences; return whether every image kept the same detections."""
    same = True
    for index, (name, prompt) in enumerate(_cases()):
        def pair(key: str, index: int = index) -> tuple[np.ndarray, np.ndarray]:
            return mlx[f"{index}_{key}"], upstream[f"{index}_{key}"]

        scores = [
            _sigmoid(logits) * _sigmoid(presence)
            for logits, presence in zip(pair("logits"), pair("presence"), strict=True)
        ]
        mlx_masks, upstream_masks = pair("masks")
        active = scores[1] > 0.1
        raw_ious = [_iou(a > 0, b > 0) for a, b in zip(mlx_masks[active], upstream_masks[active])]
        print(
            f"{name} ({prompt!r}): raw max|Δscore| {np.abs(scores[0] - scores[1]).max():.2e}, "
            f"max|Δbox| {np.abs(np.subtract(*pair('boxes'))).max():.2e}, "
            f"max|Δmask logit| {np.abs(mlx_masks - upstream_masks).max():.2e}, "
            f"min mask IoU over {int(active.sum())} queries above 0.1 "
            f"{min(raw_ious, default=1.0):.4f}"
        )
        mlx_scores, upstream_scores = pair("final_scores")
        if len(mlx_scores) != len(upstream_scores):
            same = False
            print(f"  detections differ: MLX {len(mlx_scores)}, upstream {len(upstream_scores)}")
            continue
        if len(mlx_scores) == 0:
            print("  no detections on either side")
            continue
        final_masks = pair("final_masks")
        ious = [_iou(a, b) for a, b in zip(*final_masks, strict=True)]
        differing = [int((a != b).sum()) for a, b in zip(*final_masks, strict=True)]
        print(
            f"  {len(mlx_scores)} detections: max|Δscore| "
            f"{np.abs(mlx_scores - upstream_scores).max():.2e}, max|Δbox| "
            f"{np.abs(np.subtract(*pair('final_boxes'))).max():.3f} px, mask IoU "
            f"{[round(value, 4) for value in ious]}, differing pixels {differing}"
        )
    return same


def _record(upstream: dict[str, np.ndarray], path: Path) -> None:
    """Write the upstream detections per image as the real-weights test reference."""
    images = []
    for index, (name, prompt) in enumerate(_cases()):
        masks = upstream[f"{index}_final_masks"]
        images.append({
            "file": name,
            "text_prompt": prompt,
            "scores": [round(float(value), 6) for value in upstream[f"{index}_final_scores"]],
            "boxes_xyxy": [
                [round(float(value), 3) for value in box]
                for box in upstream[f"{index}_final_boxes"]
            ],
            "mask_areas": [int(mask.sum()) for mask in masks],
        })
    reference = {
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": f"facebook/sam3@{SAM3_PT_REVISION}/sam3.pt",
        "device": "cpu",
        "resolution": RESOLUTION,
        "confidence_threshold": THRESHOLD,
        "images": images,
    }
    path.write_text(json.dumps(reference, indent=2) + "\n")
    print(f"wrote {path}")


def main() -> None:
    """Run both sides and compare, or run the upstream side alone with ``--upstream-output``."""
    parser = argparse.ArgumentParser(description="Compare sam3-mlx with upstream SAM 3.")
    parser.add_argument("--upstream-python", type=Path)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--upstream-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help="Write the upstream reference JSON here")
    args = parser.parse_args()
    if args.upstream_output is not None:
        _run_upstream(args.upstream_source, args.upstream_output)
        return
    if args.upstream_python is None:
        parser.error("--upstream-python is required")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "upstream.npz"
        subprocess.run(
            [str(args.upstream_python), "-I", __file__, "--upstream-source",
             str(args.upstream_source), "--upstream-output", str(output)],
            check=True,
        )
        with np.load(output) as stored:
            upstream = dict(stored)
    same = _compare(_run_mlx(), upstream)
    if args.record is not None:
        _record(upstream, args.record)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
