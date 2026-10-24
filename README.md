# sam3-mlx

Unofficial native MLX port of
[SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3) image-segmentation
components for Apple Silicon.

Segment objects in images from a text prompt (`"shoe"`) or boxes, locally on Mac.

- SAM 3 image segmentation with text and geometric prompts
- Pinned default checkpoint, verified by revision
- Selected-frame video: per-frame segmentation over image folders or videos
- Unsupported paths fail explicitly; no silent PyTorch/CUDA fallback

> [!NOTE]
> SAM 3.1 Object Multiplex and temporal tracking are experimental and not covered by
> the image-runtime parity claim.

Part of [MedMLX](https://github.com/MedMLX).
[Technical notes](https://github.com/MedMLX/sam3-mlx/blob/main/docs/technical.md) · [Usage](https://github.com/MedMLX/sam3-mlx/blob/main/docs/usage.md) · [Verification](https://github.com/MedMLX/sam3-mlx/blob/main/docs/verification.md)

[Quick start](#quick-start) · [Input and output](#input-and-output) · [Technical details](#technical-details) · [License](#license)

Runtime wheels and source distributions omit development tools, tests, benchmarks
and qualification evidence. Those workflows remain available in the public Git
checkout; see the [verification workflow](https://github.com/MedMLX/sam3-mlx/blob/main/docs/verification.md).

## Requirements

Apple Silicon Mac with Metal, Python 3.13 or newer, MLX 0.32.4+ (< 0.33) and
[uv](https://docs.astral.sh/uv/). Default image weights download automatically
from a pinned Hugging Face revision.

Experimental SAM 3.1 builders can also download and convert their pinned official
checkpoint, then reuse a verified local bundle. The first conversion requires
PyTorch. See [setup details](https://github.com/MedMLX/sam3-mlx/blob/main/docs/usage.md#experimental-sam-31-checkpoint-setup).

## Quick start

Install the package and optional extras:

```bash
uv pip install sam3-mlx            # or: uv add sam3-mlx
uv pip install "sam3-mlx[viz]"     # optional: matplotlib plotting helpers
uv pip install "sam3-mlx[video]"   # optional: OpenCV video decoding
```

Build the image model and segment from a text prompt:

```python
from sam3_mlx import build_sam3_image_model
from sam3_mlx.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, resolution=1008)

state = processor.set_image("image.jpg")
state = processor.set_text_prompt("shoe", state)

masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
```

`resolution` must be a multiple of 14; lower is faster but less detailed. `1008`,
`672` and `504` are covered by the release parity checks.

For video frames:

```python
from sam3_mlx import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3")
```

This default runs independent image inference on each frame. An explicit
tracker checkpoint enables experimental detection and memory tracking with
persistent IDs and mask or point refinement. Setup, limitations, reduced precision, and checkpoint
conversion are in [docs/usage.md](https://github.com/MedMLX/sam3-mlx/blob/main/docs/usage.md).

## Input and output

Image inference takes an image plus text or geometric prompts and returns
masks, boxes, and confidence scores. Selected-frame video inference applies
the image model independently to each frame. Temporal tracking and SAM 3.1
multiplex retain the experimental scope described below.

### MedMLX integration

Installing this package registers a `medmlx.models` entry point
(`sam3_mlx.medmlx:MODEL`) on top of the stable image and selected-frame APIs
above. `sam3_mlx.medmlx.run` segments one image from a text/box/point prompt,
or one chosen frame of an image folder or video, and writes `masks.npy` and
`prediction.json` to an output directory. It does not add tracking or
multiplex capability; see [`sam3_mlx/medmlx.py`](https://github.com/MedMLX/sam3-mlx/blob/main/sam3_mlx/medmlx.py).

## Technical details

### Verification scope

The verification workflow compares raw outputs with the pinned official PyTorch
implementation. See [docs/verification.md](https://github.com/MedMLX/sam3-mlx/blob/main/docs/verification.md).

The [image parity reports](https://github.com/MedMLX/sam3-mlx/blob/main/parity/README.md)
record the tested source commit, checkpoint identities, frozen prompt/resolution
matrix, raw outputs, and numerical limits. A release requires fresh source-bound
image evidence and exact checkpoint lineage; these small fixtures do not measure
representative task accuracy.

### Support limits

Unsupported paths raise `Sam3MlxUnsupportedError`:

- **Apple Silicon / MLX only.**
- **Interactive image inference needs separate head weights.** Its full-model
  comparison covers a small pinned image and prompt fixture.
- **Video tracking requires an explicit tracker attachment.** Semantic detection
  can be followed by mask or point refinement. Official temporal equivalence
  remains unqualified.
- **SAM 3.1 multiplex is experimental.**
- **No training or benchmark evaluation.** Export predictions and run an official
  evaluator separately.
- **Parity is scoped** to pinned text and geometric image cases at 1008/672/504.

Development commands, oracle setup and typing checks are in
[docs/technical.md](https://github.com/MedMLX/sam3-mlx/blob/main/docs/technical.md#development-checks).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Portions of this repository are derived from, adapted from, or structured for
parity with the official SAM 3 implementation
([`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)). Original
SAM materials are copyright Meta Platforms, Inc. and distributed under the SAM
License.

This repository does not ship official Meta checkpoint weights or converted SAM
checkpoint weights. Official checkpoints are on Hugging Face
([`facebook/sam3`](https://huggingface.co/facebook/sam3),
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)); comply with the SAM
License and applicable access terms.

Distributed under the SAM License; see [`LICENSE`](https://github.com/MedMLX/sam3-mlx/blob/main/LICENSE). Not affiliated with
or endorsed by Meta. The CLIP/open_clip tokenizer components retain their MIT
terms; see [THIRD_PARTY_NOTICES.md](https://github.com/MedMLX/sam3-mlx/blob/main/THIRD_PARTY_NOTICES.md) for attribution and
full license texts.

## Implementation mapping

[Central implementation map](https://github.com/MedMLX/Med-MLX/blob/main/implementation-maps/sam3-mlx.json) records upstream-to-native changes, source references and qualification scope.
