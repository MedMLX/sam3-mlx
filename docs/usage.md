# sam3-mlx usage notes

Details behind the [README](../README.md).

## Checkpoints and precision

Default image weights download from a pinned `mlx-community/sam3-image`
revision, and its conversion manifest is checked before model mutation.

To preserve checkpoint source details, convert from the official PyTorch checkpoint with
`convert_from_pytorch=True` and an immutable source revision. Conversion writes
`conversion-manifest.json` with source and output checkpoint names, mapped-key
counts, ignored tracker keys, and dtype counts. Publication is lock-protected,
validated in a temporary sibling directory, and makes the manifest visible
last. PyTorch is intentionally not installed by a package extra; use a separate
conversion environment.

FP16, BF16, and mixed runtime policies are available but are not covered by the
checked-in FP32 image parity reports. Reduced-precision Hub loads must use a
separate immutable artifact whose manifest proves its dtype inventory:

```python
model = build_sam3_image_model(
    precision="fp16",
    hf_repo="owner/sam3-image-fp16",
    hf_revision="<40-character-commit>",
)
```

Alternatively, create a local precision-specific checkpoint directly from the
pinned official source:

```python
model = build_sam3_image_model(
    precision="fp16",
    convert_from_pytorch=True,
    conversion_source_revision="<40-character-official-commit>",
)
```

## Image resolution

`Sam3Processor.resolution` is the square image size fed into the ViT backbone. It
must be a positive multiple of the patch size (`14`). Lower resolutions are
faster but lose detail. Exact local-attention window grids are multiples of
`336` (`14 × 24`): `336`, `672`, and `1008`. `336` is an optional fast tier, not
an accuracy-equivalent default. `504` is still in the frozen release matrix even
though it pads to the same `48×48` window topology as `672`.

The runtime accepts other multiples of 14, but the frozen official-vs-MLX release
matrix covers only `1008`, `672`, and `504`.

## Selected-frame video and advanced options

`build_sam3_predictor()` defaults to the selected-frame SAM 3 path. Request
`version="sam3.1"` explicitly for the experimental multiplex predictor; that
path prepares pinned official weights automatically when `checkpoint_path` is
omitted and `load_from_HF=True`. Its public builders default to the implemented
complex-pair RoPE path. Real-arithmetic RoPE remains unsupported and
`use_rope_real=True` fails fast.

The selected-frame path accepts image paths, image folders, PIL image sequences,
and OpenCV-decodable video files. It performs independent image inference on
each frame:

- `out_obj_ids` are frame-local detection IDs;
- no persistent temporal identity or tracker memory is claimed;
- cross-frame `remove_object` is rejected;
- prompt mutation during propagation is rejected except cancellation;
- full-resolution output history is not retained;
- image folders and video files use bounded host caches;
- encoded frame and text features are bounded per session.

Asynchronous image-folder preloading is rejected in selected-frame mode because
the legacy loader retains every decoded frame and would violate the bounded
memory contract.

Experimental tracker and multiplex builders are available under
`sam3_mlx.experimental`. Import them from that namespace:

```python
from sam3_mlx.experimental import build_tracker, build_sam3_multiplex_video_predictor
```

For direct frame loading, use `sam3_mlx.model.io_utils.load_resource_as_video_frames`,
which returns a frame provider with original dimensions and on-demand frame access.

Official PyTorch checkpoint downloads default to the version's canonical
immutable revision. An explicit override must also be a full commit SHA:

```python
from sam3_mlx import download_ckpt_from_hf

checkpoint = download_ckpt_from_hf("sam3")
custom_checkpoint = download_ckpt_from_hf("sam3", revision="<40-character-commit>")
```

## Experimental SAM 3.1 checkpoint setup

SAM3.1 setup reuses the pinned local source at
`checkpoints/official/facebook-sam31/sam3.1_multiplex.pt` when present, or
downloads the pinned official source. First conversion requires PyTorch in the
conversion environment and writes raw weights to
`checkpoints/derived/sam31-raw-fp32`; `local_weights_dir` selects another bundle
directory. Cache reuse checks the pinned source revision and the
manifest's weight and index bindings before strict loading, without
deserializing the Torch source again. An invalid existing bundle raises instead
of being overwritten. An explicit `checkpoint_path` takes precedence over
automatic setup.

```python
predictor = build_sam3_predictor(version="sam3.1")
```

Repository users can install the locked conversion dependencies with
`uv sync --group oracle`. Setup tests exercise small synthetic checkpoints; they
do not establish real-checkpoint temporal parity or change the selected-frame
default.

## Experimental SAM3 mask tracking

The SAM3 predictor can attach a loaded memory tracker explicitly through
`tracker_checkpoint_path`. This path supports semantic detection with persistent
IDs, caller-supplied mask seeds, and mask or point refinement. It uses the loaded
image model's SAM2 neck and the original SAM3 tracker weights at `1008`; the
ordinary predictor default remains selected-frame inference.

Prepare the tracker artifact from the official checkpoint pinned in
`sam3_mlx.release_contract`. The source path must be relative to this repository;
the converter rejects an existing output directory. This conversion command requires
the public Git checkout; its scripts are omitted from runtime distributions:

```bash
uv run --no-sync --group oracle --with einops python \
  scripts/convert_sam3_tracker_checkpoint.py \
  --source checkpoints/official/facebook-sam3/sam3.pt \
  --output-dir checkpoints/derived/sam3-tracker-raw-fp32
```

The artifact retains every `tracker.*` tensor in its original layout. The
canonical loaders normalize the 149 image-head tensors and the full tracker
independently, with strict key and shape coverage. Its schema-versioned manifest
records the original revision, tensor count, and dtype policy. The converter shares the raw tensor conversion and atomic writer used
by the SAM3.1 conversion path.

```python
from sam3_mlx import build_sam3_predictor

predictor = build_sam3_predictor(
    version="sam3",
    checkpoint_path="checkpoints/mlx-community/sam3-image/model.safetensors",
    load_from_HF=False,
    tracker_checkpoint_path="checkpoints/derived/sam3-tracker-raw-fp32/model.safetensors",
    resolution=1008,
)
session_id = predictor.start_session("frames/")["session_id"]
predictor.add_mask(session_id, frame_idx=0, obj_id=1, mask=seed_mask)
for result in predictor.propagate_in_video(
    session_id, propagation_direction="forward", start_frame_idx=0
):
    masks = result["outputs"]["out_binary_masks"]
predictor.close_session(session_id)
```

`seed_mask` is a two-dimensional boolean mask in image coordinates. Refinement
replays that object's conditioning masks into a fresh tracker state; other
objects retain their states. New objects can be added after propagation.
Features are shared across object states and bounded by
`frame_feature_cache_size`. Cancel propagation before changing prompts through
the request API. Stale direct-generator results are discarded after a state
generation changes.

With the tracker attached, a text or box prompt starts semantic detection and
assigns persistent object IDs. Mask and point refinements can then target an
existing ID. A fresh text or box prompt replaces the tracking session. Point
refinement requires `obj_id`; points cannot include text or semantic boxes in
the same call. Direct tracker refinement rejects `use_prev_mem_frame=True`.
The SAM3.1 multiplex model does not expose the original SAM3 mask action.

Manual seeds and refinements export confidence `1.0`; semantic detections retain
their detector confidence. The tracker object probability resolves overlaps.
Offline session tests exercise scripted detections and real MLX memory kernels;
pinned-source CPU tests compare synthetic association and hotstart decisions.
Neither establishes real-weight temporal parity or release qualification.

The loaded-checkpoint session probe additionally exercises three frames with
actual detections, mask and point refinement, cached propagation, and a visual
box reset. It establishes execution of those requests, not equivalence to the
official video outputs. Tracker states remain separate per object; upstream
batches objects detected on the same frame, which can affect memory selection.

Image preprocessing defaults to the exact release path. The opt-in `fast` and
device-native `native` MLX paths are available for local evaluation, but neither
is part of the release-qualified default:

```python
processor = Sam3Processor(model, preprocessing="fast")
native_processor = Sam3Processor(model, preprocessing="native")
```

See [`verification.md`](https://github.com/MedMLX/sam3-mlx/blob/main/docs/verification.md) and [`parity/README.md`](https://github.com/MedMLX/sam3-mlx/blob/main/parity/README.md) for the
exact pass/fail scope and machine-readable reports.

Checkpoint loading requires complete required-component coverage by default. For
controlled development experiments only, `build_sam3_image_model` accepts
`strict_checkpoint_loading=False`; the resulting audit is available as
`model.checkpoint_load_report`.

## Session cleanup and memory reporting

Session close accepts an integer `clear_cache_threshold` from 0 to 100, defaulting
to 80. With `run_gc_collect=True`, it collects released objects and clears the
MLX allocator cache when active plus cached allocations reach that percentage
of Metal's recommended working-set budget. Returned `free_bytes` and `free_pct`
describe process allocator headroom against that budget, not system-wide free
GPU memory. `allocated_bytes` and `cache_bytes` are process-wide counters;
`reserved_bytes` is their sum. The snapshot also reports `active_session_count`.
After clearing the cache, close returns a fresh snapshot:

```python
response = predictor.close_session(session_id, clear_cache_threshold=80)
memory = response.get("gpu_mem")
```

`run_gc_collect=False`, missing-session closes, and requests that cancel a
still-loading session return no memory snapshot.

## Release assurance

The release contract pins source and checkpoint revisions, image paths and sizes,
prompts, resolutions, precision, thresholds, and comparison algorithms.

The hardened release process includes:

1. a self-contained pinned official CPU oracle;
2. raw official and MLX masks, boxes, and scores in compressed NPZ bundles;
3. deterministic Hungarian maximum-IoU object assignment;
4. exact 1,400-tensor checkpoint lineage with an embedded conversion manifest;
5. an independent auditor that replays every reported metric;
6. matching recorded source revisions;
7. wheel and source-distribution content checks.

See [`verification.md`](https://github.com/MedMLX/sam3-mlx/blob/main/docs/verification.md) for commands and [`technical.md`](technical.md)
for the complete system and risk map.

## Batched geometric image prompts

`set_image_batch` accepts one image per batch entry. Existing `add_geometric_prompt`
and `add_point_prompt` append the same normalized prompt to each image. For different
prompt counts or coordinates per image, use `add_geometric_prompt_batch` with
normalized cxcywh boxes or `add_point_prompt_batch` with normalized xy points:

```python
state = processor.set_image_batch([first_image, second_image])
processor.add_geometric_prompt_batch(
    [[[0.5, 0.5, 0.4, 0.4]], []], [[True], []], state, run_grounding=False
)
processor.add_point_prompt_batch(
    [[[0.3, 0.4]], [[0.6, 0.7], [0.2, 0.3]]],
    [[True], [True, False]], state, run_grounding=False,
)
processor.run_grounding(state)
```

Empty entries add no geometry to that image. Ragged entries become the canonical
`Prompt` sequence-first embeddings and per-image padding masks; result arrays
retain each image's original resolution.

The attached SAM3 tracker supports `add_tracker_new_points(...,
use_prev_mem_frame=True)`. Refinement uses the object's prior conditioning and
tracked memory, then rebuilds canonical conditioning state so later propagation
recomputes affected outputs. The default remains `False`.

Standalone `experimental.build_tracker(with_backbone=True, checkpoint_path=...)`
loads a converted checkpoint containing both official detector and tracker keys.
For separate artifacts, also supply `backbone_checkpoint_path` with the converted
image checkpoint. Both inputs undergo the canonical key/layout mapping and strict
coverage audit; missing backbone weights raise before parameters are modified.

`geo_encoder_use_img_cross_attn=False` accepts the same geometry graph as `True`,
matching the pinned upstream builder, where this argument is unused. A separate
geometry graph was rejected because it would invent different inference semantics
and checkpoint keys.

The video builder also forwards `strict_state_dict_loading=False` to the existing
image loader and supports `apply_temporal_disambiguation=False` with the official
ablation policy: zero hotstart delay, zero hotstart removal thresholds, and no
periodic detector reconditioning. Automatically attached trackers use the matching
memory-selection setting.

Detector-backed video sessions apply the official joint presence score exactly
once. A native image graph that returns class-only logits is projected to video
joint logits using the same canonical formula as the joint image decoder; a graph
that already returns joint logits is consumed directly. This projection preserves
the supplied image model's scoring configuration and includes the upstream
inverse-sigmoid and logit clamps.

Video builders accept `has_presence_token=False` for the pinned upstream's
class-only detector scoring. The official decoder graph and checkpoint keys
stay the same. An injected image model must expose uncombined class logits
(`supervise_joint_box_scores=False`) for this option.

Compressed OpenCV videos accept `async_loading_frames=True`. The loader
prefetches host RGB frames into an eight-frame cache, normalizes selected frames
on the inference caller, and stops its worker and releases the decoder on
session close. Multiplex demo initialization uses indexed normalized frames
rather than stacking the video. Direct loader callers must call `close()`;
explicit access to the compatibility `frames` or `images` properties materializes
the video. TorchCodec/CUDA decoding remains outside the native MLX backend.

Use the native builders documented here for video inference. Predictors inherit
session lifecycle behavior directly from `Sam3BasePredictor`; `Sam3VideoBase`
provides configuration, association helpers, and evaluator export for multiplex
models. Unsupported orchestration and Hydra construction shells have been
removed.

Request dictionaries retain their documented keys and defaults. Typed callers
can import payload schemas such as `AddMaskRequest`, `AddPromptRequest`, and
`PropagateInVideoRequest` from `sam3_mlx.model.sam3_base_predictor`. Multiplex
initialization, propagation, and grounding methods expose named parameters;
image constructors accept `multimask_output` for the multimask setting.
