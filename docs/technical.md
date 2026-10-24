# sam3-mlx technical notes

## Current product boundary

The stable, release-scoped surface of `sam3_mlx` is the SAM 3 image runtime plus
a selected-frame request API: image model construction, pinned checkpoint
loading, text and geometric prompting, and frame-local inference. The package is
research software; see the evidence limits below.

SAM 3.1 multiplex and temporal tracking remain experimental. They are not
covered by the image-runtime parity claim.

Runtime source distributions include the package, legal files, README and user
docs. Development Make targets, scripts, tests and qualification evidence are
maintained in the [public Git checkout](https://github.com/MedMLX/sam3-mlx).
The qualification sections below describe that checkout workflow.

Historical image parity is retained in Git as two schema-v2 reports, raw replayable
NPZ outputs, and checkpoint lineage. Each records the source and checkpoint
revisions for its measured run; later runtime edits need new parity evidence.

## Package map

### Stable API

- `build_sam3_image_model`
- `build_sam3_predictor`
- `build_sam3_video_model`
- `build_sam3_video_predictor`
- `download_ckpt_from_hf`
- `Sam3MlxUnsupportedError`

Top-level imports are lazy. Tracker and multiplex builders live under
`sam3_mlx.experimental` and are excluded from stable `__all__`; deprecated
top-level aliases remain available for source compatibility.

Repository paths such as `agent`, `eval`, and Triton-specific code are
source references or future-port material. Artifact validation verifies that
these paths do not enter the wheel.

`sam3_mlx.medmlx` is a MedMLX `medmlx.models` entry point consumer of the
stable API above (image segmentation only); it is not itself part of the
stable API and adds no new capability.

## Runtime data flow

### Image inference

```text
path / PIL / NumPy image
  -> RGB conversion
  -> TorchVision-matched FP32 resize and normalization
  -> one visual-boundary cast (`fp32` / `fp16` / `bf16` / `mixed`)
  -> MLX vision backbone
  -> image feature state

Caller-normalized NCHW/NHWC tensor
  -> `set_preprocessed_image` (skips PIL)
  -> same visual-boundary cast
  -> MLX vision backbone

text prompt
  -> tokenizer and language backbone
  -> language features

box / point prompt
  -> normalized geometry
  -> geometry encoder

image + language + geometry
  -> `predict_raw` (fixed-shape logits, boxes, low-res masks, presence)
  -> `predict` convenience wrapper
  -> confidence filtering
  -> interpolation to original image size
```

`Sam3Processor` owns mutable per-image state. Processor resolutions may be any
positive multiple of 14. Exact window-aligned staging sizes are 336, 672, and
1008. Release parity is measured only at 1008, 672, and 504. 336 is an optional
fast tier, not an accuracy-equivalent default.

### Checkpoint path

```text
builder
  -> model component construction
  -> checkpoint normalization
  -> key / shape / coverage audit
  -> model mutation
  -> MLX evaluation mode
```

The default image checkpoint is pinned by immutable Hugging Face revision.
Official PyTorch downloads default to a canonical immutable revision;
explicit overrides require a full commit SHA. Conversion
records source revision, source and output file names, key counts, ignored
tracker keys, and dtype counts.

The package default is the FP32 release artifact. FP16, BF16, and mixed policies
must use a local conversion or a separately pinned Hub artifact. Reduced-
precision Hub artifacts require a conversion manifest whose floating dtype
inventory exactly matches the requested policy; loading the default FP32
artifact and casting it after model mutation is rejected.

`Sam3Processor(preprocessing="parity")` retains the TorchVision-matched release
path. `preprocessing="fast"` uses MLX antialiased resize and records a distinct
benchmark runtime policy. The fast path has a pixel-level bound but remains
unattested for segmentation outputs.

### Selected-frame video

```text
image / folder / PIL sequence / OpenCV video
  -> bounded frame provider
  -> session state
  -> frame-local image inference
  -> streamed detections
```

This is not temporal tracking. It does not maintain persistent identities or
tracker memory. `out_obj_ids` are frame-local. Cross-frame object removal is
rejected.

Image folders use a bounded on-demand RGB cache. Video files use a bounded
OpenCV cache. Encoded frame features and text features are bounded per session.
Full-resolution output history is not retained.

The legacy asynchronous folder loader eagerly retains every decoded frame. The
hardened public predictor rejects that configuration rather than violating the
bounded-memory claim.

## Concurrency and lifecycle

`Sam3BasePredictor` is the lifecycle authority for every public and future
subclass. It uses a global session-registry lock plus a reentrant lock per
session. Session state includes a version, propagation flag, cancellation event,
and closing/closed flags. `LifecycleSafeSam3BasePredictor` remains only as a
zero-logic compatibility subclass.

The hardened base enforces:

- `shutdown()` is terminal and prevents future sessions;
- an in-flight loader cannot publish after shutdown;
- `close_session(id)` cancels publication when that ID is still loading;
- a state that finishes loading after close or shutdown is disposed;
- reservation IDs cannot be cleared and reused to overwrite a live state;
- disposal clears state and marks the session closed even if a frame provider's
  `close()` raises;
- diagnostic session snapshots do not iterate a mutating registry unsafely.

Closing an existing session with garbage collection enabled returns MLX allocator
telemetry. Reserved bytes include active and cached allocations; reported free
bytes describe headroom against Metal's recommended working-set budget, not
system-wide free memory. At the default 80% pressure threshold, close clears the
MLX cache and reports a fresh snapshot. Missing sessions and
`run_gc_collect=False` closes do not query allocator memory.

Shared position-encoding, RoPE, and decoder-coordinate LRUs are synchronized.
This is required because concurrent sessions share one model.

## Experimental SAM3 mask sessions

The SAM3 video builder accepts an explicit tracker model or tracker checkpoint.
A checkpoint attachment shares the loaded image backbone's SAM2 neck and loads
the complete SAM3 tracker through the canonical strict loader. The default
selected-frame path remains unchanged.

`MaskTrackingSession` owns typed object states, mask and point conditioning links,
semantic prompts, association decisions, action history, hotstart state,
cached frame predictions, bounded shared features, phase, and generation.
Each object has a singleton tracker state so adding an object after another
object's memory preflight remains legal. Refinement constructs a fresh state
and replays that object's conditioning masks, invalidating stale downstream
memory. Request actions reject mutation during propagation; direct generators
discard results when their generation changes.

With an attached tracker, text or box detections seed persistent objects that
can then be refined with masks or points. Point refinement requires `obj_id`
and excludes text or semantic boxes in the same call. A fresh semantic prompt
replaces the tracking session.

The canonical association boundary applies score filtering, mask NMS,
many-to-many matching, capacity allocation, keep-alive and hotstart rules.
Propagation actions choose full detector/tracker execution, partial instance
updates, or cached results. Memory encoding applies overlap constraints across
objects. Point refinement consumes previous same-frame logits and removes
nearby mask-only conditioning; stale downstream memory is invalidated.

Tracker states remain separate per object to support late additions and isolated
conditioning replay. Upstream groups same-frame detections in a shared state;
that batching was deferred because it requires a joint membership, removal,
and refinement cutover. State-wide memory selection can affect neural outputs,
so this attachment does not claim the original batching behavior or numerical
temporal equivalence.

Manual seeds and refinements export confidence `1.0`; semantic detections retain
their detector confidence. Tracker probability resolves overlaps. Synthetic
session tests and pinned-source CPU association comparisons do not qualify
real-weight temporal equivalence or long-sequence memory use.

## Experimental SAM 3.1 path

The source tree includes multiplex detector/tracker composition, memory encoding,
mask decoding, checkpoint-key normalization, and a predictor wrapper. Builders
can acquire the pinned official SAM3.1 checkpoint, convert its raw tensors to
MLX weights, and reuse a bundle validated against pinned source metadata,
tensor counts, and index byte accounting.
An explicit checkpoint path bypasses acquisition; an invalid cached bundle raises
without being replaced. Initial conversion requires PyTorch.

Synthetic checkpoint tests cover acquisition routing, tensor transport, strict
loading, and cache rejection. They do not qualify real-checkpoint outputs. The
runtime still lacks sequence-level official-vs-MLX evidence, identity-transition
parity, long-video memory characterization, and complete multi-object validation.
Public multiplex builders default to `use_rope_real=False`; the unsupported
real-arithmetic RoPE path stays fail-fast.

A temporal release belongs to 0.2 and needs its own frozen contract.

The multiplex demo initializer preserves its upstream dimension-first signature
for inspection and also accepts resource-first calls through the base interface.
Reordering the signature was rejected because it breaks upstream API inspection;
suppressing the override error was rejected because it hides incompatible calls.

## Release evidence architecture

### Frozen contract

`sam3_mlx.release_contract` pins:

- package version;
- official source commit;
- official and MLX checkpoint revisions;
- calibration and holdout image paths and sizes;
- exact prompt matrix;
- confidence threshold and resolutions;
- metric thresholds;
- CPU oracle precision and allowed adaptations;
- schema and comparison algorithm versions.

Changing any item requires new evidence.

### Committed-source gate

`sam3_mlx.source_binding` requires all source changes to be committed before
evidence generation. Dirty paths are allowed only under:

- `parity/receipts/`
- `parity/manifests/`
- `parity/evidence/`
- `benchmarks/baselines/`

`README.md`, `docs/technical.md` and `docs/verification.md` are also allowed
by this gate.

The parity reports, raw NPZ metadata, and checkpoint-lineage report each
record the same source commit. `audit_release_evidence.py` rejects mixtures
of results from different source revisions.

### Hardened oracle

The official CPU oracle verifies the checkout, submodules, checkpoint, image,
prompt matrix, threshold, and actual import source paths. The cache identity
includes the code and checkpoint revisions, image, prompt matrix, threshold,
precision, and CPU adapters. The checkout is checked before and after the run.

### Replayable parity

The hardened parity runner stores raw official and MLX masks, boxes, and scores
in compressed NPZ files. Detections are paired using deterministic Hungarian
maximum-mask-IoU assignment rather than greedy pairing. Pairwise IoU computation
uses bounded scratch memory instead of materializing an `N x N x pixels` tensor.

The report references the raw bundle by path. Summary metrics are therefore
recomputable instead of merely asserted.

### Hardened lineage

The lineage gate hard-pins source and published checkpoints, embeds the complete
conversion manifest, records the source commit, and requires exact equality of
1,400 canonical tensor keys, shapes, dtypes, and values.

### Independent replay

`audit_release_evidence.py` reads the two parity reports and checkpoint lineage
directly. It verifies the pinned revisions, loads NPZ outputs with pickle disabled,
recomputes assignment and all count, IoU, box, and score metrics, and checks the
embedded conversion manifest plus recorded 1,400-tensor lineage. It also requires
reports, raw outputs, and lineage to name the same source revision.

`make release-evidence-audit` replays those measurements. `make release-check`
also runs the functional suite, numerical oracle checks, and package checks.

## Validation matrix

| Layer | Gate |
| --- | --- |
| Package | clean-wheel install, stable exports, excluded source paths |
| Preprocessing | TorchVision numerical comparison |
| Checkpoint | key, shape, and required-component coverage |
| Checkpoint source | immutable revisions |
| Source binding | one committed source revision across all artifacts |
| Lineage | 1,400 exact canonical tensors |
| Image outputs | replayed official-vs-MLX evidence at 1008/672/504 |
| Controls | positive text, negative text, positive box, positive/negative box |
| Performance | synchronized latency and active-memory samples |
| Video API | resource, session, cancellation, and bounded-cache contracts |
| Concurrency | cache stress, close/loading, and shutdown/publication tests |
| Artifacts | wheel and sdist content inspection |

## Issues addressed during release hardening

- Unsynchronized shared LRUs.
- Session publication after shutdown.
- Session publication after close while loading.
- Duplicate-ID risk after reservation clearing.
- Incomplete cleanup after frame-provider close failure.
- Racy live-session diagnostics.
- Unbounded asynchronous folder loading in selected-frame mode.
- Oracle imports not proven to originate from the pinned checkout.
- Greedy object pairing.
- Unbounded pairwise-IoU scratch memory.
- Summary-only parity evidence.
- Caller-selected lineage revisions.
- Evidence files that were internally valid but generated from different source
  commits.

## Release gate status

The retained reports cover their recorded source revision. Runtime changes
require fresh official parity before an accuracy claim.

Creating a version tag or publishing an artifact is a separate distribution
action and is not implied by the checked-in evidence.

## Remaining product risks

- Two pinned images prove implementation parity, not broad segmentation quality.
- No external task-diverse validation corpus is part of the release gate.
- Selected-frame inference can be mistaken for tracking if documentation drifts.
- Random-access OpenCV seeking may be slow on long videos.
- True SAM 3.1 temporal behavior is unvalidated.
- `model_builder.py` is a large maintainability hotspot.
- Reduced-precision and fast-preprocessing output parity is not yet attested.
- Performance evidence represents one Apple Silicon host.

## Highest-leverage next work

1. Add a broader external image corpus without weakening the frozen holdout.
2. Run output-level parity and multi-host benchmarks for fast preprocessing.
3. Add a sequential video decode path for forward propagation.
4. Split checkpoint normalization and component factories out of
   `model_builder.py` without changing the stable release contract.
5. Treat temporal tracking as a separate 0.2 assurance program.

## Development checks

`make qa` runs lint and strict Pyright in the existing environment. It does not
install extras, run tests or execute model oracles. The type gate covers
`sam3_mlx`, scripts, tests and experiments; environments and vendored upstream
code are excluded. Use `make test TESTS=tests/path.py` for affected behavior,
`make test` for the full suite, and `make parity-qa` for the explicit oracle run.
Prepare that run with `uv sync --locked --group oracle`. The lock pins Torch
2.13.0 and TorchVision 0.28.0. The shared core comes from its immutable Git pin.
Set `ROI_ALIGN_ORACLE_RUN=path/to/python` to use a separate provisioned oracle
interpreter; the default is the project Python selected by `PYTHON`.

Ruff rejects explicit `Any` imports throughout the checked source tree. Request
payloads use tagged schemas, and dynamic oracle imports use typed protocols.
Strict Pyright remains the required type gate.
