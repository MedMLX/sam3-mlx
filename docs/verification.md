# Parity and release evidence

All commands on this page require the public Git checkout. Runtime source
distributions omit this workflow, its scripts, tests, benchmarks and evidence.

## Scope

The 0.1.x parity claim is limited to the SAM 3 image runtime. It covers the
frozen image, prompt, resolution, checkpoint, precision, and threshold matrix in
`sam3_mlx.release_contract`.

It does not claim:

- qualified SAM 3.1 temporal equivalence or persistent cross-frame identity;
- training or official evaluation tooling;
- unmeasured processor resolutions;
- broad task-level segmentation quality.

The repository includes an experimental SAM 3.1 temporal/multiplex runtime,
a versioned comparison harness, and long-sequence characterization tooling.
An earlier Apple-only receipt did not run the official CUDA predictor. A later
[characterization](../parity/receipts/sam31-expanded-temporal-20261026.json)
compared official CUDA BF16 with MLX FP32 over seven sequences: three passed the
quality gate, and the bedroom masks diverged after correction. The diagnostic
corpus is not an independent holdout or a temporal release qualification.

## Inference API target

The inference API inventory in `sam3_mlx.api_contract` targets the separate,
clean official checkout at `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`.
This does not change the frozen source or weight pins for historical numerical
release evidence. New unsupported mask and propagation controls name the API
target revision; existing unsupported surfaces retain their original source records.

Run the source-based declaration audit:

```bash
uv run --no-sync python scripts/audit_inference_api.py \
  --official-checkout ../sam3 \
  --out benchmarks/results/inference-api-current.json \
  --check
```

The audit covers selected inference builders, image processors, interactive
wrappers, request actions, and multiplex entry points. It compares names,
parameter kinds, positional order, and default expressions without importing
the official runtime. It records source revisions and the dirty local state.
Additional optional port arguments are allowed unless they consume upstream
variadic positional arguments. Backend default adaptations are explicit value
pairs. A signature match does not establish working options, output-schema
parity, numerical equivalence, or complete repository feature coverage.

A historical declaration report at `benchmarks/results/inference-api-current.json`
covered 40 surfaces: 31 signature matches, four recorded backend differences,
and five API gaps. That local report may not be present in a checkout; rerun the
audit to determine the current gaps. This inventory is separate from the runtime
qualification results below.

The current implementation preserves these deliberate differences:

- Predictor direct calls inherit the configured confidence threshold when no
  override is supplied. Switching their defaults to upstream's fixed `0.5`
  was rejected because it changes an existing tested configuration contract.
- Official checkpoint downloads default to the version's canonical immutable
  revision. Explicit overrides also require commit SHAs. Using an unpinned Hub
  default was rejected because it obscures the checkpoint's source.
- Video builders retain bounded synchronous loading defaults. Adopting eager
  asynchronous loading without a bounded implementation was rejected because
  it violates the public memory contract.
- The stable default remains SAM3 frame-local inference until SAM3.1 temporal
  behavior is qualified. Switching the version string after qualifying only
  checkpoint setup was rejected because loading does not establish tracking fidelity.
- The multiplex demo honors an explicit `add_mask_to_memory=True` as a local
  extension; its default is now upstream's `False`. Dropping the explicit
  request was rejected because callers deliberately request memory encoding.

Temporal numerical qualification remains a runtime gap. With an explicit SAM3
tracker attachment, semantic detections seed persistent objects that can then
be refined with masks or points. Pinned-source CPU tests cover synthetic
association decisions and hotstart transitions; offline session tests use
scripted detector outputs and real MLX memory kernels. These checks do not
establish real-weight temporal parity or release qualification.

Automatic SAM3.1 setup uses one packaged raw converter and verified bundle
parser; cache reuse checks pinned source metadata, tensor counts and index byte
accounting before strict loading.
Setup tests use tiny synthetic weights; they do not requalify the real checkpoint or
establish temporal parity. The full automatic-setup qualification below also
exercises the pinned real bundle. Session close implements the upstream cache threshold
argument using explicit MLX process allocator headroom against Metal's
recommended budget. These metrics describe the process allocator rather than
CUDA or system-wide free memory. SAM3 mask sessions now attach an explicit loaded
tracker; this does not promote the selected-frame default or qualify the original
video runtime. Multi-GPU
distribution, Flash Attention, and compilation need explicit MLX policies rather
than signatures that silently ignore them.

## Interactive head qualification

`scripts/run_interactive_head_parity.py` compares the pinned official Torch
interactive predictor and MLX predictor on shared synthetic image features and
149 shared head tensors. It exercises multiple object prompt groups, points,
boxes, mask input, single-mask and multimask logits, and images of different
sizes. The output has an explicit schema version, source revisions, file sizes
and modification times, adapters, runtime devices, seed, tolerances, and
per-output numerical verdicts.

```bash
uv run --no-sync --group oracle --with einops python \
  scripts/run_interactive_head_parity.py \
  --official-checkout ../sam3 \
  --official-revision 0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc \
  --seed 731 \
  --out benchmarks/results/interactive-head-parity.json
```

The annotation-only tracker import is omitted explicitly and recorded in the
report so the CPU oracle can load the predictor without the unrelated video
stack. Runtime uses of that omitted type fail. This qualification does not
cover the image backbone, released checkpoint accuracy, mask cleanup,
temporal tracking, or performance, and does not replace the frozen image parity
reports.

The semantic checkpoint-session probe uses the pinned official image notebook's
`assets/images/test_image.jpg` and `shoe` prompt. This is a positive execution
fixture chosen from the official example before native outputs were measured.
The earlier `person` prompt produced zero detections in both clean official
source revisions at the unchanged 0.5 threshold: its presence-weighted maximum
score was about 0.245 under CPU BF16. Native FP32 likewise produced zero detections
(maximum about 0.233). That negative observation does not establish a port defect
or tracking parity. The source-matched official `shoe` example produced 12
scores above both the 0.5 detection and 0.7 new-object thresholds. Session and
association replay remain engineering checks of the captured workload.

## Full-checkpoint interactive qualification

`scripts/run_interactive_checkpoint_parity.py` runs the full `1008` image
backbone on the pinned example and independent holdout images. Each image
exercises points, boxes, mixed prompts, multiple object prompt groups, mask
refinement, and mask-only input. A heterogeneous image batch adds two cases,
for 14 cases in total. Hole cleanup is enabled and a skipped upstream cleanup
operation fails the run.

```bash
uv run --no-sync --group oracle --with einops python \
  scripts/run_interactive_checkpoint_parity.py \
  --official-checkout ../sam3 \
  --official-revision 0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc \
  --official-checkpoint checkpoints/official/facebook-sam3/sam3.pt \
  --mlx-checkpoint checkpoints/mlx-community/sam3-image/model.safetensors \
  --tracker-checkpoint checkpoints/derived/sam3-tracker-raw-fp32/model.safetensors \
  --out benchmarks/results/interactive-checkpoint-parity-final
```

Prepare the raw tracker artifact with
`scripts/convert_sam3_tracker_checkpoint.py`, as described in
[`usage.md`](usage.md). This preserves the complete tracker and lets
the image loader select the 149 image-head tensors while accounting for the
160 video-only tensors. The CPU reference uses the same scoped tokenizer
resource locator and explicit FP32-resize/uint8-round policy described below.
This interactive lane runs FP32 with autocast disabled, including its existing
three-cast CPU fused-operator adapter. The versioned text/box image reference
now shares that FP32 adaptation; neither lane qualifies default BF16 or CUDA
execution.

The Torch CPU and MLX Metal phases run in separate processes. Reports bind
package and official source revisions, source file sizes and modification times,
checkpoint revisions, image paths and sizes, case specifications, runtime devices,
and raw NPZ outputs. Source file sizes and modification times are rechecked during
execution; comparison rejects mismatched bindings or incomplete output inventories.
Failed attempts retain partial evidence.

The verdict uses the existing mask-IoU and score thresholds below. Raw logits
are retained and their errors reported separately; passing masks and scores
does not qualify raw-logit equivalence. The CPU oracle records any adaptation
needed by the original runtime and does not establish unmodified CUDA
equivalence. This development report does not replace the historical release
reports or broader validation on representative images.

The historical local report at `benchmarks/results/interactive-checkpoint-parity-final/report.json`
passed all 14 cases with matching 42-array inventories on both backends. The
minimum object mask IoU was `0.999974846`, the minimum case mean was
`0.999991615`, and the maximum score error was `0.000007034`. Both predictors
used hole area `256`, sprinkle area `0`, and mask threshold `0`; upstream
cleanup executed. Package sources, scripts, and inputs remained stable.
The Torch CPU oracle explicitly changes three fused-operation BF16 casts to
FP32. Its source-checked adapter is recorded in the report. Raw-logit errors
remain diagnostic and CUDA equivalence is unmeasured.

## Mask-session and SAM3.1 checkpoint qualification

`scripts/run_mask_checkpoint_session.py` exercises a loaded SAM3 tracker through
the public mask request and propagation API. Its report binds the source,
image weights, raw tracker weights, inputs, encoded conditioning memory, and
frame outputs. It establishes execution of this opt-in path; it does not
compare temporal outputs against the official CUDA runtime.

```bash
uv run --no-sync --group oracle --with einops python \
  scripts/run_mask_checkpoint_session.py \
  --official-checkout ../sam3 \
  --image-checkpoint checkpoints/mlx-community/sam3-image/model.safetensors \
  --tracker-checkpoint checkpoints/derived/sam3-tracker-raw-fp32/model.safetensors \
  --out benchmarks/results/mask-checkpoint-session-qualified-v4
```

The local report at `benchmarks/results/mask-checkpoint-session-qualified-v4/report.json`
passed strict loading of 1,400 image tensors, 149 interactive-head tensors,
and all 309 tracker tensors, with no required missing weights or incompatible
shapes. It preserved the exact seed mask and object ID, decoded the next-frame
mask, and evaluated finite conditioning and propagated memory of shape
`[1, 64, 72, 72]`. Removal, reset, and close completed. This is a two-frame
execution fixture; the lifecycle tests additionally exercise refinement, late
objects, reverse propagation, eviction, cancellation, and both memory-selection
settings.

The same runner's `--mode semantic` exercises actual detector-to-tracker
execution, mixed instance refinement, and visual prompting:

```bash
uv run --no-sync --group oracle python scripts/run_mask_checkpoint_session.py \
  --official-checkout ../sam3 \
  --image-checkpoint checkpoints/mlx-community/sam3-image/model.safetensors \
  --tracker-checkpoint checkpoints/derived/sam3-tracker-raw-fp32/model.safetensors \
  --mode semantic \
  --out benchmarks/results/semantic-checkpoint-session-final-v3
```

The semantic session must retain detector-seeded identities with nonempty masks
across all three frames. It then exercises mask and point refinement, removal
of nearby mask-only conditioning, partial propagation, exact cached results,
visual-box prompting, object removal, semantic reset, and close with MLX memory
telemetry. Other objects must retain their tracker states during refinement.
The saved artifact includes actual detector and tracker inputs for all three
association decisions, bound to the executed source and strict weight audits.

The earlier six-object `person` receipt is historical and does not qualify the
current presence-weighted score path. Its current-source replay fails before
association because the pinned official model also returns no detections for
that prompt. Current qualification uses the independently documented `shoe`
example above; detection and association thresholds remain unchanged.

Checkpoint-derived association comparison replays the captured inputs through
unmodified, source-pinned upstream CPU association routines. Every matched,
reconditioning, allocation and keep-alive decision must agree exactly across
three frames. Synthetic policy comparisons additionally cover explicit
association cases and forward/reverse hotstart transitions. Saved-input
association replay does not compare NMS or upstream neural outputs.

The attachment uses separate tracker states per object. Upstream batches
same-frame detections, and state-wide memory selection can change neural
outputs. This execution probe does not qualify that batching behavior,
long-sequence memory use, or official CUDA temporal numerical equivalence.

`scripts/qualify_sam31_checkpoint.py` qualifies a bundle prepared by
`scripts/convert_sam31_checkpoint.py`. Its gates verify the canonical original
checkpoint revision, conversion manifest, exact raw tensor transport, strict full
model key/shape coverage, finite loaded parameters, and evaluated detector-text
and tracker-memory forwards on Metal. It rechecks source and input file sizes
and modification times at the end. Regenerated RoPE buffers are accounted for explicitly.
Neither passing these gates nor passing a shape-only audit establishes
SAM3.1 sequence-level equivalence or permits a default-version change.

```bash
uv run --no-sync --group oracle python scripts/convert_sam31_checkpoint.py \
  --source checkpoints/official/facebook-sam31/sam3.1_multiplex.pt \
  --output-dir checkpoints/derived/sam31-raw-fp32
uv run --no-sync --group oracle python scripts/qualify_sam31_checkpoint.py \
  --source checkpoints/official/facebook-sam31/sam3.1_multiplex.pt \
  --bundle checkpoints/derived/sam31-raw-fp32 \
  --automatic-setup \
  --output benchmarks/results/sam31-automatic-setup-qualification-final-v2.json
```

The local SAM3.1 automatic-setup report at
`benchmarks/results/sam31-automatic-setup-qualification-final-v2.json`
passed exact key, shape, dtype, and value comparisons for all 1,591 retained
official FP32 tensors. The strict model load covered 1,835 target parameters,
with no required missing weights, extras, or incompatible shapes. The 32
complex RoPE buffers are regenerated, as is the language attention mask.
Detector text encoding and tracker memory encoding executed on Metal with
finite outputs. Source and checkpoint revisions remained stable during both
qualification runs. These local development reports describe their recorded
source revisions and file signatures; they do not attest subsequent edits,
the final commit, or a release.

A historical combined local run passed 1,700 tests with the source-pinned association
oracle and checkpoint capture enabled, plus Ruff and strict Pyright (`make qa`
itself runs only the two static checks), with five test warnings. The retained
historical image evidence replays with `make release-evidence-audit`; it
describes its recorded source, not these later runtime changes. Fresh release
evidence must bind the final committed source before publication. The artifact
paths above identify this run; choose fresh output directories for a new run.

## Frozen numerical contract

Every non-empty case requires:

- exact detection count;
- per-object mask IoU at least `0.95`;
- case mean mask IoU at least `0.99`;
- box-coordinate L-infinity error at most `2.0` pixels;
- score absolute error at most `0.025`.

The confidence threshold is `0.5`. Measured resolutions are `1008`, `672`, and
`504`.

The calibration and holdout images, prompt cases, official source revision,
official checkpoint, published MLX checkpoint, CPU precision, and allowed CPU
adaptations are specified in `sam3_mlx.release_contract`.

## Evidence layers

### 1. Preprocessing

Synthetic RGB inputs are compared against TorchVision preprocessing across
multiple aspect ratios. Release preprocessing tests must run with TorchVision
available:

```bash
make preprocess-parity-check
```

### 2. Checkpoint lineage

A fresh conversion from the pinned official checkpoint is compared with the
pinned published MLX artifact after canonical runtime-layout normalization. A
passing report requires exactly 1,400 matching tensor keys, shapes, dtypes, and
values.

The schema-v2 lineage report embeds the complete conversion manifest and records
the generating source commit. It distinguishes the historical published artifact's
converter version (`0.1.2`) from the freshly executed reproduction's current
package version (`0.1.3`). A current reproduction must identify its actual
converter version; historical metadata is never reused to label a fresh run.

### 3. Upstream oracle

The hardened official oracle verifies the official checkout before and after
inference and binds its cache to:

- official code revision and clean status;
- official checkpoint revision;
- image name;
- case-spec name;
- confidence threshold;
- the `pinned-source-cpu-fp32-v1` reference profile and native FP32 policy;
- the `addmm-act-three-cast-fp32-v1` adapter, its exact source owner and rewrite;
- the explicitly allowed CPU adaptations;
- actual import paths inside the pinned official checkout.

The CPU reference locates the legacy tokenizer resource through
`importlib.resources` and verifies its exact bytes against the pinned Git blob.
It explicitly selects FP32 bilinear resize followed by uint8 rounding, preserving
the TorchVision 0.24 input contract under patched TorchVision 0.28. Floating model parameters
remain FP32 and CPU autocast is disabled. The source-checked fused adapter changes
exactly the three `self`, `mat1` and `mat2` casts in official `addmm_act` from
BF16 to FP32; its matmul, GELU, arguments and weights remain unchanged. The adapter
lives in `scripts/_oracle_runtime.py`, which is part of the cache file-stat
identity and the report's committed source identity. Execution checks actual
parameter and output dtypes before recording them. The checkpoint additionally
retains 32 complex64 rotary buffers alongside its 1,368 float32 tensors.

This is FP32 parity with a precision-adapted pinned official source, not
compatibility with its default BF16 or CUDA execution. The former schema-4 BF16
reference failed `text_shoe_504`: minimum mask IoU `0.8716707022` and mean
`0.9860993842`, below the unchanged `0.95` / `0.99` gates. Ten of eleven cases
passed. That failure remains unqualified; same-input diagnostics localized the
discrepancy to BF16 forward arithmetic rather than preprocessing or final mask
projection. The original schema-4 reports, raw arrays and lineage are retained
byte-for-byte in [the BF16 snapshot](../parity/receipts/bf16-v4-source-A2bd9/).
Paths inside that nested snapshot are relative to its own root, and its source
identity remains `2bd9b7a135f3f14b1f9e80b21f332851d90af337`.

The FP32 contract keeps the same two images, eleven cases, prompts, resolutions,
checkpoint/source pins, exact detection-count requirement and numerical thresholds.
It requires newly executed canonical evidence; private diagnostic comparisons are
not relabelled as release proof.

The current direct-source reference environment uses Python 3.13, Torch 2.13.0,
TorchVision 0.28.0, NumPy 2.5.4, Pillow 13.0.0 and ftfy 6.3.1. This is an explicit
metadata exception to the frozen upstream distribution constraints (`numpy<2`,
`ftfy==6.1.1`). The pinned source, exact tokenizer bytes and mandated pixel
preprocessing are checked independently; numerical evidence applies to this
recorded environment.

Public reports explicitly mark a `public-report` metadata projection. They omit
only the cache lookup identity containing private absolute filesystem paths;
full transport metadata stays in the private oracle NPZ cache. Source, checkpoint,
image, prompt, configuration and precision bindings remain unchanged, as do the
actual outputs and numerical comparison data.

A cache missing any binding is not release evidence. Both image parity runners
require oracle cache schema 5; older caches must be recomputed with the hardened
official oracle.

### 4. Raw output parity

The hardened parity runner stores official and MLX masks, boxes, and scores for
every case in compressed NPZ bundles. Object pairing uses deterministic Hungarian
maximum-mask-IoU assignment.

Schema-v3 reports reference the raw bundle by path. Summary metrics are
recomputed by the release auditor rather than trusted directly. Both the report
and raw bundle record the exact `sam3_mlx` source commit that generated them.

### 5. Recorded source revision

The reports, raw NPZ metadata, and checkpoint-lineage report record the same
source commit. Direct replay rejects mixed revisions. Test results are shown
by pytest when the suite runs; no aggregate receipt or generated documentation
copy is required.

## Generate hardened evidence

Commit all source changes first. Evidence generators reject non-evidence
worktree changes. Use a clean official checkout at the revision recorded in
`sam3_mlx.release_contract`.

### Calibration profile

```bash
uv run python scripts/run_hardened_image_parity.py \
  --official-checkout /path/to/pinned/sam3 \
  --official-checkpoint /path/to/sam3.pt \
  --official-python /path/to/oracle-python \
  --mlx-checkpoint /path/to/model.safetensors \
  --image /path/to/pinned/sam3/assets/images/test_image.jpg \
  --profile example \
  --out parity/receipts/example-image-parity.json \
  --evidence-out parity/evidence/example-image-parity.npz
```

### Independent holdout profile

```bash
uv run python scripts/run_hardened_image_parity.py \
  --official-checkout /path/to/pinned/sam3 \
  --official-checkpoint /path/to/sam3.pt \
  --official-python /path/to/oracle-python \
  --mlx-checkpoint /path/to/model.safetensors \
  --image /path/to/pinned/sam3/assets/images/groceries.jpg \
  --profile holdout \
  --out parity/receipts/holdout-image-parity.json \
  --evidence-out parity/evidence/holdout-image-parity.npz
```

### Checkpoint lineage

```bash
uv run python scripts/validate_checkpoint_lineage_hardened.py \
  --official-checkpoint /path/to/sam3.pt \
  --published-checkpoint /path/to/published/model.safetensors \
  --reproduced-checkpoint /path/to/reproduced/model.safetensors \
  --reproduction-manifest /path/to/reproduced/conversion-manifest.json \
  --out parity/manifests/checkpoint-lineage.json
```

Run the release suite with zero failures, skips, or deselections before making
a release claim. Retain the measured parity reports, raw outputs, and checkpoint
lineage for that source revision.

## Validate a candidate

```bash
make qa
make parity-qa
make release-evidence-audit
make artifact-check
make release-check
```

`release-evidence-audit` invokes `audit_release_evidence.py` directly. It loads
raw NPZ outputs with pickle disabled, replays every case, verifies cache and
lineage settings, and checks the recorded source revisions. It prints the result
without writing another report. `validate_release.py` builds and inspects the
wheel and source distribution.

## Precision and preprocessing candidates

The frozen image reports use FP32 with parity preprocessing. A later
[image characterization](../parity/receipts/image-bf16-20261027.json) passed 56
cases each for FP32 and BF16 against their official CUDA references. The FP32
reference uses a recorded precision adapter, and the cases were used to calibrate
the implementation rather than as an independent holdout. This characterization
does not replace the frozen reports or establish temporal parity.

Candidate FP16 artifacts and the fast preprocessing path use the legacy
comparison entrypoint only for local qualification; they do not alter the frozen
image reports.
The runner requires the candidate artifact's immutable identity and validates
its dtype manifest before loading:

```bash
uv run python scripts/run_image_parity.py \
  --official-checkout /path/to/pinned/sam3 \
  --official-checkpoint /path/to/sam3.pt \
  --official-python /path/to/oracle/python \
  --mlx-checkpoint /path/to/model.safetensors \
  --mlx-checkpoint-repo owner/sam3-image-fp16 \
  --mlx-checkpoint-revision <40-character-commit> \
  --image assets/images/test.jpg \
  --precision fp16 \
  --preprocessing parity \
  --out benchmarks/results/fp16-parity.json
```

Run both example and holdout profiles, positive and negative geometric cases,
and `scripts/benchmark_image_runtime.py --precision ...` for every candidate.
The parity runner fails on NaN/Inf boxes or scores; the benchmark records
latency, peak active memory, precision, preprocessing mode, and checkpoint revision.

## Current repository state

The independent replay covers 11 cases across the example and holdout profiles.
The historical 2026-09-25 run used source
`c0bbdeb3380b54f4c996c0884c49ad20f3c1f04d`, MLX `0.32.2`, Python `3.13.5`,
and FP32 parity preprocessing; its test suite recorded 1,163 passing tests with
zero skips or deselections. The reports retain the checkpoint revisions,
thresholds, latency samples, and memory measurements. Re-run official parity
after runtime changes land on the final committed source.

This worktree is **not** accuracy-attested, and no claim is made about how its
speed compares with other SAM 3 MLX implementations. Official release
resolutions remain `1008`, `672`, and `504`. `336` is an optional exact-window
staging tier; it is not accuracy-equivalent to `1008` until a representative
corpus shows otherwise.

## Current performance evidence

The source-bound Apple-Silicon run measured synchronized full `set_image` plus
text grounding with one warmup and five samples per resolution. Median ranges
across the example and holdout profiles were:

- `1008`: `1.61–1.68 s`;
- `672`: `0.70–0.83 s`;
- `504`: `0.49–0.56 s`;
- peak active MLX allocation: about `8.42 GB`.

These measurements describe one Apple-Silicon host and the source commit named
by the parity reports. They are not performance guarantees for later commits
and do not support a “fastest SAM 3 MLX” claim.

## Performance regression benchmark

The standalone benchmark reuses the synchronized release timing contract while
also measuring preprocessing, `set_image`, full image-plus-text inference, and
the same workflow with cached text features. Artifacts record raw samples,
p50/p95, peak active MLX memory, runtime policy, host identity, checkpoint revision,
and source commit. The benchmark enables the opt-in `compile=True` image path;
`--no-compile` records the eager path as a distinct runtime policy so unlike
artifacts cannot be compared.

Synchronized profiling on the Apple M1 Max showed that the visual backbone,
not preprocessing, dominates 504-pixel image setup. MLX compilation reduced
median `set_image` latency from 317 ms to 296 ms (6.8%) and full
image-plus-text latency from 369 ms to 344 ms (6.9%) over seven timed runs. This
is a host-specific result, not a cross-device performance claim. The checked-in
tests compare compiled and eager outputs only on a small stand-in backbone
(`tests/runtime/test_vl_combiner_compile.py`), so output equality of the
compiled full model is not established by the committed evidence.

Run the fixed synthetic workload on the local pinned checkpoint:

```bash
make benchmark
```

Compare it with the reviewed Apple M1 Max baseline. The gate rejects dirty
artifacts, different workloads or runtime environments, and any default metric
that regresses by more than 10% after a 3% noise band. Seven timed repetitions
reduce sensitivity to transient host scheduling noise:

```bash
make benchmark-regression-check
```

The committed baseline is host-specific. A different chip, MLX version, Python
version, workload, or synchronization policy requires a separately reviewed
baseline rather than bypassing the like-for-like check.

## Intentionally deferred or blocked

- investigate the observed SAM 3.1 bedroom correction mismatch and qualify
  temporal identity and numerical behavior on independent representative
  sequences using a compatible CUDA BF16 host and MLX;
- broader representative-image validation without changing the frozen
  thresholds;
- long-sequence runs above 300 frames under a separately reviewed execution cap;
- any default switch to FP16, BF16, `fast`, or `native` preprocessing.

## Known upstream behavioral notes

- Presence-logit clamp: the official decoder computes
  `intermediate_layer_presence_logits.clamp(...)` but discards the result, so
  the official model never actually clamps presence logits. The MLX port
  mirrors this effective behavior and does not clamp; the
  `clamp_presence_logits` / `clamp_presence_logit_max_val` constructor flags
  are stored for API compatibility but are inert, exactly as upstream.
