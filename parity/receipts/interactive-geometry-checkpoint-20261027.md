# Cached interactive-image checkpoint qualification

On 2026-10-27, six supported interactive result sets matched bitwise between
baseline `03bddc7b44ec230a2905a0490fb7ec8738c00750` and
current `53f78cb8edd54a55abb4e74fbcc8e38fae261abc`. All 12 mask candidates were
nonempty; all three current-source rejection checks passed. This is evidence
across the commit interval, which includes the cached-geometry, optional-head,
compiled-output, and processor-state repairs, not attribution solely to one fix.

## Configuration and inputs

Both sequential processes used Apple M1 Max Metal, macOS 27.0.1, Python 3.13.5,
MLX 0.32.4, and NumPy 2.3.5. Settings were eager FP32, seed 0, evaluation mode,
parity preprocessing, resolution 1008, segmentation and interactivity enabled,
and strict loading. Runtime/numerical settings matched; no global compilation
override was set. Cleanup used threshold 0, hole area 256, and sprinkle area 0.

The existing `mlx-community/sam3-image` revision was
`b72a14d8127e17e6f2a3d2e075bbbf4307ba146e`. The existing 309-tensor FP32 tracker
artifact had a conversion manifest identifying `facebook/sam3` revision
`3c879f39826c281e95690f02c7821c4de09afae7`. Each model loaded 1,549 tensors,
including all 149 interactive-head and 22 SAM2-neck parameters. Only the generated
language attention mask was absent; no extra model keys or shape mismatches
remained. The loader classified 160 tracker source tensors as video-only.
Complete loading audits matched between revisions.

Fixtures were `assets/images/test_image.jpg` (1280 x 720) and
`assets/images/groceries.jpg` (800 x 534) in `facebookresearch/sam3` revision
`7f473cfd983d5eb43040ff608f66bff0d7bdcb53`, under the documented SAM License.
Each process encoded the first image singly, then both as a heterogeneous batch.
Fixed point prompts used `(0.45W, 0.5H)`, positive label 1, and three candidates;
box prompts used `(0.15W, 0.15H, 0.8W, 0.85H)` and one candidate.

## Results and gates

All 24 arrays per revision had the declared shape/dtype and finite floating
values. Original-resolution logits, IoU scores, clipped 288 x 288 low-resolution
logits, and masks thresholded at zero were byte-for-byte equal across revisions.
No tolerance was introduced or relaxed. Foreground counts were identical:

| Result set | Foreground pixels per mask candidate |
| --- | --- |
| single_test_image_point | 5110, 6662, 2341 |
| single_test_image_box | 22293 |
| batch_test_image_point | 5110, 6662, 2341 |
| batch_groceries_point | 5248, 198775, 17393 |
| batch_test_image_box | 22293 |
| batch_groceries_box | 79657 |

All candidates were nonempty. Repeated single/batch candidates are not distinct
object counts or a ground-truth quality measurement. After every supported call,
the prior predictor feature objects, image-set/batch flags, and original-size
list were restored. Current-source rejection checks also preserved that state:

- Real features encoded at 504: `ValueError` requesting resolution 1008.
- Two feature rows with one size pair: feature-batch `ValueError`.
- Unequal height/width list lengths: `AssertionError` identifying the mismatch.

## Resource observations and scope

The run spanned `2026-10-27T06:10:06.335493+00:00` through
`2026-10-27T06:10:21.255327+00:00`. Children exited successfully in
7.253 and 7.261 seconds, within 120 seconds each and 300 seconds overall.
Peak child RSS was 3,829,628,928 bytes under a 16 GiB guard; peak reported MLX
active memory was 7,269,523,500 bytes. RSS was sampled every 0.5 seconds and
checked against child high-water values; this is not a hard instantaneous cap.
The configured 512 MiB cache setting was not a hard cap either: observed cache
reached 566,390,183 bytes.

Exact revisions/import paths and input sizes/mtimes were verified around each
child. Existing assets were reused; no weights were downloaded, converted, or
redistributed. Raw predictions and the bounded probe remain local. This is scoped
MLX interval nonregression and API-contract evidence, not official Torch/CUDA
parity, broader accuracy, arbitrary-resolution support, or a speed/memory gain.
No production change was needed.
