# Compiled image output impact: six-case follow-up

This is a bounded FP32 comparison on three public upstream scenes, not a frozen
release gate or a claim of general numerical equivalence. It follows the
[compiled-output repair qualification](compiled-output-contract-20261027.json).
No production arithmetic was changed for this investigation.

Both lanes used source `a3f37ab21f8270358a459e9b784e3a7ce5381841`, MLX 0.32.4,
Python 3.13.5, and an Apple M1 Max. The existing `mlx-community/sam3-image`
checkpoint was pinned to `b72a14d8127e17e6f2a3d2e075bbbf4307ba146e`.
Settings were FP32, parity preprocessing, seed 0, evaluation mode, strict
checkpoint loading, segmentation enabled, interactive head disabled, text cache
size 0, and confidence threshold 0.5. The model's `compile` option was the only
changed runtime setting; global compilation was not disabled.

The public assets were `assets/images/truck.jpg`, `assets/images/groceries.jpg`,
and `assets/videos/0001/0.jpg` in `facebookresearch/sam3`, revision
`7f473cfd983d5eb43040ff608f66bff0d7bdcb53`. They were unchanged from the prior
qualification pin `2814fa619404a722d03e9a012e083e4f293a4e53`. These are upstream
examples under the documented SAM License, not asserted public-domain images.

## Measured output behavior

| Prompt | Resolution | Detections eager / compiled | Minimum mask IoU | Changed mask pixels | Maximum box error, pixels | Maximum retained-score error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| truck | 1008 | 1 / 1 | 1 | 0 | 0 | 0 |
| paper bag | 1008 | 4 / 4 | 1 | 0 | 6.103515625e-05 | 5.960464478e-08 |
| car | 1008 | 1 / 1 | 1 | 0 | 0 | 5.960464478e-08 |
| person | 1008 | 4 / 4 | 1 | 0 | 0.0001220703125 | 5.960464478e-08 |
| truck | 504 | 1 / 1 | 1 | 0 | 0 | 1.788139343e-07 |
| zzzz_not_a_real_class_qqq | 1008 | 0 / 0 | not applicable | not applicable | not applicable | not applicable |

All 11 nonempty instance masks were pixel-identical. The empty negative control
earns no positive segmentation coverage. Detection query selections were
unchanged at thresholds 0.25, 0.5 and 0.75 across all 200 queries per case. Raw
and public semantic maps had zero changes in the diagnostic `logit > 0` mask;
the actual public semantic contract remains unthresholded logits.

All applicable public-output gates passed: equal counts and query identities,
matching shapes/dtypes, finite values, mask IoU minimum 0.95 and mean 0.99,
box error at most 2 pixels, and score error at most 0.025. These existing gates
were kept separate from the unchanged strict numerical checks.

## Numerical limits and disposition

`rtol=atol=1e-5` still fails for raw instance masks, raw semantic maps and public
semantic maps in all six cases. The car case also fails for raw detection
logits; the 504 truck case also fails for raw box coordinates. None of those
failures was relaxed or relabeled as a numerical pass. The observed differences
were nonmaterial to the measured public decisions in this small set; they could
matter on other inputs or near other decision thresholds. No offending
arithmetic operator was isolated, and no production workaround is justified by
these results alone.

Each lane loaded one model and processed the declared cases in order with fresh
image state. Compiled grounding graphs were reused across images, gained a
second shape entry for resolution 504, and reused the 1008 entry afterward.
The two sequential children exited successfully during
`2026-10-27T05:37:58.982625+00:00` through
`2026-10-27T05:38:20.127108+00:00`. Peak child RSS was 4,661,559,296 bytes under a
16 GiB guard. The configured 512 MiB cache limit was not treated as a hard
observed cap. Temporary raw staging peaked at 350,159,244 bytes and was removed
after comparison. Raw arrays and the probe remain local; no weights or image
data are redistributed here. No speedup or memory-optimization claim is made.
