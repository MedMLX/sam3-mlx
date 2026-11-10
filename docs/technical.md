# Technical notes

## Speed and memory (Apple M5 Max, MLX 0.32.4)

Measured on an Apple M5 Max (40-core GPU, 128 GB) with MLX 0.32.4 on Metal and
the pinned `mlx-community/sam3-image` weights, at the default 1008-pixel
resolution, on the chest X-ray fixture. Times are warm medians in fresh
processes after the model is loaded, from 6 counted rounds alternating the three
`precision` settings; ratios are against FP32 with their 97% intervals. Another
workload shared the host during these rounds; on a quiet machine an FP32 image
with one prompt took about 500 ms.

| Step | `"fp32"` (default) | `"tf32"` | `"bf16"` |
| :--- | ---: | ---: | ---: |
| One image with one text prompt (`set_image` + `set_text_prompt`) | 564 ms | 281 ms (×0.48-0.53) | 422 ms (×0.74-0.76) |
| Another text prompt on the same image | 44 ms | 36 ms (×0.80-0.84) | 39 ms (×0.83-0.88) |

The process footprint for one image was within 1% across the three settings; a
further BF16 prompt added 590 MB more than an FP32 one. In FP32, peak MLX memory
through `medmlx run`, weights included, is 5.4 GB for one image and 6.0 GB for a
tracked video session.

Peak memory does not depend on the input image size, because every image is
resized to the 1008-pixel model input: it was the same for a 450×280 CT slice
and a 2560×2930 radiograph. A tracked video session peaked at the same 6.0 GB
over 24 and 96 frames.

### Where the time goes

The ViT trunk (32 blocks, width 1024, 72×72 tokens) is about 90% of an image.
Its matrix multiplies dominate: per image, the MLP projections take about
230 ms, the attention input projection 74 ms and the output projection 25 ms
in full FP32. Attention itself (28 windowed and 4 global blocks) takes about
22 ms, and RoPE, layer norms and the patch embedding under 10 ms together.

In MLX 0.32.4, float32 matrix multiplies use the M5 neural-accelerator kernels
only when TF32 is enabled. `medmlx-core` sets an unset `MLX_ENABLE_TF32` to `0`
so results match the FP32 upstream reference, so with `"fp32"` these
projections run on the regular GPU kernels at 11-15 TFLOP/s. `"tf32"` sets the
variable to `1`; the projections then run about three times faster and
attention is unchanged. For the MLP's first projection (5184 × 1024 by 1024 ×
4736), one product takes 3.3 ms in FP32, 1.0 ms in TF32 and 0.8 ms in BF16.

The fusion encoder's attention heads are 32 wide, below the 64-wide minimum of
MLX's fused attention kernel. Without a mask, `attention_metal.py` runs them
on an FP32 flash-attention Metal kernel at their real width: each 5184-token
self-attention takes 2.1 ms instead of 4.4 ms when zero-padded to 64. A
further text prompt on the same image takes 27% less time (paired fresh-process
rounds, 97% ratio interval 0.68-0.87) and its peak process footprint is
230 MB lower. Against the padded path, masks
on the six medical fixtures are identical, scores differ by at most 1.2e-7 and
boxes by 6e-5 px. Masked 32-wide heads are still zero-padded to 64 for queries
longer than 8 tokens, which keeps the result and avoids MLX's unfused path that
materializes the full score matrix; padding lowered peak memory from 8.6 to
5.7 GB for one image.

### Accuracy by precision

MLX reads `MLX_ENABLE_TF32` at the first float32 product and keeps it for the
process, so `"tf32"` applies to every MLX computation in it; a build whose
`precision` disagrees with the process setting raises an error. Differences
from upstream PyTorch SAM 3 (FP32, CPU) on the six medical fixtures in
`tests/fixtures/upstream_text_prompts.json`, as the largest absolute difference
per image (relative for mask area):

| Fixture | Detections | `"fp32"` score / box / mask area | `"tf32"` | `"bf16"` |
| :--- | ---: | :--- | :--- | :--- |
| Chest X-ray | 2 | 6e-6 / 0.0005 px / 0% | 0.0002 / 0.05 px / 0.03% | 0.0067 / 0.15 px / 0.04% |
| Dermoscopy | 1 | 2e-6 / 0.0006 px / 0% | 0.0008 / 0.07 px / 0.03% | 0.0028 / 0.25 px / 0.01% |
| Microscopy | 4 | 4e-5 / 0.002 px / 0.005% | 0.0046 / 0.36 px / 0.24% | 0.012 / 0.48 px / 0.69% |
| CT, MRI, ultrasound | 0 | — | — | — |

Detection counts match upstream at every precision. Upstream's own examples run
the model under BF16 autocast.
