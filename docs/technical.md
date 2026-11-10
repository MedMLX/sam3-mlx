# Technical notes

## Speed and memory (Apple M5 Max, MLX 0.32.4)

Measured on an Apple M5 Max (40-core GPU, 128 GB) with MLX 0.32.4 on Metal and
the pinned `mlx-community/sam3-image` weights, at the default 1008-pixel
resolution. Times are warm medians in a fresh process, after the model is loaded,
on the chest X-ray fixture. The one-image times come from 7 alternating rounds
of the two settings in fresh processes; TF32 took 0.44-0.46 of the FP32 time
(98% interval).

| Step | Full FP32 (default) | TF32 (`MLX_ENABLE_TF32=1`) |
| :--- | ---: | ---: |
| One image with one text prompt (`set_image` + `set_text_prompt`) | ~530 ms | ~240 ms |
| Another text prompt on the same image | ~50 ms | ~25 ms |
| Peak MLX memory through `medmlx run`, weights included | 5.4 GB | 5.4 GB |
| Peak MLX memory, tracked video session | 6.0 GB | 6.0 GB |

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
so results match the FP32 upstream reference, so by default these projections
run on the regular GPU kernels at 11-14 TFLOP/s. With TF32 they run about three
times faster, and the image encoder takes ~190 ms instead of ~475 ms; attention
is unchanged.

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

### TF32 accuracy

`MLX_ENABLE_TF32` is read once per process and applies to every MLX
computation in it, so sam3-mlx does not change it. Differences from upstream
PyTorch SAM 3 (FP32, CPU) on the six medical fixtures in
`tests/fixtures/upstream_text_prompts.json`:

| Fixture | Detections | Score | Box | Mask area |
| :--- | ---: | ---: | ---: | ---: |
| Chest X-ray | 2 | 0.0001 | 0.05 px | 0.04% |
| Dermoscopy | 1 | 0.0008 | 0.07 px | 0.03% |
| Microscopy | 4 | 0.0049 | 0.38 px | 0.21% |
| CT, MRI, ultrasound | 0 | — | — | — |

Values are the largest absolute difference per image (relative for mask area).
Detection counts match upstream in every case. In full FP32 the same
differences are at most 4.3e-5 in score, 0.003 px in box and two pixels per
mask. Upstream's own examples run the model under BF16 autocast.
