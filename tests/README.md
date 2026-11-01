# Test suite layout

Tests are grouped by the contract they protect:

- `api/`: public exports, builders and video predictor sessions.
- `checkpoint/`: checkpoint key mapping, tensor layout, loading and the pinned
  parameter tree.
- `image/`: the image processor, mask head, ROI Align, resize preprocessing, the
  interactive predictor and the medical fixtures.
- `medmlx/`: the registered runner's inputs, weights and outputs.
- `perflib/`: connected-component labeling.
- `runtime/`: attention, encoder, decoder, geometry, position encoding, caches,
  precision and other MLX building blocks.
- `unsupported/`: the error raised for upstream features that are not ported.
- `video/`: video frame loading, per-frame inference and box tracking.

Fixture paths live in `tests/_paths.py`.
