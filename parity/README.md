# Parity evidence index

This index maps repository claims to their machine-readable evidence. Only
release-qualifying evidence is retained in-tree. Exploratory, failed,
invalidated, and synthetic qualification artifacts may be removed when they are
no longer part of the release contract.

A prior broader synthetic image corpus exposed mask-IoU failures at the frozen
thresholds. Those exploratory artifacts were removed; broader representative-
image validation remains deferred in `../docs/verification.md`.

| Surface | State | Evidence |
| --- | --- | --- |
| Hardened 11-case SAM 3 image gate | source-bound; inspect recorded verdicts | `receipts/example-image-parity.json`, `receipts/holdout-image-parity.json`, and matching NPZ files under `evidence/` |
| Reviewed Apple M1 Max image-runtime baseline | reviewed for source `70a0736` | `../benchmarks/baselines/apple-silicon-m1-max-20260829.json` |

Versioned release contracts and checkpoint lineage are under `contracts/` and
`manifests/`. The reports are authoritative for source
commit, checkpoint identity, exact command, environment, thresholds, and metrics.
