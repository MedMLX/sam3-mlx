# Parity evidence index

This index maps repository claims to their machine-readable evidence. Only
current release evidence and the explicitly retained prior BF16 failure are
listed below. Exploratory and synthetic artifacts do not establish release
qualification.

A prior broader synthetic image corpus exposed mask-IoU failures at the frozen
thresholds. Those exploratory artifacts were removed. A seeded COCO val2017
corpus (`manifests/coco-val2017-v1.json`) now characterizes broader images;
adding representative images to the release gate remains deferred in
`../docs/verification.md`.

Later diagnostic characterizations are recorded in
[`image-bf16-20261027.json`](receipts/image-bf16-20261027.json),
[`fast-preprocessing-20261028.json`](receipts/fast-preprocessing-20261028.json),
[`coco-val2017-corpus-20261028.json`](receipts/coco-val2017-corpus-20261028.json), and
[`sam31-expanded-temporal-20261026.json`](receipts/sam31-expanded-temporal-20261026.json).
They do not replace release evidence; their referenced raw diagnostic files may
not be present in a checkout.

| Surface | State | Evidence |
| --- | --- | --- |
| Hardened 11-case precision-adapted pinned-source FP32 image gate | source-bound; inspect recorded verdicts | `receipts/example-image-parity.json`, `receipts/holdout-image-parity.json`, and matching NPZ files under `evidence/` |
| Prior CPU BF16 schema-4 gate | failed, unqualified; exact source-A snapshot | `receipts/bf16-v4-source-A2bd9/` (reports and raw arrays retain their original source and precision) |
| Reviewed Apple M1 Max image-runtime baseline | reviewed for source `70a0736` | `../benchmarks/baselines/apple-silicon-m1-max-20260829.json` |

Versioned release contracts and checkpoint lineage are under `contracts/` and
`manifests/`. The reports are authoritative for source
commit, checkpoint identity, exact command, environment, thresholds, and metrics.
