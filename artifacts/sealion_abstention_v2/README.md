# SEA-LION abstention experiment V2: historical diagnostic

This run is retained for reproducibility and is **not an approved replacement model**. A subsequent audit found wrong or ambiguous speaker references in assistant-authored augmentation: incoming Friend questions used “I/my” while the supplied facts described User. This potentially invalidates 72 supported synthetic targets. Original team data and original 790 training records were unaffected.

See `data_quality_correction.json` for the audit and `../sealion_abstention_v3/README.md` for the corrected run, reset from the original English adapter. V1/V2 scores cannot establish a clean improvement. No defaults or original checkpoints were replaced.

The verified results-only archive `results_only_v1.tgz` contains data, code, logs and evaluations, without weights. The V2 archive includes the audit in its summary.
