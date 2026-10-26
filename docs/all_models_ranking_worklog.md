# All-model ranking work log

## 26 October 2026

The user approved pursuing a pretrained shared ranker without a new training run
and asked for the remaining work to be documented. The requested brainstorming
workflow requires review of the written design before implementation.

| Work item | Status | Evidence |
| --- | --- | --- |
| Check current code, model cards, data boundaries and score calibration | Complete | Existing sources and completed comparison report reviewed |
| Install missing V5 inference dependencies | Complete | Existing virtual environment; PEFT 0.21.2, SentencePiece 0.2.2, Accelerate 1.15.0 |
| Verify real local V5 inference | Complete for fixed subset | 6/6 exact saved reply texts across three validation conversations; `roles/D_transformer/LOCAL_VERIFICATION.md` |
| Measure local V5 runtime | Complete for fixed subset | CPU-only PyTorch; 50.92 s initial load, 4.51-6.44 s per Standard/Local pair |
| Freeze proposed shared ranking method and review scope | Written for review | `docs/all_models_ranking_design.md` |
| Implement experimental All models mode and frozen source-neutral scorer | Pending written-design review | No product implementation changes yet |
| Replay selector against the existing 50-item candidate/rating pool | Pending implementation | No new training labels or judge calls planned |
| Live demo verification, setup instructions and method documentation | Pending implementation | Existing default modes remain unchanged |

The 4,000 completed ratings, 190/50 development split, locked test set, checkpoint
files and original exports are preserved. The received V5 base/adapter continue
to live outside tracked source. No PR, push or model retraining is claimed.
