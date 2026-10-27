# V6 cross-encoder fine-tuning plan

Approved by Jereme on 27 October 2026. The user also approved deterministic
mismatched human-written replies as **weak negatives**, with the false-negative
limitation documented. This fulfils the submitted proposal's V6 experiment;
success is a completed, honestly evaluated training run, not a promised gain.

## Fixed scope and data boundaries

- Initialize from `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Retain the frozen baseline.
- Train only on the 190 independently reviewed `data/dev/train.jsonl` items.
  Their Standard/Local replies supply 380 positive pairs. Sample up to two
  same-style donor replies per positive from different training conversations;
  reject either known positive reply and duplicate normalized donor text.
- These sampled mismatches are algorithmic weak supervision, **not human-checked
  negative labels**. Some may still be valid replies. No LLM generates labels or
  new reply text. Do not claim more independent conversations from extra pairs.
- Use the 50 development validation conversations for early stopping. Validation
  negatives draw from training replies only. Keep the 60 test items unopened.
- The existing 4,000 Gemma ratings remain evaluation-only. They never enter the
  training bundle, model inputs, loss, or checkpoint selection.
- Export a minimal private Kaggle bundle: speaker-labelled input, requested
  style, reply, numeric pair label and provenance IDs; remove author/reviewer
  names and free-text review comments. Do not upload the repository or `.env`.

## Training and runtime

Format the selected M2 conversation with its actual speakers, incoming peer
message, requested Standard/Local style and candidate reply. A shared encoder
and one scoring head handle both styles. Use binary cross-entropy with logits,
AdamW, learning rate 2e-5, weight decay 0.01, batch size 16, maximum pair length
512, seed 425 and at most four epochs. Select the best checkpoint by validation
pair loss only. Record history, package/device details, data hashes, base revision,
optimizer updates and evidence that weights changed. Refuse CPU execution in the
Kaggle production notebook; a separate explicit CPU smoke check is allowed.

Save the trained model/tokenizer and a provenance manifest separately from the
frozen scorer. Preserve the reply when truncating, record truncation, and retain
the original source-neutral scoring and duplicate/provenance rules. Add an
explicit fine-tuned selector option; do not replace the existing default.

## Evaluation and promotion

After training is complete, replay the exact saved four-source validation
proposals against the frozen rating records. Compare the fine-tuned selector,
frozen scorer, legacy heuristic, fixed MiniLM and fixed V5. Also report a frozen
scorer with the same style-aware input to distinguish formatting from training.
Do not alter the source models, saved proposals or ratings to improve results.
Report contextual appropriateness, simulated would-send, source selections,
paired conversation uncertainty and latency. Retain the existing caveats about
one correlated AI judge and validation reuse. Promote only with supporting
evidence; a negative result remains a valid V6 experiment.

## Kaggle connection and publication

Register the official remote MCP at `https://www.kaggle.com/mcp` in local Codex
configuration, outside Git. Authenticate through the user's own Kaggle session;
never request credentials in chat. Keep the dataset and notebook private. Check
accelerator availability before launch, use the smallest available GPU, and
download the completed checkpoint for local integration. Weights and raw runs
stay ignored. If MCP authentication is incompatible, document that limitation
and use the existing signed-in Kaggle browser/official CLI when authorized.

## Verification and record keeping

Tests must cover split/overlap guards, deterministic negatives and their exact
provenance, no held-out donors, no mutation of source data, finite train loss,
real optimizer updates, and loading/selection of a trained checkpoint. Run a
small local optimizer smoke check before consuming a Kaggle GPU session. Save
commands, run status and results in the V6 handoff. Commit source/docs/tests in
small chunks; do not push until requested.

Status, 27 October: private Kaggle GPU run completed with 288 optimizer updates;
epoch2 was selected and its weights were downloaded and verified locally.
Authenticated MCP requests now work through the official CLI credential helper.
The trained selector is available separately and remains experimental. Full
results, commands and limitations: [V6 training handoff](../roles/E_rerank_demo/V6_TRAINING.md).
