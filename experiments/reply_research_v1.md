# Research audit: improving mT5 student replies and Singlish

This is an evidence review and proposed experimental design, not a new training launch. Preserve the mT5-base architecture and existing checkpoints. Scope is actual conversation-to-reply generation and meaning-preserving Singlish, not multiple-choice diagnostic performance. Read-only reproducible target/output audit: scripts/audit_reply_research_v1.py; numerical results: reply_research_v1_audit.json. No locked test read, new corpus download, training, checkpoint replacement or gold-label alteration.

## What the evidence establishes

1. Human evaluation is decisively negative: on30 fresh conversational inputs hybrid5 preferences, earlier paired5, tie1, both bad19. These are whole-pair preferences, not separately measured STANDARD or LOCAL accuracies. Only one per-case issue flag was explicitly recorded; absent flags are not passes. User aggregate feedback identifies unnatural Can openings and neutral copies/near-copies.
2. Training targets drifted substantially from original team-authored replies. Compared after case/punctuation normalization,187/190 LOCAL targets and14/190 STANDARD targets differ in shared_intent_v1_reviewed.json. Its audit's117 local edits are edits to generated teacher candidates, NOT a count of changes from original student targets. Sentence-initial Can, or Can! occurs27/190 originals versus48/190 curated targets. All initial Can forms, including grammatical questions, occur33 versus51. This is measured style concentration introduced/amplified during curation, not proof every change was bad. Hybrid output starts Can20/30; earlier paired14/30. Exact copy counts do not capture near-copies.
3. Full STANDARD training overfit: epoch8 at3e-4 train loss.0334,validation3.7708; epoch1 validation2.1106. Full1e-4 best epoch2 validation2.1242 versus epoch8 validation2.8897. LoRA did not eliminate generalization difficulties. Full-model capacity to fit small data is established; useful generalization is not.
4. Sequential LOCAL training and loss use authored STANDARD anchors, while actual inference uses generated anchors. Validation explicitly describes its LOCAL loss as oracle-conditioned. A reference-based LOCAL loss cannot establish rewrite quality for arbitrary generated drafts.
5. The current LOCAL model sees conversation+STANDARD and can answer the conversation again rather than rewrite the chosen intent. Observed changes include I can do it for you→you can do it for you, refusal→I go run first, and when done→before bidding ends. This makes anchor-only rewriting a testable alternative, not a proven fix.
6. The rewrite checkpoint used in the new comparison is final step784; its report has best reference loss at epoch4. Earlier rewrite experiments fixed final-step exposure for a controlled research comparison, not deployment selection. Using this final adapter as the only practical rewriter candidate does not establish that the best available checkpoint was tested. Best-vs-final is a cheap development diagnostic; do not repeatedly select using the now-reviewed30 cases and call them untouched.
7. The app's plain-context API guesses alternating speakers, but10/190 original training contexts and5/50 validation contexts do not match that alternation. Batched evaluations preserve explicit roles. Training CoSEM preparation already preserves responder-relative speaker IDs; do not falsely blame it for dropping all roles. Production input should carry explicit roles rather than guessing.
8. Stage1 assigns STANDARD to DailyDialog and LOCAL to CoSEM. Topic/source and register are confounded, unlike parallel same-content style pairs. This is a plausible contributor, not demonstrated sole cause. Stage0 already used96,295 Singlish chunks and stage1 used80,000 CoSEM plus76,052 DailyDialog pairs. Merely recommending more of the same continued pretraining ignores work already done.
9. The original nine scenario categories and nine intent categories emphasize campus logistics and requests; health/emotional support have no explicit taxonomy category. This does not prove there are zero relevant examples, but the infection response indicates conversational-act coverage needs auditing, not just ownership arithmetic. Expanded600 are assistant-generated/curated and blanket-authorized except10 individually reviewed; authorization does not turn all600 into human gold.
10. The reward scorer failed sampled-policy audits even after revision: valid alternatives were rejected and a role reversal received positive reward. The DPO pilot gave8/17 held-out preference rankings before and after; those17 pairs cover8 conversations and are not17 independent scenarios. Structured action RL/SFT selected initialization at chance. These results do not prove all RL or DPO cannot work; they do not justify another RL-first run here.

## Main hypothesis and the different approach

Keep mT5 and actual reply generation. Separate conversational adequacy from stylistic realization, but make LOCAL a constrained rewrite rather than a second conversation responder:

Conversation with explicit roles → STANDARD reply → anchor-only or exemplar-conditioned LOCAL rewrite.

Rebuild parallel data by taking a natural, appropriate LOCAL utterance as the style anchor and producing a faithful STANDARD normalization of THAT utterance. This is data-construction direction, not necessarily reversing runtime generation. Original independently authored STANDARD/LOCAL alternatives are not automatically translations; do not force a natural LOCAL answer to match a different STANDARD intention by mechanically inserting Can. Preserve original local wording when it is good; audit meaning/context rather than treating originals as infallible.

For example, supportive responses to an appointment concern, refusals, explanations, location proposals and reminders need different pragmatic realizations. Do not require a particle on each sentence. Negative examples must establish a real defect in context, not just contain Can or omit lah. Proper names, numbers and very short replies may require little surface change; recognize genuinely neutral copying without forcing awkward decoration.

A distinct runtime alternative worth testing later is LOCAL-first generation followed by STANDARD normalization. It better matches a corpus whose strongest signal is natural LOCAL, but could move response errors to the first stage; it is not an established improvement or part of the first three-run ablation.

## Data changes before training

- Restore provenance: original team targets, public corpus utterances, teacher outputs and assistant corrections stay distinguishable. Do not label assistant checks as independent human gold.
- Review target style distributions by dialogue act, not only global particle frequency. Retain naturally distributed acceptance, refusal, uncertainty, support, teasing, confirmation, requesting and explaining. Audit examples where an initial Can becomes incongruous.
- Build faithful LOCAL↔STANDARD pairs from appropriate original student utterances and selected real CoSEM utterances already available under the project's data process. Source context/roles must be adequate for response-generation supervision. An isolated LOCAL text can still supervise a rewrite pair; it must not become a context-response example with an invented incoming message.
- Include2–3 meaning-preserving surface variants for some anchors, with balanced sampling per original group. Do not count paraphrases as independent conversations or multiply one generic opener hundreds of times. More assistant-generated volume without this control has already failed.
- Train the rewriter on multiple faithful STANDARD phrasings, including accepted generated training-only drafts. Never replace an authored STANDARD anchor with a semantically different generated draft while retaining the old LOCAL target. Correct/rematch or exclude the pair first.
- Treat the last30 reviewed cases and all previously inspected failure sets as development material going forward. Newly created variants inherit their original group's split.

## Generalization protocol

No honest guarantee of zero overfitting is possible. Define success on unseen conversation groups and monitor it.

Split by original conversation/source author and paraphrase/template family before augmentation, retrieval or teacher rewriting. Do not create a supposedly clean validation split by reshuffling examples already seen by the starting checkpoint. Deduplicate within/across groups, including normalized/entity-renamed near-duplicates. Keep a fresh development set for early stopping and a separate frozen final evaluation unseen during changes. Existing data/test/test.jsonl stays locked until the project-defined final evaluation, not accessed in this audit. The old50 validation set has been reused extensively;34 of its conversations explicitly became DPO training for that model. Its status must be model-specific.

Evaluate familiar student topics with new wording/entities as well as withheld combinations and topics. Include ordinary food/chat, emotional support, help, plans, refusals, ownership, status and uncertainty—not predominantly hand-built constraint puzzles. Track training and development token losses per task, but report real generated reply quality. For LOCAL report both reference-anchor diagnostic loss and generated-anchor paired evaluations; neither is a substitute for the other.

Use frequent validation/checkpoints, low learning rates, bounded epochs and patience with a minimum improvement threshold. Treat epoch0 as a candidate. Set these using development evidence before comparison. Keep adaptation capacity/optimizer/decoding fixed in the first ablation; LoRA is not a guarantee against memorization. Fine-tuning with rehearsal or lower-layer freezing is a secondary regularization ablation, not something to stack into every new run.

## Three controlled LOCAL experiments

All use the same frozen STANDARD generator, same repaired parallel training data, grouped splits and evaluation STANDARD strings. This isolates LOCAL changes; none can improve a wrong STANDARD reply by rewriting it.

| Run | LOCAL input | Question |
|---|---|---|
| A: repaired baseline | Conversation + STANDARD | How much does target repair alone help under the current input design? |
| B: anchor-only | STANDARD only | Does removing conversation prevent answering anew and reduce changed meanings? |
| C: exemplar-conditioned | STANDARD + one short training-only style example | Does demonstrating appropriate Singlish improve naturalness beyond B? |

For C choose examples by appropriate linguistic/dialogue-act similarity; omit irrelevant entities/facts and never use a validation/test reply as an exemplar. Train with the same exemplar mechanism used at inference; leave original groups out of their own retrieval. Retrieve language patterns, not an answer to copy. This is a hypothesis requiring B/C comparison; example text cannot be allowed to overwrite the chosen draft's names, times, negation or roles. Different input lengths imply unequal FLOPs even under equal update budgets.

A is retrained using repaired targets and otherwise matched settings; compare to the existing adapter as an incumbent, but avoid attributing every difference to a single factor if initialization/budgets change. B/A isolate context removal; C/B isolate exemplar conditioning. Use one predeclared seed for screening and replicate the selected promising design before a strong improvement claim. Use validation-best checkpoints, not final training loss or whichever sample sounds best.

## Separate STANDARD track: actual reply relevance

Keep the existing epoch1 model as a frozen incumbent, not a proven high-quality solution. Prepare context→reply demonstrations with adequate support/empathy, answering requests, practical proposals and appropriate uncertainty. Incoming-message semantics determine the conversational act: an infection disclosure needs a relevant acknowledgment/follow-up, not an invented meeting arrangement. Student-selectable plausible personal details/refusals remain allowed; explicit contradictions and role reversals do not.

Use naturally phrased minimal contrasts: owner/borrower, completed/not completed, feasible/unavailable, asking where/suggesting where. These remain real conversations with complete natural reply targets, not A/B/C/D answers. Audit role annotations and give the application the same explicit roles. Consider a small auxiliary contradiction/unlikelihood objective only after positive demonstrations work, using manually verified harmful spans; preserve valid alternatives and include ordinary likelihood so the model learns what to say, not merely to lower scores of bad responses. Existing unlikelihood primarily targeted echoes, not a general contradiction verifier. This must be a separate SFT-vs-SFT+negative comparison, not combined with every data change.

## Low-cost diagnostics and acceptance criteria

Before training, compare best-vs-final rewrite checkpoints and greedy-vs-beam4 with fixed anchors on a development batch. This can reveal whether late overfit or decoding amplifies generic openings. Do not equate lower Can frequency with better Singlish, ban Can globally, add particles by regex, or assert no_repeat_ngram_size fixes a repeated opener across different conversations. Generate3–4 candidates only if selection is independently validated; the failed reward judge is not a trustworthy reranker.

Primary endpoints: (1) STANDARD is a sensible next user reply, (2) LOCAL preserves its essential meaning, (3) LOCAL is natural recognizable Singlish. All three matter, with an absolute both-bad/usable-pair measure alongside A/B preference. Mark missing issue ratings as unassessed. Report per-topic/act results, confidence intervals clustered by conversation, and explicit contradiction/role-change counts. Small samples should be reported as exploratory, not decisive wins. Prefix rates, exact copies and lexical diversity are secondary diagnostics, never style scores.

A practical predeclared pilot goal could be a substantial reduction from19/30 both-bad and improved LOCAL naturalness with no rise in meaning changes, but that old30 is now development and must not serve as the independent success test. Set numerical acceptance criteria and the fresh evaluation packet before the next run; no promise that a specific recipe will attain them.

## Relevant primary research and limits of transfer

- Liang et al., ACL2025, **Colloquial Singaporean English Style Transfer with Fine-Grained Explainable Control**: https://aclanthology.org/2025.acl-long.1309/ . Six linguistic dimensions support designing richer style supervision than particle counts. The PDF describes a140K **non-parallel** dataset; it is not140K ready-made human-approved paired replies or proof the published LLM system transfers to mT5. Public linked repository https://github.com/dungxibo123/colloquial_tst has limited documentation; full data availability, provenance and permitted use must be checked before ingestion. No new data downloaded here.
- Li et al., ACL2020, **Don't Say That! Making Inconsistent Dialogue Unlikely with Unlikelihood Training**: https://aclanthology.org/2020.acl-main.428/ . Supports a controlled contradiction-negative objective as an alternative to relying on a learned RL judge. Its gains do not guarantee improvement here; bad negatives can suppress valid replies.
- Lee et al., ACL2022, **Deduplicating Training Data Makes Language Models Better**: https://aclanthology.org/2022.acl-long.577/ . Supports deduplication and honest train/evaluation separation, not a claim that our exact source states currently overlap (the audit finds zero exact expanded/validation full-text matches).
- Gururangan et al., ACL2020, **Don't Stop Pretraining: Adapt Language Models to Domains and Tasks**: https://aclanthology.org/2020.acl-main.740/ . Domain adaptation can help in studied classification tasks; this project has already done Singlish adaptation, so another blanket CoSEM pretraining run is not the first justified step.
- Welleck et al., **Neural Text Generation with Unlikelihood Training**: https://arxiv.org/abs/1908.04319 . Repetition/common-token overuse may involve both learning and decoding; it does not establish that the frequency of Can alone is an error criterion.

## Recommended order

First target-quality repair with original LOCAL kept as the style reference and meaning-preserving normalization. Then grouped fresh evaluation and cheap checkpoint/decoding checks. Run the three LOCAL ablations while separately preparing a focused STANDARD-content repair set. Combine only after component metrics improve. Keep RL/DPO deferred until demonstrations, generalization and reward labels support them. No new GPU jobs were launched as part of this research audit.
