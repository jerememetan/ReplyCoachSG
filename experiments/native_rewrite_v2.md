# Native-target LOCAL rewrite ablations v2

User authorized research-directed data repair, new evaluation instances and three controlled LOCAL experiments. Jobs355353(full_context),355354(anchor_only),355355(exemplar), one RTX3090/job. At launch verification full_context and exemplar running, anchor_only pending group active-job quota; no quota bypass or unrelated cancellation. Results pending, no completion wait.

## Data repair and provenance

Original data/dev/train.jsonl untouched.183/190 original LOCAL strings preserved byte-for-byte;7 quarantined for malformed/ambiguous meaning, clear role conflicts or context incompatibility. STANDARD normalized to match LOCAL, not LOCAL rewritten to match an independently authored STANDARD intention. Repair decisions/English edits in scripts/repair_native_targets_v2.py and private original_repair_ledger.json. This is assistant audit, not new human gold. Stage1 initializers have no team fine-tuning, but fixed epoch1 STANDARD generator previously saw original190; author-held validation is component evidence, not pristine end-to-end evaluation.

Selected150 unmodified native CoSEM replies from existing project train/validation chats, no corpus download or original source edits. Deterministic sampling favors marked colloquial utterances with varied openings and chat caps; this is not a representative corpus sample or keyword-based naturalness certificate. Existing privacy/vulgarity filters applied; unclear fragments/names/mistranslations then audited manually. Offline cached Gemma4 12B supplied English-only candidates (normalization job355328); its uncertainty flags were not treated as truth. Retained100, quarantined50, manually corrected30 English normalizations including tense/negation/modal changes and an opposite-direction crop-in/crop-out error. LOCAL was never generated or normalized. All public-derived source text and raw translation candidates live in ignored data/public/native_rewrite_v2/, not Git or public reports. No raw public utterances quoted here.

Final283 native LOCAL targets:224 train(148 original,76 public),59 validation(35 original authorE,24 public-heldout chat replies). Original authors and CoSEM chats disjoint across partitions; exact normalized LOCAL overlap rejected. Alternative STANDARD anchors are only unambiguous contraction expansions, sampled per root item each epoch, not counted as independent data or upweighted duplicated rows. No arbitrary automatic prefix insertion or global Can ban. Dataset scripts and ledgers preserve decisions; six targeted tests include byte-for-byte LOCAL preservation and source group isolation.

## Fresh generation evaluation

Forty new assistant-authored student conversation instances frozen before training,20 development and20 synthetic evaluation; no reference LOCAL replies or training labels. Fixed generated STANDARD anchors from unchanged epoch1 full3e-4 checkpoint (job355335), identical across all three arms. No anchor/target from these40 enters optimization or native validation checkpoint selection. No locked product test opened. Same broad domains as prior work; no claim of a random human sample or no pretraining overlap. Thirty earlier user-reviewed whole-pair cases are now development evidence, not called fresh tests. Original authorE and CoSEM component validation are likewise reported honestly as reused-source task diagnostics.

## Controlled trials

A/full_context: conversation+actual STANDARD, original LOCAL_REWRITE serialization. B/anchor_only: STANDARD only. C/exemplar: STANDARD plus a short native training LOCAL style example. All use same paired data, full582M mT5 stage1 initialization and base hash20b20e1ac9382c2ca83b6e0b2c665ec4fb1104f853c0e2489d9b0ad205f871ac. Same AdamW lr3e-5,weight_decay.01,batch4,seed425,max8epochs,clip1,ordinary target likelihood. Original dropout retained. A/B isolate removal of conversation; C/B isolate style exemplar conditioning. These are not extra letter tasks, RL or DPO.

The incumbent was q/v LoRA trained on a different target/multitask mix and step budget. Our matched full-model trials therefore do NOT isolate data repair alone versus that incumbent. Input lengths and FLOPs also differ among A/B/C, though root examples/update settings match. No claim that full tuning or exemplar retrieval guarantees improvement.

C retriever fit only on training STANDARD, retrieves training LOCAL. Excludes query original author/chat group and normalized identical STANDARD; prefers same coarse dialogue act and short examples without numbers/day names. Target LOCAL never affects query selection; validation/test references never enter index. Same mechanism used during train and inference. Style exemplar is a wording cue, not facts to copy; correctness/naturalness still require inspection. Exact source checked in tests. These controls do not mathematically prevent the model copying example facts.

## Loss tracking and checkpoints

Record canonical training target token loss and separate original/public validation target losses every half epoch. Select minimum macro average of the two validation domains, baseline included, improvement threshold.0001,patience3 evaluations(1.5epochs),max8epochs. This manages overfitting but cannot promise none. All source/target budgets384/64 checked against every target/anchor variant and fresh generated source with the actual tokenizer; observed maxima191/29(full_context),50/29(anchor_only),81/29(exemplar). No truncation.

Only after checkpoint selection generate native reference-anchor validation diagnostics and all40 common generated-anchor cases. Generated-anchor review is distinct from reference-conditioned target loss. No automatic judge/winner promotion. Preserve actual STANDARD strings and LOCAL-conditioning traces; inspect relevance, essential meaning, natural Singlish and context/role errors separately. Prefix/copy rates are diagnostics, not quality scores.

Checkpoints contain native_rewrite_config.json, not the legacy app input flag: they need native_rewrite_v2 serialization (and C's saved style pool), so do not point the old full-context rewriter blindly at B/C. Selected checkpoints and reports remote artifacts/<mode>/. Source/data/settings hashes in native_rewrite_v2_recipe.json; preflight maxima/tokenizer hash in native_rewrite_v2_preflight.json. Existing checkpoints/default remain intact.

## Incumbent controls and STANDARD track

A worker also runs old rewrite_s425 best/final at greedy and beam4 on the20 development anchors only; old base explicitly loaded from identical stage1 weights. These cheap checkpoint/decoding controls are diagnostic, not used to adapt training or select via the fresh20 evaluation cases.

Prepared13 separate assistant-authored STANDARD repair demonstrations from previous reviewed conversations (support/acknowledgments, answering questions, completed state, budgets, availability, owner/return roles). Some improve appropriateness rather than correct explicit contradictions. Human whole-pair Both-bad votes are NOT recast as component labels. These cases are not used in the three LOCAL runs or claimed as fresh evidence; broader content training needs a separate reliable validation set. See experiments/student_content_repair_v2.md.

## Verification and next handoff

Six tests passed, including all three actual tiny CPU training/selection/generation paths; input isolation/no target leakage for B/C; training-only exemplar pool and group exclusion; explicit roles/anchor preservation; actual dataset LOCAL-byte preservation and split isolation. Worker compilation and shell syntax passed. Both active GPU jobs reached validation logging without errors at initial check. No training completion or quality gain assumed. No new Streamlit server or automatic production switch. Next session retrieve all three reports, inspect training/validation curves and every clear meaning/style failure; compare all predeclared arms on frozen evaluation, preserve uncertainty and avoid post-hoc relabeling.
