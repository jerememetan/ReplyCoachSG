# SEA-LION restart from the saved public-dialogue checkpoint

All four jobs completed exit0: preflight358725(1m38s), task358726(19m54s), rewrite358727(14m49s), collector358728(1s). Task stopped after4epochs and selectedepoch2; rewrite stopped after3epochs and selectedepoch1. Queue observed empty. Current launch records are complete_reviewed. This run used constant taskLR1e-4 and rewriteLR3e-5, with no warmup/decay.

## Completed findings

Assistant inspected all75actual paired conversations and24paired fixed-reference rewrite diagnostics. Some LOCAL replies are shorter/more casual, e.g. charge reminder→“okok, i charge then bring”, playlist→“Ya, I done leh”, photo request→“okok, i send you a photo”. These are isolated wording improvements, not independent human ratings.

The exact user flu/video example remains unchanged English: “okay, we will send you videos when we are playing” in both stages. Correct English borrowing Friend headphones becomes “yes pls, borrow mine?”; completed diagram/unfinished explanation becomes “Ya, whole thing done leh.” Correct fixed-reference L8 correction becomes agreement toL3. New English also accepts14-dollar noodles with9available, where original refused. No reliable overall winner or default promotion established.

Full raw comparison: results/review/comparison.csv and .json. Assessment.json records24per-case fresh reviews and reused-case observations; verification.json checks198paired output records, actual prompts/anchors, source digests and stage handoff. Archive414195bytes/142regularfiles/no weights; SHA8bc0ab76fac65d6f484cf5a54c8f62f20d90cfed604e50be567951aeb66af60f. All110included payload files match frozen hashes; seven metadata/license files omitted by collector match local staged copies, with full payload verified by remote runtime. Both sources and all epoch snapshots remain intact.

The user deferred a possible warmup/decay test until after this review. No schedule change or new training was launched by this status request. Constant learning rates were not established as optimal; scheduling and overfitting have not been causally separated.
## Data and training

Starting weights are the saved post-CoSEM/DailyDialog LoRA from full117634public pairs, not repair25 and not a new random adapter. Initial adapter SHA b0dfd6b9e70ba93b3dcbd9a0b76fd35800848fee1508f5c4e85937892acfd36d. Original SEA-LION v1 3B base remains frozen; only10,485,760 rank8 LoRA parameters train. Runtime verifies all loaded adapter tensors match the saved source before training.

A separate task copy has768conversations/1536English+LOCAL targets:177original team conversations and591synthetic. The22flagged source conversations are set aside rather than edited. Only audited synthetic LOCAL wording differs from the corresponding source records. The190mT5A derivative proposals are not used for this SEA run. No original source file was rewritten.

Task training starts from the public adapter, LR1e-4, effective batch8, up to8epochs, patience2. The selected task adapter is saved as the English generator. A separate copy is then trained on766English-anchored Singlish pairs, LR3e-5, same batch/epoch/patience limits. Two additional original pairs with found-versus-reserved seat differences are set aside from rewrite only. New audit permits ordinary context-dependent ellipsis, including yep/okok acknowledgments; older literal-omission exclusions are not blindly reused.

Every epoch including0 is saved. Task selection is half old50equal-style reference CE plus half20new-style equal-style CE; rewrite selection is half old42eligible-pair tokenCE plus half20new-style rewrite tokenCE. No hard retention guard. Reference likelihood is not reply quality; save and review alternatives if actual outputs disagree with CE selection. Intermediate generations use validation only.

## Evaluation and preservation

Original50validation conversations and their targets stay unchanged. The42old rewrite-eligible pairs are a fixed historical subset. Separate20style-validation conversations and24fresh illustrative evaluation conversations were frozen before submission. These are assistant-authored following five human style preferences, not independently human-rated gold. The fresh conversations include related everyday scenarios, so they do not establish broad independent generalization.

The paired evaluation has75inputs:24fresh assistant cases, the reused user flu/video case, and50reused original validation cases. Original repair25 is replayed with its exact portable CLI. The new pipeline generates English first, then uses its actual English as the Singlish anchor. A separate24case fixed-reference rewrite diagnostic isolates rendering from English generation. References never enter ordinary generation. Generation failures remain explicit; there is no reranking, copy penalty or postprocessing repair.

The comparison is a practical old-versus-new pipeline comparison with different training histories, not an isolated causal estimate of target wording. No duplicate old-target control or repeat public training. No default promotion. Original model bundles, saved public adapter, protected backup and original train/validation/split remain unchanged. Source weights are shared by path, without another multi-GB base copy.

Artifacts: recipe.json, data/, stage/, payload.tgz and launch.json. Remote outputs will contain task/rewrite training reports, epoch adapters, validation snapshots,75actual paired outputs, fixed-reference diagnostics, review/comparison.json/.csv and a results-only archive. Check actual accounting/errors on resume; do not relaunch duplicates or infer success from an empty queue. Jobs may continue after this session ends.
