# V6 selection diagnosis — 27 October 2026

Inspected the installed, unaugmented trained V6 checkpoint. Recomputed all 100
Standard/Local decisions on the same 50 development conversations and reproduced
their selected text and logits (absolute tolerance1e-5). Original packet, source
export and 4,000-rating hashes matched the saved evaluation. No training, new
judge calls or test-set access was involved.

## Read the individual decisions

The local [browser casebook](http://127.0.0.1:8512/) shows the exact M2 context,
incoming message, selected and rejected replies, proposing models, reranker
logits, existing AI-persona means and explanations, and the augmented V6 choice.
Search an item ID or use the style/outcome filters. Decisions are ordered by the
gap to the highest AI-rated available reply. They are not filtered by a new
manually invented error-matching rule.

Private generated evidence is under
`artifacts/reranker_v6/diagnostics_20261027_01/`: `diagnostics.json`, `casebook.md`,
`build_diagnostics.py`, `render_report.py`, `viewer/index.html` and the console
log. The HTML is self-contained; it makes no remote requests. To reopen after
the temporary server stops, open `viewer/index.html` in a browser or run:

```powershell
.\.venv\Scripts\python.exe -m http.server 8512 --bind 127.0.0.1 --directory artifacts/reranker_v6/diagnostics_20261027_01/viewer
```

## What the measurements show

| Diagnostic, out of 50 per style | Standard | Local |
| --- | ---: | ---: |
| Selected reply tied the highest available AI mean | 28 | 26 |
| A higher AI-rated alternative was available | 22 | 24 |
| Gap to highest AI mean was at least 1 point | 19 | 16 |
| Every available reply had AI mean below 3/5 | 8 | 1 |
| Selected reply rated higher than V5's proposal | 10 | 9 |
| Selected reply tied V5's proposal rating | 28 | 30 |
| Selected reply rated lower than V5's proposal | 12 | 11 |
| Any candidate input exceeded 512 tokens | 0 | 0 |
| Selected reply was supplied only as a safety fallback | 0 | 0 |

The hypothetical highest-AI-rated available choice averages 4.344 Standard /
4.760 Local, versus V6's 3.528 / 3.878. This is a diagnostic selection ceiling
computed after seeing ratings, not an implementable selector or unbiased quality
estimate. It suggests selection headroom under this judge. Nine cases also have
a candidate-pool limitation under the same ratings: selection cannot create an
appropriate response when none is supplied.

Exact duplicate text is grouped with the same whitespace normalization as the
ranker. Its lowest anonymous candidate ID supplies ratings, matching the original
replay rather than choosing the most favorable duplicate rating. One duplicate
group per style has different AI means across its separately judged source
records (maximum spread0.3 Standard / 0.2 Local), another sign of judge noise.

## Concrete failures from inspection

These are manually inspected examples, not counts from a comprehensive semantic
annotation of all 100 cases.

| Case | Input detail | Selected reply | Available alternative | Observed problem |
| --- | --- | --- | --- | --- |
| ITEM-C-023 Standard | Both people said they were free Saturday; peer offers a night-market invitation | V5: “Sure, I'll be free on Friday.” | MiniLM: “Sure! What time should we meet?” | Invents a different day; the alternative avoids the contradiction |
| ITEM-A-046 Standard | Peer proposes “2pm u can anot” | BM25/MiniLM: “Sure, what time should we meet?” | V5: “Okay, I'll see you then” | Asks for information just supplied |
| ITEM-B-034 Local | You offered an update; peer asks “can tell me what I miss?” | V4: “you can send me the details” | BM25: “can, i send you the link now” | Reverses who should send the information; the alternative has the right direction, although its invented link still merits review |
| ITEM-E-009 Standard | Peer asks to borrow your charger after3 | BM25/MiniLM: “Ok , I will return once I have charged my phone” | V5: “Sure, I'm going to be able to do that.” | Replies as the borrower instead of the lender |
| ITEM-A-044 Local | Peer reminds you to submit the project | V4: “yes i am sorry i can't it i will be able to find it” | V5: “no worries, thanks!” | Chooses incoherent generated text over a fluent acknowledgment |

For ITEM-A-046, the mistaken reply scores **1.836**, while V5's relevant reply
scores **-2.121**. This is an actual ranking error rather than a display/source
mapping problem. The augmented model keeps this mistaken choice. These logits
are not confidence probabilities.

No input reached the token limit, so 512-token truncation does not explain these
results. M2 still selects only up to two previous turns; four of the 100 cases
have earlier source history outside that window. The browser report identifies
the exact context used rather than implying that the ranker saw every turn.
The day/time examples above have their relevant facts within the supplied M2.

## Leading training-gap hypothesis

The current trainer learns reviewed positive pairs versus two deterministic
donor replies from other training conversations. Donors are not chosen by
semantic difficulty or generated as competing replies to the same conversation.
It optimizes binary pair loss, not the later AI-persona appropriateness scores.
There is no explicit supervision contrasting otherwise plausible replies that
reverse speaker roles, change a day, ignore an already answered question, or
contain broken generator output.

This training-versus-selection mismatch is a plausible contributor to the
observed errors. The diagnosis does not establish the model's internal reasoning
or prove a single cause. More epochs, one keyword heuristic or a new architecture
cannot be inferred to fix it from this analysis alone.

Abbreviation augmentation expands spelling coverage while inheriting the same
reply labels. It does not add the missing distinctions above. It changed only
seven Standard and five Local selections in the saved replay and retained
several clear failures. Its mixed overall result is documented in
[the augmentation experiment](../reranker_augmentation.md).

## The AI judge also needs scrutiny

Do not blindly turn the highest-rated option into a new gold label:

- ITEM-E-007 Standard: the peer asks where to meet; MiniLM's “ok sure, let's eat
  at 2 instead” invents a time/reschedule yet receives an AI mean of5.0.
- ITEM-A-044 Standard: an alternative introduces a Sunday deadline absent from
  the context, yet receives5.0. The selected reply is also problematic, so this
  is not evidence that either response is perfect.
- ITEM-C-058 Local: an alternative mentions a “small table” during a sports-court
  conversation yet receives4.9. The selected departure-plan repetition may also
  be less severe than its1.0 rating suggests.

These are analyst concerns about semantic/factual consistency, not replacement
human-survey scores. The interface keeps the existing ratings intact and flags
the concerns separately. Persona explanations can also introduce unsupported
claims; inspect the actual conversation rather than treating explanations as
proof. The 22/24 missed-choice counts mean disagreement with this AI judge,
not 46 independently human-confirmed model errors.

## Focused next experiment

Collect difficult alternatives for **training conversations**, then review a
small set of appropriate-versus-wrong pairs covering ownership, time consistency,
answered questions and generation quality. Preserve multiple valid replies where
possible. Compare that training change against the current checkpoint on the
unchanged validation set. Do not move the diagnosed validation examples directly
into training or use their AI ratings as training labels.

Separately spot-check the judge on the flagged examples before using automated
scores to claim progress. Keep the 60 held-out examples for the final evaluation.
This analysis adds a diagnosis and report viewer; it does not retrain or switch
the active selector.
