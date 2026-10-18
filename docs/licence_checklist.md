# Licence and privacy checklist

Role A owns this document. Tick before a dataset enters training or the report.

## Hard rules

- No WhatsApp/Telegram exports with real people unless every participant consents and the text is fully rewritten.
- No student IDs, phone numbers, emails, hostel rooms, or identifiable course-group names.
- No LLM-generated dataset items, gold replies or training labels.
- No hosted LLM/API calls inside the system (CS425 rule).

## Evaluation only exception agreed 17 October 2026

Jereme reports professor approval to replace human output raters with persona-based
AI reviewers. The initial ten personas rate appropriateness, naturalness and
style fit; this is not permission to generate the authored dataset or call a
hosted LLM inside ReplyCoach. Preserve independent human peer review of dataset items.

Role E now owns the judging workflow. The standard judge is installed local
Gemma4 12B through Ollama, configured in ignored `.env`; there is no hosted
integration or API key. Keep inputs anonymous and privacy-checked even locally.
Before a full evaluation, check allowed data and runtime and manually spot-check
ratings. Keep the same judge/settings for every compared system, record known
contextual and persona/style biases, and do not claim equivalence to human feedback.
Do not supply personal details, credentials, restricted handouts or repository
files to the judge. Keep ratings as evaluation records, never gold training labels. See
`split_and_rating_protocol.md` for the current protocol and implementation status.

## Team-authored data

| Field | Decision |
| --- | --- |
| Owner | The team |
| Licence to submit for CS425 | Yes |
| Can be public later | Only after all members agree |
| Status now | `draft_needs_team_review` |

## Public data candidates

| Resource | Use | Licence | Status |
| --- | --- | --- | --- |
| DailyDialog (Li et al., 2017) | V3/V4 general dialogue pairs | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en); Hugging Face mirrors such as [roskoN/dailydialog](https://huggingface.co/datasets/roskoN/dailydialog) | Academic OK; share-alike if we redistribute derivatives |
| Cornell Movie-Dialogs | Optional extra English dialogue | Research use; cite Danescu-Niculescu-Mizil & Lee | Not first choice: too cinematic |
| google/mt5-small | V5 Kaggle experiment | Apache 2.0 | Allowed locally / on Kaggle |
| sentence-transformers MiniLM | V2 embeddings | Apache 2.0 | Allowed locally |

DailyDialog homepage: [http://yanran.li/dailydialog](http://yanran.li/dailydialog)

Do not download National Speech Corpus or SEAME unless Role A confirms the licence covers this assignment.

## Model cards to record in Week 9–11

- Parameter count
- Training time
- GPU type (Kaggle T4 / Colab / CPU)
- Peak memory
- Whether V5 beat V4 on the same development set
