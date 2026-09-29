# CS425 proposal draft (max 3 pages, 11 pt)

Replace the bracketed names, then cut this down to three A4 pages including references. Due 2 Oct 2026, 23:59.

## 1. Motivation and goal

Exchange students in Singapore often receive mixed English–Mandarin/Singlish messages when coordinating everyday plans: lunch, a library seat, running late, a last-minute jio. A reply that is too stiff sounds foreign; a reply that overuses slang sounds fake and can be socially risky. We build a **reply coach**, not an autonomous chatbot. Given a peer message and a short session, it suggests two sendable replies: standard English and light local English.

Hypothesis: a hybrid of normalised retrieval, a small generator, and a reranker will beat retrieval-only and generator-only on blind human ratings of appropriateness, style fit, and usefulness.

## 2. Scope and non-goals

In scope: casual peer coordination among students.  
Out of scope: customer service, official email, tutoring, dating, open-domain chit-chat.  
Local style is English-led. We never force slang or invent Mandarin. Low-confidence inputs fall back to safe English.

## 3. Method

End-to-end: dynamic memory → V1 local-term normalisation → retrieve vetted pairs (V0 BM25, then V2 embeddings) → generate extra candidates with either V4 (attention-RNN) or V5 (locally fine-tuned mT5-small) → V6 rerank → two replies.

Memory M2 keeps the newest two turns and adds up to two relevant older turns from the last eight, first with BM25, later with embeddings. Conversations are not stored across sessions.

V5 is a Kaggle-gated ~300M-parameter experiment. If it is not clearly better than V4 on the same development set, we drop it. No LLM APIs.

## 4. Data

Public DailyDialog (CC BY-NC-SA 4.0) supplies general dialogue patterns for V3/V4. The target for the dual-style task is 300 team-authored items, each independently peer-reviewed and checked for privacy. After all items complete review, a fixed, balanced 80:20 split assigns 240 development items and 60 untouched benchmark items. No LLM-generated labels.

## 5. Evaluation

Main evidence is blind human evaluation on held-out conversations, comparing retrieval-only, generator-only, and hybrid. Ratings: contextual appropriateness, intended-style fit, usefulness to an exchange student. Automatic overlap metrics are secondary. We also report compute cost and V4 vs V5.

## 6. Implementation

Trainable on CPU for V0–V4; V5 on Kaggle/Colab GPU. Shared JSONL schema and candidate format. Code is organised so each role has a runnable artefact every Friday.

## 7. Roles

A data/eval; B retrieval/memory; C RNN+attention; D Transformer experiment; E reranker/demo. Names: [A], [B], [C], [D], [E].

## References

Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). DailyDialog. IJCNLP.  
Xue, L. et al. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. NAACL.  
Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.  
Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate.
