"""Output-level abstention diagnostics; no input-to-refusal rules."""
import re

REFUSAL = 'Information not available.'


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def canonical_refusal(text):
    return normalized(text) == normalized(REFUSAL)


def uncertainty_cue(text):
    return bool(re.search(r"information (?:is )?not available|not enough information|don't know|do not know|not sure|cannot tell|can't tell|need more (?:context|information)",text,re.I))


def value_match(text, values):
    # A lexical diagnostic, not entailment: subject/polarity may still be wrong.
    reply=' '+normalized(text)+' '
    return all(any(' '+normalized(alias)+' ' in reply for alias in aliases) for aliases in values)


def metrics(rows):
    missing=[r for r in rows if r.get('requires_refusal') is True]
    answerable=[r for r in rows if r.get('requires_refusal') is False]
    facts=[r for r in answerable if r.get('reference_values')]
    recall=sum(canonical_refusal(r['english']) for r in missing)/max(1,len(missing))
    over=sum(canonical_refusal(r['english']) for r in answerable)/max(1,len(answerable))
    return dict(missing_cases=len(missing),answerable_cases=len(answerable),
        exact_abstention_on_missing=sum(canonical_refusal(r['english']) for r in missing),
        uncertainty_cue_on_missing=sum(uncertainty_cue(r['english']) for r in missing),
        exact_over_refusals=sum(canonical_refusal(r['english']) for r in answerable),
        uncertainty_cue_on_answerable=sum(uncertainty_cue(r['english']) for r in answerable),
        exact_abstention_recall=recall,exact_over_refusal_rate=over,
        answerability_balanced_score=.5*(recall+1-over),
        supported_fact_cases=len(facts),supported_value_matches=sum(value_match(r['english'],r['reference_values']) for r in facts),
        generation_flags=[r['item_id'] for r in rows if r['english_status']!='complete'],
        note='Exact standard-response classification and lexical value checks, not conversational or semantic accuracy. Noncanonical uncertainty is reported separately.')


def eligible(scores, baseline):
    return (not scores['generation_flags'] and scores['exact_over_refusal_rate']<=.10
        and scores['supported_value_matches']>=baseline['supported_value_matches']-1)
