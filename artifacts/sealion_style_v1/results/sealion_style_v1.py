"""Experimental copy-aware rewrite ranking; guards are heuristics, not entailment.

Only the style lexicon is learned, from the supplied training pairs. Generation
rules propose candidates; they never silently rewrite the selected model output.
"""
from collections import Counter
import math
import re

PARTICLES = {'lah', 'leh', 'lor', 'meh', 'sia', 'hor', 'ah', 'loh', 'liao'}
LOCAL_WORDS = {'paiseh', 'makan', 'jio', 'shiok', 'aiyo', 'anot', 'ya', 'yah', 'lah', 'leh', 'lor', 'liao'}
FUNCTION_WORDS = set('a an the i me my mine you your yours we us our ours they them their he him his she her it its is am are was were be been being have has had do does did will would could can should may might must to of for at in on with and or but if so as that this these those please yes no not already yet still just very really sure okay ok thanks thank'.split())
FUNCTION_WORDS.add('cannot')
INITIAL_WORDS = FUNCTION_WORDS | set('my yes no sorry hello hi tomorrow today yesterday morning afternoon evening tonight later before after outside inside let when where what which who why how happy good great sounds thank alright all maybe probably certainly unfortunately definitely'.split())
CONTENT_ALIASES = {'makan':'eat', 'paiseh':'sorry', 'jio':'invite', 'ya':'yes', 'yah':'yes', 'tmr':'tomorrow', 'tml':'tomorrow', 'tonite':'tonight', 'wanna':'want', 'gonna':'going', 'ur':'your', 'u':'you', 'rmb':'remember', 'msg':'message', 'cant':'cannot'}
TIME_WORDS = set('monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday morning afternoon evening tonight before after inside outside'.split())
COLOURS = set('purple blue red green yellow black white orange pink brown grey gray'.split())
NUMBER_WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),map(str,range(21))))
NUMBER_WORDS.update(dict(zip('thirty forty fifty sixty seventy eighty ninety hundred'.split(),map(str,[30,40,50,60,70,80,90,100]))))
ACTION_ROOTS = {'returned':'return','returning':'return','borrowed':'borrow','borrowing':'borrow','lent':'lend','lending':'lend','sent':'send','sending':'send','gave':'give','giving':'give','brought':'bring','bringing':'bring','collected':'collect','collecting':'collect','paid':'pay','paying':'pay','booked':'book','booking':'book','reserved':'reserve','reserving':'reserve','uploaded':'upload','uploading':'upload','submitted':'submit','submitting':'submit','bought':'buy','buying':'buy','got':'get','getting':'get','waited':'wait','waiting':'wait','met':'meet','meeting':'meet','helped':'help','helping':'help'}
ACTIONS = set(ACTION_ROOTS.values())


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?::\d+)?[a-z]*", text.lower().replace('’', "'"))


def canonical(text):
    return ' '.join(words(text))


def without_particles(text):
    return ' '.join(w for w in words(text) if w not in PARTICLES)


def local_features(text):
    value = text.lower().replace('’', "'")
    result = []
    if re.search(r'\b(i|you|we|they)\s+(waiting|going|coming|heading|working|trying)\b', value):
        result.append('progressive_without_copula')
    if re.search(r'\balready[.!?]*\s*$', value):
        result.append('final_already')
    if re.search(r'^\s*can\s+(?:please\s+)?(?:get|help|send|share|check|bring|lend|return|give)\b', value):
        result.append('subjectless_can_request')
    if set(words(text)) & (LOCAL_WORDS-PARTICLES):
        result.append('local_lexical_item')
    if re.search(r'\b(my|your|our|the)\s+\w+(?:\s+\w+){0,2}\s+(?:purple|blue|red|green|ready|free|busy)\b', value) and not re.search(r'\b(is|are|was|were)\b', value):
        result.append('descriptive_copula_omission')
    return result


def copy_eligible(anchor):
    return len(words(anchor)) >= 4 and not local_features(anchor) and not (set(words(anchor)) & LOCAL_WORDS)


def copy_features(anchor, candidate):
    exact = canonical(anchor) == canonical(candidate)
    particle_only = not exact and without_particles(anchor) == without_particles(candidate)
    return dict(exact_copy=exact, unnecessary_copy=exact and copy_eligible(anchor),
                particle_only_change=particle_only)


def protected_atoms(text):
    value = text.replace('’', "'")
    raw = re.findall(r"\$?\d+(?::\d+|\.\d+)?[A-Za-z]*|[A-Za-z]+\d+[A-Za-z0-9]*|[A-Za-z]+(?:'[A-Za-z]+)?", value)
    atoms = set()
    for token in raw:
        lower = CONTENT_ALIASES.get(token.lower(),token.lower())
        if any(c.isdigit() for c in token):
            atoms.add(lower.lstrip('$'))
        elif lower in TIME_WORDS or lower in COLOURS:
            atoms.add(lower)
        elif lower in NUMBER_WORDS and lower != 'one':
            atoms.add(NUMBER_WORDS[lower])
        elif token[0].isupper() and lower not in INITIAL_WORDS | LOCAL_WORDS and lower not in {'i', "i'm", "i've", "i'll", "i'd"}:
            atoms.add(lower)
    # Alphanumeric room identifiers must not be split into letters and digits.
    atoms.update(x.lower() for x in re.findall(r'\b(?:[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+)\b', value))
    return atoms


def negative(text):
    value = text.lower().replace('’', "'")
    value = re.sub(r'\b(no problem|no worries|not a problem|never mind)\b', '', value)
    return bool(re.search(r"\b(no|not|never|cannot|can't|won't|don't|doesn't|didn't|haven't|hasn't|hadn't|isn't|aren't|wasn't|weren't|couldn't|wouldn't|shouldn't)\b", value))


def actor_actions(text):
    value = text.lower().replace('’', "'")
    value = re.sub(r"\b(i|you|we|they|he|she)'(?:ll|ve|re|m|d)\b", r'\1 ', value)
    tokens = words(value)
    result = {}
    auxiliaries = set('will would can could should may might must am are is have has had not already still just please also to'.split())
    for index, token in enumerate(tokens):
        if token not in {'i','you','we','they','he','she'}:
            continue
        for later in tokens[index+1:index+7]:
            if later in auxiliaries:
                continue
            action = ACTION_ROOTS.get(later, later)
            if action in ACTIONS:
                result.setdefault(action, set()).add(token)
            break
    return result


def content(text):
    result = set()
    for token in words(text):
        token = CONTENT_ALIASES.get(token, token)
        if token in FUNCTION_WORDS or token in PARTICLES:
            continue
        token = ACTION_ROOTS.get(token, token)
        result.add(token)
    return result


def guard(anchor, candidate):
    """Conservative surface checks; passage is not proof of semantic equivalence."""
    reasons = []
    if not candidate.strip():
        return dict(passed=False, reasons=['empty'], content_recall=0.)
    if re.search(r'\b(?:Friend|You|Other)(?: \([A-Z_]+\))?:|English draft:|Local rewrite:|rewrite the english', candidate, re.I):
        reasons.append('transcript_or_instruction_leak')
    original, proposed = protected_atoms(anchor), protected_atoms(candidate)
    # Case changes do not delete names (e.g. Mei -> mei).
    proposed.update(atom for atom in original if atom in words(candidate))
    original.update(atom for atom in proposed if atom in words(anchor))
    if original-proposed:
        reasons.append('missing_protected_atom')
    if proposed-original:
        reasons.append('new_protected_atom')
    if negative(anchor) != negative(candidate):
        reasons.append('negation_changed')
    if ('?' in anchor) != ('?' in candidate):
        reasons.append('question_force_changed')
    before, after = actor_actions(anchor), actor_actions(candidate)
    if any(action in after and agents != after[action] for action, agents in before.items()):
        reasons.append('explicit_actor_changed')
    for direction in ['to me', 'to you', 'to us', 'to them']:
        other = re.search(r'\b'+direction+r'\b', candidate.lower())
        if other and not re.search(r'\b'+direction+r'\b', anchor.lower()):
            reasons.append('recipient_changed')
            break
    source_content, target_content = content(anchor), content(candidate)
    recall = len(source_content & target_content)/max(1,len(source_content))
    if source_content and recall < .5:
        reasons.append('low_surface_content_recall')
    return dict(passed=not reasons, reasons=reasons, content_recall=recall)


def grammar_candidates(anchor):
    """Small optional phrase proposals; model scoring and guard still apply."""
    original = anchor.strip().replace('’', "'")
    proposals = []
    def add(value, rule):
        if value != original and guard(original, value)['passed']:
            proposals.append(dict(text=value, provenance='grammar:'+rule))
    value = re.sub(r"\bI(?: am|'m)\s+(waiting|going|coming|heading|working|trying)\b", r'I \1', original)
    value = re.sub(r'\b(You|We|They) are\s+(waiting|going|coming|heading|working|trying)\b', r'\1 \2', value)
    add(value, 'present_progressive_copula')
    match = re.fullmatch(r"(Yes,\s*)?I(?: have|'ve) already (booked|submitted|sent|paid|checked|finished|uploaded|reserved|completed|returned) (.+?)[.!]?", original, re.I)
    if match:
        prefix = 'Yah, I ' if match[1] else 'I '
        add(prefix+match[2]+' '+match[3].rstrip('.!')+' already.', 'completion_already')
    match = re.fullmatch(r'Could you (?:please )?(get|help|send|share|check|bring|lend|return|give) (.+?)\??', original, re.I)
    if match:
        add('Can '+match[1].lower()+' '+match[2].rstrip('?.')+'?', 'subjectless_can_request')
    match = re.fullmatch(r'(My (?:\w+\s+){0,3}\w+) is (purple|blue|red|green|ready|free|busy)\.?', original, re.I)
    if match:
        add(match[1]+' '+match[2]+'.', 'descriptive_copula')
    if original.lower().startswith('yes, '):
        add('Yah, '+original[5:], 'affirmative_yah')
    # A restricted phrase lexicon; no inversion of the retrieval glossary.
    if re.match(r'(?i)^sorry(?:,?\s+I\b|\s+(?:for|about)\s+(?:the\s+)?(?:delay|late|being late|mix.?up|confusion|wait|mistake)\b)', original):
        add(re.sub(r'(?i)^sorry\b','Paiseh',original), 'apology_lexicon')
    return list({canonical(p['text']):p for p in proposals}.values())


def ngrams(text):
    tokens = words(text)
    result = set()
    for width in [1,2,3]:
        for start in range(len(tokens)-width+1):
            part = tokens[start:start+width]
            # Content-only cues and particles alone should not drive style.
            if any(any(c.isdigit() for c in w) for w in part) or all(w in PARTICLES for w in part):
                continue
            if any(w in FUNCTION_WORDS or w in (LOCAL_WORDS-PARTICLES) for w in part):
                result.add(' '.join(part))
    return result


def fit_lexicon(train_pairs, minimum_documents=5):
    english, local = Counter(), Counter()
    for row in train_pairs:
        english.update(ngrams(row['standard_anchor']))
        local.update(ngrams(row['response']))
    size = len(train_pairs)
    weights = {}
    for phrase in sorted(english.keys() | local.keys()):
        if english[phrase]+local[phrase] >= minimum_documents:
            value = math.log((local[phrase]+1.)/(english[phrase]+1.))
            weights[phrase] = max(-2., min(2.,value))
    return dict(training_pairs=size, minimum_documents=minimum_documents, weights=weights,
                scope='Training-only phrase log-odds heuristic; not a calibrated naturalness model.')


def style_score(text, lexicon):
    features = ngrams(text)
    values = [lexicon['weights'][p] for p in features if p in lexicon['weights']]
    lexical = sum(values)/max(1,len(values))
    return max(-1.,min(1.,lexical+.2*len(local_features(text))))


def rank_candidates(anchor, candidates, lexicon, copy_weight=.5, style_weight=.2):
    result = []
    for item in candidates:
        checked = guard(anchor,item['text'])
        if item.get('status','complete') != 'complete':
            checked['passed']=False
            checked['reasons'].append('incomplete_generation')
        copying = copy_features(anchor,item['text'])
        local = style_score(item['text'],lexicon)
        score = -item['mean_nll'] + style_weight*local
        score -= copy_weight*int(copying['unnecessary_copy'])
        score -= .25*int(copying['particle_only_change'])
        result.append(dict(item, guard=checked, copy=copying, style_score=local, rerank_score=score))
    valid = sorted([r for r in result if r['guard']['passed']],key=lambda r:r['rerank_score'],reverse=True)
    return valid, result


def auxiliary_eligible(row):
    return (copy_eligible(row['standard_anchor']) and
            canonical(row['standard_anchor']) != canonical(row['response']) and
            without_particles(row['standard_anchor']) != without_particles(row['response']) and
            guard(row['standard_anchor'],row['response'])['passed'])


def copy_margin_loss(positive_ce, copy_ce, weight, margin=.2):
    import torch
    return .1*positive_ce + weight*torch.relu(margin+positive_ce-copy_ce)


def beam_nbest(model, tok, prompt_ids, boundary, num_beams=8, returns=8,
               max_new_tokens=64, no_repeat_ngram_size=3, device='cuda'):
    """Legacy-cache beam search with a kth-finished early-stop bound."""
    import torch
    from torch.nn import functional as F
    from replycoach.generate.sealion_beam_v1 import reorder_legacy_cache,banned_next,rendered
    if not 1 <= returns <= num_beams or not prompt_ids or len(prompt_ids)+max_new_tokens>384:
        raise ValueError('Invalid beam count or token budget')
    model.eval()
    with torch.no_grad():
        current=torch.tensor([prompt_ids],device=device);past=None;active=[([],0.)];finished=[]
        def score(candidate):return candidate[1]/max(1,len(candidate[0]))
        for position in range(max_new_tokens):
            mask=torch.ones((len(active),len(prompt_ids)+position),device=device,dtype=torch.long)
            output=model(input_ids=current,attention_mask=mask,past_key_values=past,use_cache=True)
            logits=output.logits[:,-1,:].float()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite beam logits')
            logp=F.log_softmax(logits,dim=-1)
            for i,(tokens,_) in enumerate(active):
                blocked=banned_next(tokens,no_repeat_ngram_size)
                if blocked:logp[i,list(blocked)]=-torch.inf
            cumulative=logp+torch.tensor([x[1] for x in active],device=device)[:,None]
            width=logp.shape[-1];values,indices=torch.topk(cumulative.reshape(-1),min(2*num_beams,cumulative.numel()))
            following=[];parents=[]
            for value,index in zip(values.tolist(),indices.tolist(),strict=True):
                if not math.isfinite(value):continue
                parent,token=divmod(index,width);tokens=active[parent][0]+[token]
                _,_,reason=rendered(tok,tokens,boundary)
                if reason or position+1==max_new_tokens:finished.append((tokens,value,reason or 'length_limit'))
                elif len(following)<num_beams:following.append((tokens,value));parents.append(parent)
            # Keep token-distinct completions; surface duplicates are removed later.
            unique={tuple(x[0]):x for x in finished}
            finished=sorted(unique.values(),key=score,reverse=True)[:num_beams]
            if not following:break
            if len(finished)>=returns and score(finished[returns-1]) >= max(v/max_new_tokens for _,v in following):break
            past=reorder_legacy_cache(output.past_key_values,torch.tensor(parents,device=device))
            active=following;current=torch.tensor([[tokens[-1]] for tokens,_ in active],device=device)
        result=[]
        for tokens,value,reason in sorted(finished,key=score,reverse=True)[:returns]:
            raw,reply,_=rendered(tok,tokens,boundary)
            result.append(dict(text=reply.strip(),raw_text=raw,token_ids=tokens,termination=reason,
                               normalized_score=score((tokens,value)),
                               status='failed_empty' if not reply.strip() else 'length_limit' if reason=='length_limit' else 'complete'))
        return result
