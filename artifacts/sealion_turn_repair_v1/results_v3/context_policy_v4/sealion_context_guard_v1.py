"""Conservative explicit-context reply rules; not a semantic correctness oracle.

Rules return evidence and provenance. Unsupported inputs retain the generator.
The neutral call reply proposes a next action; it does not infer a private topic.
"""
import re

CALL = re.compile(r'\b(call|phone|ring|voice chat|speak|talk)\b', re.I)
DAY = r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
TIME = r'(?:\d{1,2}(?::\d{2})?\s*[ap]m|\d{1,2}:\d{2})'
NEUTRAL_CALL = "I'll send you a message explaining what I wanted to discuss."


def clean(text):
    return text.replace('’', "'").strip()


def terms(text):
    stop = {'i','you','me','my','your','our','the','a','an','to','of','at','in','on','for','and','is','are','am',
        'it','this','that','we','can','could','would','will','want','wanted','need','about','call','talk','speak',
        'now','when','time','free','not','sure','have','do','what','with','let','then','okay','oh','please',
        'chat','phone','ring','voice','discuss','discussion','conversation','s','just','moment','minute',
        'minutes','few','bit','quick','give','get','later','another','afterwards','break'}
    return {w.lower() for w in re.findall(r'[A-Za-z]+', text) if w.lower() not in stop}


def result(english, rule, evidence):
    return dict(english=english, action='grounded_rule', rule=rule, evidence=evidence,
        provenance='Deterministic context rule, not neural English generation.',
        limitation='Limited explicit patterns; not a general entailment verifier.')


def money(text):
    return [float(x) for x in re.findall(r'\$\s*(\d+(?:\.\d{1,2})?)', text)]


def amount(value):
    return '$'+format(value, 'g')


def minutes(text):
    text=re.sub(r'\s+', '', text.lower())
    match=re.fullmatch(r'(\d{1,2})(?::(\d{2}))?([ap]m)?', text)
    if not match:return None
    hour=int(match[1]);minute=int(match[2] or 0);suffix=match[3]
    if minute>=60 or hour>=24 or (suffix and not 1<=hour<=12):return None
    if suffix:hour=hour%12+(12 if suffix=='pm' else 0)
    return hour*60+minute


def borrowing(row):
    bindings=[]
    if any(re.search(r'\b(?:returned|gave\b.*\bback|no longer have)\b',t,re.I) for t in row['context']):
        return bindings
    for speaker,text in zip(row['speakers'],row['context'],strict=True):
        if speaker not in {'user','peer'}:continue
        opposite='peer' if speaker=='user' else 'user'
        text=clean(text)
        for pattern,borrower,owner in [
            (r'\bI borrowed your ([\w -]+?)(?=[.;,!?]|\s+(?:last|yesterday|for|and)\b|$)',speaker,opposite),
            (r'\bYou borrowed my ([\w -]+?)(?=[.;,!?]|\s+(?:last|yesterday|for|and)\b|$)',opposite,speaker),
            (r'\bYou lent me your ([\w -]+?)(?=[.;,!?]|\s+(?:last|yesterday|for|and)\b|$)',speaker,opposite),
            (r'\bI lent you my ([\w -]+?)(?=[.;,!?]|\s+(?:last|yesterday|for|and)\b|$)',opposite,speaker)]:
            for m in re.finditer(pattern,text,re.I):bindings.append((m[1].strip(),borrower,owner,text))
        antecedent=re.search(r'\b(?:your|my) ([\w -]+?) (?:is|was|still)\b',text,re.I)
        if antecedent and re.search(r'\bI borrowed it\b',text,re.I):
            bindings.append((antecedent[1].strip(),speaker,opposite,text))
        if re.search(r'\byou lent it to me\b',text,re.I):
            bindings.append(('it',speaker,opposite,text))
    return bindings


def guard_context(row, english):
    incoming=clean(row['message'])
    mine=[clean(text) for speaker,text in zip(row['speakers'],row['context'],strict=True) if speaker=='user']
    theirs=[clean(text) for speaker,text in zip(row['speakers'],row['context'],strict=True) if speaker=='peer']
    default=dict(english=english,action='keep_model',rule=None,evidence=[],
        provenance='Original neural English reply; no supported context rule applied.',
        limitation='No rule match does not establish correctness.')
    # Purpose is a private fact when it was never supplied. Do not invent one.
    purpose=re.search(r'\b(?:what|which|why|reason|topic)\b.*\b(?:about|regarding|discuss|call|talk|speak|purpose|topic)\b',incoming,re.I)
    call_history=any(CALL.search(t) for t in mine)
    if purpose and call_history:
        for text in reversed(mine):
            for sentence in re.split(r'(?<=[.!?])\s+',text):
                if re.match(r'I (?:need|want|wanted|would like|was hoping) (?:your advice|to (?:ask|discuss|figure|check|go through)|help with)',sentence,re.I):
                    topic=re.split(r'\s+I can (?:talk|speak)\b',sentence)[0].strip()
                    return result(topic.rstrip('.?!')+'.','stated_call_purpose',[text])
        supported=terms(english).intersection(set().union(*(terms(t) for t in mine)))
        asking_back=re.search(r'^(?:could|can|would) you (?:call|ring|phone)|^(?:what|why).*\?',english,re.I)
        if supported and not asking_back and not re.search(r'\burgent\b',english,re.I):return default
        return result(NEUTRAL_CALL,'unstated_call_purpose_text_first',mine)
    # Match explicit numeric spending caps only, never inferred preferences.
    cap=None;cap_evidence=None
    for text in mine:
        m=re.search(r'(?:\bbudget\b[^.!?]*?|\b(?:at most|no more than|under|less than|up to)\s*)\$\s*(\d+(?:\.\d{1,2})?)',text,re.I)
        if not m:m=re.search(r'\bI have \$\s*(\d+(?:\.\d{1,2})?) left (?:to spend|for)',text,re.I)
        if m:cap=float(m[1]);cap_evidence=text
    purchase=re.search(r'\b(?:buy|book|reserve|order|want|afford|cost|ticket|meal|option)\b',incoming,re.I)
    gift=re.search(r"\b(?:I(?:'ll| will) pay|my treat|free of charge|a gift|I am paying)\b",incoming,re.I)
    if cap is not None and purchase and not gift:
        prices=money(incoming)
        if not prices and theirs:prices=money(theirs[-1])
        if re.search(r'\b(?:which|what).*(?:afford|within.*budget)\b',incoming,re.I):
            affordable=[p for p in prices if p<=cap]
            if affordable:
                return result(f'The {amount(min(affordable))} option is within my {amount(cap)} budget.',
                    'explicit_affordable_option',[cap_evidence,theirs[-1] if theirs else incoming])
        if len(prices)==1 and prices[0]>cap:
            return result(f'That is over my {amount(cap)} budget. Could we find something cheaper?',
                'explicit_budget_limit',[cap_evidence,incoming,*theirs[-1:]])
    # Reject a concrete proposed time earlier than an explicit same-day bound.
    proposed=re.findall(TIME,incoming,re.I)
    if len(proposed)==1 and re.search(r'\b(?:meet|make|be there|meeting|work|come|join)\b',incoming,re.I):
        for text in reversed(mine):
            if re.search(r'\btomorrow\b',incoming,re.I) and re.search(r'\btoday\b',text,re.I):continue
            m=re.search(r"\b(?:cannot|can't|can not)\s+(?:leave|reach|arrive|meet|make|be there)\b[^.!?]*?\b(?:before|until)\s*("+TIME+r')',text,re.I)
            if m:bound=m[1]
            elif re.search(r"\b(?:cannot|can't) (?:leave|reach|arrive)(?: [A-Za-z]+){0,3} (?:earlier|before then)\b",text,re.I):
                found=re.findall(TIME,text,re.I)
                if len(found)!=1:continue
                bound=found[0]
            else:continue
            before=minutes(proposed[0]);after=minutes(bound)
            if before is not None and after is not None and before<after:
                return result(f'I cannot make {proposed[0]}. I am not available before {bound}.',
                    'explicit_earliest_time',[text,incoming])
    # Scope a stated busy period to the same queried day/part of day.
    queried=re.findall(DAY,incoming,re.I)
    if len(set(d.lower() for d in queried))==1 and re.search(r'\b(?:work|meet|make|book|use|free|available|join|come)\b',incoming,re.I):
        day=queried[0]
        for text in reversed(mine):
            for clause in re.split(r'[.;]|\bbut\b',text,flags=re.I):
                if not re.search(r'\b'+re.escape(day)+r'\b',clause,re.I):continue
                busy=re.search(r"\b(?:(?:cannot|can't) (?:go out|make|meet|come|join)|unavailable|on duty|shift|exam)\b",clause,re.I)
                if not busy:continue
                if re.search(r'\b(?:may|might|whether|perhaps|if|not sure)\b',clause,re.I):continue
                # A negated obligation does not make someone unavailable.
                if re.search(r"\b(?:no|not|don't|do not)\b[^.;]*\b(?:shift|exam|on duty)\b",clause,re.I):continue
                part=re.search(r'\b(morning|afternoon|evening|night)\b',clause,re.I)
                query_part=re.search(r'\b(morning|afternoon|evening|night)\b',incoming,re.I)
                if part and (not query_part or part[1].lower()!=query_part[1].lower()):continue
                period=day+(' '+part[1] if part else '')
                return result(f'I am not available on {period}. Could we choose another time?',
                    'explicit_busy_day',[clause.strip(),incoming])
    # Borrower/lender facts, including a single explicit pronoun antecedent.
    if re.search(r'\b(?:who|which of us|which one|should|needs? to|am I|are you)\b.*\b(?:return|pass|bring)\b',incoming,re.I) and not re.search(r'\b(?:already|has|have|did)\b',incoming,re.I):
        bindings=borrowing(row)
        matches=[b for b in bindings if b[0]=='it' or terms(b[0]).issubset(terms(incoming))]
        if not matches and len(bindings)==1 and re.search(r'\b(?:it|that|which of us)\b',incoming,re.I):matches=bindings
        unique={(obj,who,owner) for obj,who,owner,e in matches}
        if len(unique)==1:
            obj,borrower,owner,evidence=matches[-1]
            phrase='it' if obj=='it' else ('your ' if owner=='peer' else 'my ')+obj
            if borrower=='user':answer=f'I need to return {phrase} to you.'
            else:answer=f'You need to return {phrase} to me.'
            return result(answer,'explicit_borrowing_direction',[evidence])
    # Retrieve a first-person location only when the friend asks where to find me.
    if re.search(r'\bwhere\b.*\b(?:are you|find you|look for you|you waiting)\b',incoming,re.I):
        for text in reversed(mine):
            m=re.search(r"\bI (?:am|['’]m) (?:waiting |standing |sitting )?(?:at|in|inside|by|next to)\b[^.!?]*",text,re.I)
            if m:return result(m[0].strip().rstrip('.')+'.','explicit_own_location',[text])
    # A follow-up asks which part/object was requested: retrieve that request.
    if re.search(r'\b(?:which|what|do you need)\b',incoming,re.I) and not re.search(r'\b(?:time|day|when)\b',incoming,re.I):
        verbs=re.findall(r'\b(?:check|edit|review|proofread|send|forward|design)\b',incoming,re.I)
        which=re.search(r'\bwhich (?:one|form|notes|document)\b',incoming,re.I)
        if verbs or which:
            for text in reversed(mine):
                m=re.search(r'\b(?:Please(?: help me)?|Could you|Can you|Can u) (check|edit|review|proofread|send|forward|design)\b[^.!?]*',text,re.I)
                if m and (which or m[1].lower() in [v.lower() for v in verbs]):
                    core=terms(m[0])-{'send','forward','check','proofread','review','edit','design','help'}
                    already_answers=core and core.issubset(terms(english))
                    extra_choice=re.search(r'\b(?:and|or|both)\b',english,re.I)
                    asks_back=re.match(r'\s*(?:what|which|who|where|when|why|how)\b',english,re.I)
                    if already_answers and not extra_choice and not asks_back:return default
                    return result(m[0].strip().rstrip('.?!')+'?','explicit_requested_object',[text,incoming])
    return default
