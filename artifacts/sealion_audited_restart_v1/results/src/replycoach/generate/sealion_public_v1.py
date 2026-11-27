"""Conservative filtering and group-aware sampling of existing public dialogues.

These checks remove detectable problems; they are not semantic correctness or
independent human style labels. No replies are generated or rewritten.
"""
import hashlib,json,random,re,unicodedata
from collections import Counter,defaultdict

CONTACT=re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\w)@\w+|\b(?:\+?65[ -]?)?[689](?:[ -]?\d){7}\b|#\d{1,3}-\d{1,4}|\b(?:CS|IS|SMT|MGMT)\d{3}\b',re.I)
PLACEHOLDER=re.compile(r'\{\{|\{[A-Z]{1,4}\}|\[[^\]]+\]|<[A-Za-z_/][^>]{0,60}>|deleted this message|missed (?:voice|video) call|\ufffd',re.I)
SPEAKER_CODE=re.compile(r'\b\d{2}[A-Z]{2,6}\b')
BLOCKED=re.compile(r'\b(?:fuck\w*|wtf|stfu|shit\w*|bitch\w*|cunt\w*|dick\w*|pussy|asshole\w*|knn|knnbccb|ccb|cibai|chibai|lanjiao|pukimak|nabei)\b',re.I)
FILLER=re.compile(r'^(?:(?:ha)+|lol+|lmao|ok(?:ay)?|yes|no|yup|yep|oh|hmm+|thanks|thank you|good morning)[\s.!?,]*$',re.I)
CJK=re.compile(r'[\u3400-\u9fff]')
STRONG=re.compile(r'\b(?:lah|leh|lor|sia|liao|paiseh|makan|dabao|tabao|jio|anot|chope|bochap|sian)\b|can or not',re.I)
CASUAL=re.compile(r'\b(?:alr|tmr|cmi|rmb|pls|plz|u|ur|rn|cos|abt|dun|nvr|gonna|wanna)\b',re.I)


def normal(text):return ' '.join(unicodedata.normalize('NFKC',text).lower().split())


def pair_key(row):return (normal(row['message']),normal(row['response']))


def group_dd(rows):
    """Reconstruct contiguous original pair-export dialogues; fail on broken ordering."""
    groups=[];current=[];previous=None
    for row in rows:
        if not row['context']:
            if current:groups.append(current)
            current=[];previous=None
        elif previous is None or normal(row['message'])!=normal(previous['response']):
            raise ValueError('DailyDialog pair order cannot establish dialogue groups')
        current.append(row);previous=row
    if current:groups.append(current)
    result=[]
    for group in groups:
        utterances=[group[0]['message']]+[r['response'] for r in group]
        key=hashlib.sha256(json.dumps([normal(t) for t in utterances]).encode()).hexdigest()
        result.extend([{**r,'group_id':'dd-'+key} for r in group])
    return result


def assign_groups(rows,corpus):
    if corpus=='dailydialog':return group_dd(rows)
    if corpus!='cosem':raise ValueError(corpus)
    if any(not r.get('chat') for r in rows):raise ValueError('CoSEM chat metadata required')
    return [{**r,'group_id':'cc-'+hashlib.sha256(r['chat'].encode()).hexdigest()} for r in rows]


def detokenise(text):
    text=re.sub(r'\s+([,.?!;:])',r'\1',text)
    text=re.sub(r"\s+(['’])\s*(s|m|re|ve|ll|d|t)\b",r'\1\2',text)
    return ' '.join(text.split())


def clean(row,corpus):
    context=row['context'];message=row['message'];reply=row['response']
    if not isinstance(context,list) or not all(isinstance(t,str) for t in context):return None,'invalid_context'
    if not isinstance(message,str) or not isinstance(reply,str):return None,'invalid_text'
    text=' '.join([*context,message,reply])
    if CONTACT.search(text):return None,'contact_or_identifiable_pattern'
    if PLACEHOLDER.search(text):return None,'placeholder_or_export_artifact'
    if SPEAKER_CODE.search(text):return None,'embedded_speaker_code'
    if BLOCKED.search(reply):return None,'blocked_response'
    if CJK.search(reply):return None,'non_english_led_response'
    if FILLER.fullmatch(reply.strip()):return None,'filler_response'
    if normal(message)==normal(reply):return None,'exact_echo'
    if not 2<=len(reply.split())<=35 or not 1<=len(message.split())<=60 or len(text.split())>200:return None,'word_budget'
    if not any(c.isalpha() for c in message) or sum(any(c.isalpha() for c in w) for w in reply.split())<2:return None,'non_linguistic'
    expected=['user' if (len(context)-i)%2 else 'peer' for i in range(len(context))]
    if corpus=='cosem':
        if len(context)<2:return None,'insufficient_context'
        if any(len(t.split())>30 for t in [*context,message,reply]):return None,'merged_or_long_turn'
        substantial=re.sub(r'\b(?:ha(?:ha)*|lol+|lmao|omg|eh|ah|lah|leh|lor|sia)\b',' ',message.lower())
        if len(re.findall(r"[a-z]+(?:'[a-z]+)?",substantial))<2:return None,'filler_incoming'
        speakers=row.get('speakers',[])
        if speakers!=expected:return None,'non_two_party_visible_window'
    else:speakers=expected
    fix=detokenise if corpus=='dailydialog' else lambda t:' '.join(t.split())
    result=dict(context=[fix(t) for t in context],speakers=speakers,message=fix(message),response=fix(reply),corpus=corpus,group_id=row['group_id'],style='DIALOGUE')
    result['stratum']='explicit_local_markers' if STRONG.search(result['response']) else 'casual_shortforms' if CASUAL.search(result['response']) else 'other'
    result['item_id']='PUBLIC-'+hashlib.sha256(json.dumps([corpus,row['group_id'],result['context'],result['message'],result['response']],ensure_ascii=False).encode()).hexdigest()[:20]
    return result,None


def sample(rows,limit,group_cap,response_cap=3,seed=425,stratified=False):
    rng=random.Random(seed);pool=list(rows);rng.shuffle(pool)
    buckets=defaultdict(list)
    for row in pool:buckets[row['stratum']].append(row)
    # Coverage heuristics are not Singlish labels. Remaining capacity is filled
    # from all strata, with no particles appended or responses rewritten.
    wanted=[('explicit_local_markers',round(limit*.375)),('casual_shortforms',round(limit*.25))] if stratified else []
    selected=[];groups=Counter();responses=Counter();seen=set()
    def take(candidates,n):
        if n<=0:return
        count=0
        for row in candidates:
            key=pair_key(row);response=normal(row['response'])
            if key in seen or groups[row['group_id']]>=group_cap or responses[response]>=response_cap:continue
            selected.append(row);seen.add(key);groups[row['group_id']]+=1;responses[response]+=1;count+=1
            if count>=n:break
    for name,n in wanted:take(buckets[name],n)
    if len(selected)<limit:take(pool,limit-len(selected))
    rng.shuffle(selected)
    return selected
