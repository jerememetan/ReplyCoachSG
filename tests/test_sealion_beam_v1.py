import itertools,math,re
from types import SimpleNamespace
import torch
from replycoach.generate.sealion_beam_v1 import beam_decode,reorder_legacy_cache,banned_next

class Tok:
    eos_token_id=0
    def decode(self,ids,skip_special_tokens=True):return ''.join({0:'',1:'A',2:'B'}[i] for i in ids)
class Toy:
    def eval(self):return self
    def __call__(self,input_ids,attention_mask,past_key_values=None,use_cache=True):
        prefix=input_ids if past_key_values is None else torch.cat([past_key_values[0][0],input_ids],dim=1)
        # Nonuniform transition probabilities, independent of batching/cache ordering.
        tables=torch.tensor([[.15,.55,.30],[.65,.10,.25],[.30,.60,.10]])
        probs=tables[prefix[:,-1]]
        return SimpleNamespace(logits=probs.log()[:,None,:],past_key_values=[(prefix,prefix.clone())])

def test_wide_beam_matches_exhaustive_finished_and_length_limited_sequences():
    tok=Tok();result=beam_decode(Toy(),tok,[2],re.compile('IMPOSSIBLE'),num_beams=16,max_new_tokens=3,device='cpu')
    options=[];table=[[.15,.55,.30],[.65,.10,.25],[.30,.60,.10]]
    def visit(prefix,score,last):
        if prefix and (prefix[-1]==0 or len(prefix)==3):options.append((prefix,score/len(prefix)));return
        for i,p in enumerate(table[last]):visit(prefix+[i],score+math.log(p),i)
    visit([],0.,2);best=max(options,key=lambda x:x[1])
    assert result['token_ids']==best[0]
    assert math.isclose(result['normalized_score'],best[1],abs_tol=1e-6)

def test_cache_reorder_replicates_and_swaps_beam_parents():
    cache=[(torch.tensor([[1,2],[3,4]]),torch.tensor([[5,6],[7,8]]))]
    r=reorder_legacy_cache(cache,torch.tensor([1,1,0]))
    assert r[0][0].tolist()==[[3,4],[3,4],[1,2]]
    assert r[0][1].tolist()==[[7,8],[7,8],[5,6]]
    assert cache[0][0].tolist()==[[1,2],[3,4]]

def test_no_repeat_and_boundary_termination():
    assert banned_next([1,2,1],2)=={2}
    assert banned_next([1,2],1)=={1,2}
    result=beam_decode(Toy(),Tok(),[2],re.compile('A'),num_beams=4,max_new_tokens=3,device='cpu')
    assert result['termination'] in ['turn_boundary','eos']
    assert 'A' not in result['reply']

def test_reordered_actual_legacy_cache_logits_match_full_context():
    from replycoach.generate.sealion_v1 import MPTConfig,SEALionCausalLM
    c=MPTConfig(d_model=32,n_heads=4,n_layers=1,vocab_size=16,max_seq_len=32,
        attn_config={'attn_impl':'torch','alibi':False},init_device='cpu',tie_word_embeddings=True)
    m=SEALionCausalLM(c).eval();ids=torch.tensor([[3,4],[5,6]])
    with torch.no_grad():
        cache=m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True).past_key_values
        parents=torch.tensor([1,0,1]);tokens=torch.tensor([[7],[8],[9]])
        a=m(input_ids=tokens,attention_mask=torch.ones(3,3,dtype=torch.long),past_key_values=reorder_legacy_cache(cache,parents),use_cache=True).logits[:,-1]
        b=m(input_ids=torch.cat([ids[parents],tokens],dim=1),attention_mask=torch.ones(3,3,dtype=torch.long),use_cache=False).logits[:,-1]
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
