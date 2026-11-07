import json,torch
from pathlib import Path
from transformers import MT5Config
from replycoach.generate.mt5_pretrained_compat import OriginalMT5
from replycoach.generate.mt5_matched_v1 import source
ROOT=Path(__file__).resolve().parents[1]

def test_adapted_checkpoint_metadata_preserves_independent_output_head(tmp_path):
    c=MT5Config(vocab_size=32,d_model=16,d_kv=8,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,tie_word_embeddings=True)
    m=OriginalMT5(c).eval()
    with torch.no_grad():m.lm_head.weight.copy_(torch.randn_like(m.lm_head.weight))
    assert m.lm_head.weight is not m.shared.weight
    m.save_pretrained(tmp_path)
    n,info=OriginalMT5.from_pretrained(tmp_path,config=c,output_loading_info=True);n.eval()
    assert not info['missing_keys'] and not info['unexpected_keys']
    assert n.encoder.embed_tokens.weight is n.shared.weight is n.decoder.embed_tokens.weight
    assert n.lm_head.weight is not n.shared.weight
    x=torch.tensor([[4,5,1]])
    with torch.no_grad():torch.testing.assert_close(m(input_ids=x,labels=x).logits,n(input_ids=x,labels=x).logits,atol=0,rtol=0)

def test_new_evaluation_anchors_are_excluded_from_ordinary_generation_inputs():
    cases=json.loads((ROOT/'data/evaluation/restored_pipeline_v1/inputs.json').read_text())
    assert len(cases)==24 and len({r['item_id'] for r in cases})==24
    for row in cases:
        changed={**row,'diagnostic_anchor':'HIDDEN ORACLE DRAFT'}
        assert source(row,'STANDARD')==source(changed,'STANDARD')
        assert source(row,'LOCAL')==source(changed,'LOCAL')
        assert 'EXPLICIT CHOSEN DRAFT' in source(changed,'LOCAL_REWRITE','EXPLICIT CHOSEN DRAFT')
        assert 'HIDDEN ORACLE DRAFT' not in source(changed,'LOCAL_REWRITE','EXPLICIT CHOSEN DRAFT')
