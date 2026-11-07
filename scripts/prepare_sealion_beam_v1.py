"""Freeze decoding-only comparisons; checkpoints and evaluation inputs stay fixed."""
import json,hashlib,tarfile
from pathlib import Path
from replycoach.generate.sealion_v1 import checkpoint_tokenizer
from replycoach.generate.sealion_data import source
from replycoach.generate.sealion_rewrite_v1 import rewrite_source
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def main():
 tok=checkpoint_tokenizer(Path('/tmp/sealion3b_source'));old=read('data/evaluation/sealion_full_rewrite_v1/inputs.json');fresh=read('data/evaluation/restored_pipeline_v1/inputs.json');challenge=read('data/training_expansion/sealion_fidelity_v1/challenge.json')
 max_direct=max(len(tok.encode(source(r,style),add_special_tokens=False)) for r in old+fresh for style in ['STANDARD','LOCAL'])
 max_rewrite=0
 for cases,field in [(old,'actual_anchor'),(old,'diagnostic_anchor'),(fresh,'diagnostic_anchor'),(challenge,'standard_anchor')]:
  for r in cases:
   if field in r:max_rewrite=max(max_rewrite,len(tok.encode(rewrite_source(r,r[field]),add_special_tokens=False)))
 assert max_direct+64<=384 and max_rewrite+64<=384
 # A new English draft is bounded by64 SEA tokens; reserve that plus rewrite scaffolding.
 for r in old+fresh:
  scaffold=len(tok.encode(rewrite_source(r,'x'),add_special_tokens=False))
  if scaffold+64+64>384:raise ValueError('Own-English rewrite context budget')
 vendor=read('src/replycoach/vendor/sealion_v1/provenance.json')
 files=[*ROOT.glob('src/**/*.py'),*[p for p in (ROOT/'src/replycoach/vendor/sealion_v1').iterdir() if p.is_file() and p.suffix!='.pyc'],*[ROOT/p for p in ['scripts/run_sealion_dual_v1.py','scripts/run_sealion_beam_v1.py','scripts/prepare_sealion_beam_v1.py','scripts/v5_sealion_beam_v1.sbatch','tests/test_sealion_beam_v1.py','tests/test_sealion_v1_compat.py','data/evaluation/sealion_full_rewrite_v1/inputs.json','data/evaluation/restored_pipeline_v1/inputs.json','data/evaluation/restored_pipeline_v1/manifest.json','data/training_expansion/sealion_fidelity_v1/challenge.json','artifacts/sealion_full_v1/results.json','artifacts/sealion_full_rewrite_v1/results.json']]]
 files=list(dict.fromkeys(files))
 recipe=dict(experiment='sealion_beam_v1',revision=vendor['revision'],weights_sha256=vendor['weights']['lfs']['sha256'],training_performed=False,adapters_sha256=dict(direct_adapter={'adapter_model.safetensors':'55a7f02626c08894bfdc4b44f68dacf86e8f4edafb7aa6bd78aead9162698913','adapter_config.json':'f5bb26bc63c376e64ed060f54cc87a44ece14a7328f064d02abc487aa75578bc'},rewrite_adapter={'adapter_model.safetensors':'2db1043556b5678683ea33e0f91ddd924ac98b14168ed043dd57b2ce64cf9426','adapter_config.json':'238ae221adf866990a14e822621dae070a067eb00aeb3b27f61bce7e5a33627b'}),methods=dict(greedy=dict(num_beams=1,no_repeat_ngram_size=0),beam4=dict(num_beams=4,no_repeat_ngram_size=0),beam8=dict(num_beams=8,no_repeat_ngram_size=0),beam4_ng3=dict(num_beams=4,no_repeat_ngram_size=3)),common=dict(max_new_tokens=64,do_sample=False,length_penalty=1.,context_limit=384,terminal_rules='EOS or existing transcript turn boundary',beam_algorithm='Top2B cumulative-logprob candidates; up toB active, separateB finished; score=sum logp/generated_length; admissible maximumlength early-stop bound; cap candidates retained and flagged.'),evaluation=dict(reused_conversations=82,fresh_assistant_written=24,fixed_english_rewrites=82,own_english_rewrites=106,diagnostic=32,fresh_diagnostic=24,challenge=18,greedy_exact_replay_required=dict(direct=164,rewrite=132)),budgets=dict(max_direct_prompt=max_direct,max_static_rewrite_prompt=max_rewrite),limitations=['New24 cases are assistant-written development, not independent human gold; no locked test used.','Common fixed English route isolates rendering; ownEnglish route changes bothEnglish andLOCAL decoding.','Beam search optimizes model sequence score, not a correctness metric.','Length-normalized beam score can prefer capped or empty replies; all failures are retained as flags.'],default_changed=False,files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 p=ROOT/'experiments/sealion_beam_v1_recipe.json';p.write_text(json.dumps(recipe,indent=2)+'\n')
 with tarfile.open('/tmp/replycoach_sealion_beam_v1.tgz','w:gz') as tar:
  for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
  tar.add(p,arcname='recipe.json')
 print(json.dumps(dict(files=len(files),budgets=recipe['budgets'],methods=list(recipe['methods']))))
if __name__=='__main__':main()
