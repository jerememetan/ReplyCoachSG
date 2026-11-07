import copy,json,sys
from hashlib import sha256
from pathlib import Path
import pytest
from replycoach.generate.teacher_v5 import load_teacher_targets

def fixture(tmp_path):
    source=tmp_path/'train.jsonl';source.write_text('frozen source bytes\n')
    rows=[dict(item_id='T1',context=['Your book is here.'],speakers=['user'],message='Can you bring my book?',style=s,response='Original') for s in ('STANDARD','LOCAL')]
    c=dict(item_id='T1',context=rows[0]['context'],speakers=rows[0]['speakers'],message=rows[0]['message'],
           replies={'standard':"I'll bring your book.",'local':"Can, I'll bring your book."},
           review=dict(approved=True,reviewer='assistant',meaning_preserved=True,speaker_correct=True,natural_local=True,context_compatible=True))
    report=dict(schema_version=1,status='audited',source_train_sha256=sha256(source.read_bytes()).hexdigest(),teacher_settings={'model_digest':'frozen'},items=[c])
    path=tmp_path/'teacher.json';return source,rows,report,path

def load(f):
    source,rows,report,path=f;path.write_text(json.dumps(report));return load_teacher_targets(path,rows,source)

def test_only_targets_change_and_actual_roles_preserved(tmp_path):
    f=fixture(tmp_path);rows,manifest=load(f)
    assert len(rows)==2 and rows[0]['response']=="I'll bring your book."
    assert rows[0]['context']==f[1][0]['context'] and rows[0]['speakers']==['user']
    assert manifest['items']==1

@pytest.mark.parametrize('change',['source','message','roles','id','duplicate','not_audited','approval','missing_style'])
def test_wrong_source_or_unaudited_targets_rejected(tmp_path,change):
    f=fixture(tmp_path);r=f[2];c=r['items'][0]
    if change=='source':r['source_train_sha256']='wrong'
    elif change=='message':c['message']='Other input'
    elif change=='roles':c['speakers']=['peer']
    elif change=='id':c['item_id']='VALIDATION-ITEM'
    elif change=='duplicate':r['items'].append(copy.deepcopy(c))
    elif change=='not_audited':r['status']='generated_pending_audit'
    elif change=='approval':c['review']['approved']=False
    elif change=='missing_style':del c['replies']['local']
    with pytest.raises(ValueError):load(f)

def test_teacher_never_sees_authored_targets_or_reviewer_labels():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from generate_teacher_targets import teacher_payload
    item=dict(session_memory=[{'speaker':'user','text':'Hi'}],message={'text':'Hello'},replies={'standard':'secret'},intent='secret',reviewer_notes='secret')
    assert teacher_payload(item)=={'conversation':[['user','Hi']],'incoming_peer_message':'Hello'}
