"""Evaluate the frozen shared ranker using existing development proposals and ratings."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from judge_replies import load_packet, validate_rating
from replycoach.evaluation import DEFAULT_BANK, DEFAULT_QUERIES, load_evaluation_inputs, validate_output_path
from replycoach.io import read_jsonl, ROOT
from replycoach.selection_replay import replay_selection, validate_replay_inputs
from replycoach.shared_ranker import CrossEncoderScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--bank', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.check_only and args.out_dir is None:
        parser.error('--out-dir is required unless --check-only is used')
    public, mapping, hashes = load_packet(args.packet)
    inputs = load_evaluation_inputs(args.queries, args.bank)
    query_items = {item.id: item for item in inputs.queries.items}
    manifest_path = args.packet / 'ratings/run.json'
    ratings_path = args.packet / 'ratings/ratings.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['packet_hashes'] != hashes or manifest.get('calibration'):
        raise ValueError('replay needs the unchanged completed main evaluation packet')
    run_id = sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    ratings = read_jsonl(ratings_path)
    for row in ratings:
        if row['run_id'] != run_id:
            raise ValueError('rating run identity differs from the frozen manifest')
        validate_rating(row['rating'])
    for row in public:
        meta = mapping[row['candidate_id']]
        item = query_items.get(meta['item_id'])
        if (item is None or meta['bank_fingerprint'] != inputs.bank.fingerprint
                or meta['queries_fingerprint'] != inputs.queries.fingerprint
                or row['incoming_message']['text'] != item.message.text
                or row['conversation'] != [{'speaker': t.speaker, 'text': t.text} for t in item.session_memory]):
            raise ValueError('packet differs from the current reviewed development inputs')
        source = ROOT / meta['source_export']
        if sha256(source.read_bytes()).hexdigest() != meta['source_export_sha256']:
            raise ValueError('source export hash has changed')
    if {m['item_id'] for m in mapping.values()} != set(query_items):
        raise ValueError('packet must cover every current validation query')
    personas = [p['id'] for p in manifest['personas']]
    validate_replay_inputs(public, mapping, ratings, personas)
    protected = (args.queries, args.bank, manifest_path, ratings_path,
                 args.packet/'judge_inputs.jsonl', args.packet/'PRIVATE_mapping.jsonl')
    if args.out_dir is not None:
        validate_output_path(args.out_dir, protected)
    if args.check_only:
        print(json.dumps({'status': 'validated', 'conversations': len(query_items),
                          'proposals': len(public), 'existing_ratings': len(ratings), 'model_calls': 0}))
        return
    started = perf_counter()
    scorer = CrossEncoderScorer()
    model_load_seconds = perf_counter() - started
    result = replay_selection(public, mapping, ratings, personas, scorer)
    result.update(packet_hashes=hashes, rating_run_id=run_id, model_load_seconds=model_load_seconds,
                  ratings_sha256=sha256(ratings_path.read_bytes()).hexdigest(),
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir/'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (args.out_dir/'summary.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['summary'][0]))
        writer.writeheader()
        writer.writerows(result['summary'])
    print(json.dumps({'status': 'complete', 'conversations': len(query_items),
                      'new_judge_calls': 0, 'summary': result['summary'], 'output': str(args.out_dir)}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'Cannot replay selection: {exc}')
