"""Read-only inventory of retained fits and locally cached calibration rows.

No network, model forward, gradients, or historical mutations. New audit outputs
are written only beside this script. Run with ouro_project/venv/bin/python.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/moloch/ouro_project')
ART = PROJECT/'artifacts/jlens'
PARQUET = PROJECT/'artifacts/hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/train-00000-of-00002.parquet'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<22),b''):
            h.update(chunk)
    return h.hexdigest()


def raw_rows():
    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=8192,columns=['text']):
        yield from batch.column(0).to_pylist()


def main():
    old=json.loads((ART/'data/wikitext_prompts.json').read_text())
    pinned=json.loads((ART/'data/wikitext_prompts_b08601e.json').read_text())
    assert old==pinned and len(pinned)==1200
    items=json.loads((ART/'retrieved/jlens-b300-20260905-0359/eval/n100_exit3/items.json').read_text())
    normalize=lambda text:' '.join(text.lower().split())
    # Exact complete task prompts after case/whitespace normalization. This is
    # a copied-prompt audit, not an attempt to remove all factual/task vocabulary.
    eval_patterns={normalize(item['prompt']):item['name'] for item in items}
    copied_pattern=re.compile('|'.join(re.escape(x) for x in eval_patterns))
    iterator=iter(raw_rows())
    previous=''
    current=next(iterator)
    article=None
    article_row=None
    index=eligible=duplicates=0
    seen={}
    article_counts=Counter()
    first1200=[]
    exact_copies=[]
    manifest=HERE/'cached_pool_index.jsonl'
    with manifest.open('w') as stream:
        while current is not None:
            following=next(iterator,None)
            # Blank delimiters are essential: hockey table definitions include
            # '= Saves ; Sv % =' and resemble titles without this check.
            stripped=current.strip()
            if (not previous.strip() and (following is None or not following.strip())
                    and re.fullmatch(r'= [^=].* =',stripped)):
                article=stripped[2:-2]
                article_row=index
            if len(stripped)>=600:
                text_sha=hashlib.sha256(current.encode()).hexdigest()
                if eligible<1200:
                    assert current==pinned[eligible]
                    first1200.append({'calibration_index':eligible,'parquet_row':index,
                                      'article':article,'article_row':article_row,'text_sha256':text_sha})
                eligible+=1
                if text_sha in seen:
                    duplicates+=1
                else:
                    seen[text_sha]=index
                    match=copied_pattern.search(normalize(current))
                    if match:
                        exact_copies.append({'parquet_row':index,'task_name':eval_patterns[match.group()],
                                             'text_sha256':text_sha})
                    entry={'row':index,'text_sha256':text_sha,'article_row':article_row,
                           'characters':len(current),'copied_eval_prompt':bool(match)}
                    stream.write(json.dumps(entry,separators=(',',':'))+'\n')
                    article_counts[article_row]+=1
            previous,current=current,following
            index+=1
    assert index==pq.ParquetFile(PARQUET).metadata.num_rows
    assert len(first1200)==1200
    metadata=[]
    for root in (ART/'lens',ART/'retrieved'):
        for path in sorted(root.rglob('*.json')):
            if '.receipts/' in str(path):continue
            data=json.loads(path.read_text())
            if not isinstance(data,dict) or 'target_ut' not in data:continue
            entry={'path':str(path),'binary_exists':path.with_suffix('.pt').exists(),
                   **{key:data.get(key) for key in ('kind','target_ut','target_virtual','n_prompts','n_fitted','start','end',
                                                   'max_seq_len','skip_first','dim_batch','checkpoint_every','prompt_file_sha256',
                                                   'prompt_slice_sha256','generator_sha256','seconds')}}
            entry['source_count']=len(data.get('source_layers',[])) or data.get('n_sources')
            entry['source_range']=[min(data['source_layers']),max(data['source_layers'])] if data.get('source_layers') else None
            entry['n_shards']=len(data.get('shard_records',data.get('shards',[])))
            entry['has_seed']=any('seed' in key for key in data)
            metadata.append(entry)
    original_articles={row['article_row'] for row in first1200[:100]}
    shards={}
    for start,stop in ((0,8),(8,32),(32,56),(56,80),(80,100)):
        counts=Counter(row['article'] for row in first1200[start:stop])
        shards[f'{start}:{stop}']={'n_prompts':stop-start,'n_articles':len(counts),'articles':dict(counts)}
    output={'parquet':{'path':str(PARQUET),'bytes':PARQUET.stat().st_size,'sha256':sha(PARQUET),
                       'rows':index,'eligible_rows_minimum600':eligible,'unique_eligible_texts':len(seen),
                       'duplicate_eligible_records':duplicates,'eligible_article_headings':len(article_counts),
                       'unmapped_eligible_unique_rows':article_counts.get(None,0)},
            'pool_manifest':{'path':str(manifest),'sha256':sha(manifest),'rows':len(seen)},
            'calibration_json':{'old_equals_pinned':True,'n':len(pinned),
                                'old_file_sha256':sha(ART/'data/wikitext_prompts.json'),
                                'pinned_file_sha256':sha(ART/'data/wikitext_prompts_b08601e.json'),
                                'unique_texts':len(set(pinned)),'old_first100_articles':len(original_articles),
                                'remaining_pool_without_original_articles':sum(row['article_row'] not in original_articles for row in first1200[100:])},
            'copied_eval_prompts':exact_copies,'original_shards':shards,'fit_metadata':metadata,
            'search_boundaries':'All lens JSON/PT and checkpoint-like files under artifacts/jlens plus /tmp/claude-1000 were searched; no per-prompt Jacobians/checkpoint sums found.'}
    (HERE/'inventory.json').write_text(json.dumps(output,indent=2)+'\n')
    (HERE/'original_calibration_article_map.json').write_text(json.dumps(first1200,indent=2)+'\n')
    print(json.dumps({key:output[key] for key in ('parquet','calibration_json','copied_eval_prompts','original_shards')},indent=2))


if __name__=='__main__':
    main()
