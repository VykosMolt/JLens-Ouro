"""Freeze five independent calibration samples without fitting any model."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
PROJECT=Path('/home/moloch/ouro_project')
ART=PROJECT/'artifacts/jlens'
SNAPSHOT=PROJECT/'artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd'
INV=json.loads((HERE/'inventory.json').read_text())
PARQUET=Path(INV['parquet']['path'])


def digest(data):return hashlib.sha256(data).hexdigest()
def encode_hash(ids):return digest(json.dumps(ids,separators=(',',':')).encode())


def main():
    old=json.loads((ART/'data/wikitext_prompts_b08601e.json').read_text())
    old_hash={digest(x.encode()) for x in old}
    records=[json.loads(line) for line in (HERE/'cached_pool_index.jsonl').read_text().splitlines()]
    pool=[r for r in records if r['text_sha256'] not in old_hash and not r['copied_eval_prompt']]
    assert len(pool)==202772
    seeds=list(range(2026090701,2026090706))
    orders=[np.random.default_rng(seed).permutation(len(pool)) for seed in seeds]
    # Read the first 256 candidates for each independently randomized list.
    # The loop extends deterministically if prefix collisions require more.
    texts={}
    def retrieve(wanted):
        offset=0
        for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=8192,columns=['text']):
            if any(offset<=row<offset+len(batch) for row in wanted):
                for j,text in enumerate(batch.column(0).to_pylist()):
                    row=offset+j
                    if row in wanted:texts[row]=text
            offset+=len(batch)
    token=AutoTokenizer.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    token_ids=lambda text:[token.bos_token_id,*token(text,truncation=True,max_length=127).input_ids]
    old_tokens=[encode_hash(token_ids(text)) for text in old]
    old_map=json.loads((HERE/'original_calibration_article_map.json').read_text())
    tasks=json.loads((ART/'retrieved/jlens-b300-20260905-0359/eval/n100_exit3/items.json').read_text())
    norm=lambda text:' '.join(text.lower().split())
    task_prompts=[norm(item['prompt']) for item in tasks]
    fits=[]
    for number,(seed,order) in enumerate(zip(seeds,orders),1):
        accepted=[];replacements=[];used=set();cursor=0
        while len(accepted)<100:
            block=order[cursor:cursor+256]
            retrieve({pool[int(i)]['row'] for i in block if pool[int(i)]['row'] not in texts})
            for candidate in block:
                record=pool[int(candidate)];text=texts[record['row']];cursor+=1
                assert digest(text.encode())==record['text_sha256']
                assert record['text_sha256'] not in old_hash
                assert not any(prompt in norm(text) for prompt in task_prompts)
                ids=token_ids(text);key=encode_hash(ids)
                if key in used:
                    replacements.append({'candidate_rank_one_based':cursor,'row':record['row'],'reason':'duplicate_T128_model_input','token_prefix_sha256':key})
                    continue
                used.add(key)
                accepted.append({**record,'candidate_rank_one_based':cursor,'token_prefix_sha256':key,
                                 'token_length_including_bos':len(ids),'valid_positions_skip16':len(ids)-17})
                if len(accepted)==100:break
        path=HERE/f'calibration_fit_{number:02d}.json'
        payload=(json.dumps([texts[r['row']] for r in accepted],ensure_ascii=False,indent=2)+'\n').encode()
        if path.exists():assert path.read_bytes()==payload
        else:path.write_bytes(payload)
        fits.append({'fit':number,'seed':seed,'n':100,'file':str(path),'file_sha256':digest(payload),
                     'distinct_articles':len({r['article_row'] for r in accepted}),
                     'replacement_count':len(replacements),'replacements':replacements,'rows':accepted})
    def overlap(field):
        sets=[{r[field] for r in fit['rows']} for fit in fits]
        return [[len(a&b) for b in sets] for a in sets]
    plan={'status':'FROZEN_CALIBRATION_ONLY_NO_FITS','n_fits':5,'n_prompts_per_fit':100,
          'policy':'Independently seeded simple random samples without replacement within fit; between-fit overlap retained. Duplicate truncated model inputs within fit replaced by next candidate in that fit permutation.',
          'calibration_distribution_warning':'Fresh random paragraphs from the broad cached training shard differ in coverage from the historical deterministic 100-paragraph prefix covering 12 articles.',
          'cached_parquet':INV['parquet'],'pool_index_sha256':INV['pool_manifest']['sha256'],
          'unique_pool_before_exclusions':len(records),'historical_exact_texts_excluded':len(old_hash),
          'copied_eval_rows_excluded':sum(r['copied_eval_prompt'] for r in records),'eligible_sampling_pool':len(pool),
          'historical_calibration_file_sha256':INV['calibration_json']['pinned_file_sha256'],
          'tokenizer_snapshot':str(SNAPSHOT),'tokenizer_json_sha256':digest((SNAPSHOT/'tokenizer.json').read_bytes()),
          'model_input_rule':'tokenizer(text, truncation=True, max_length=127).input_ids, then prepend BOS; no model forward.',
          'fit_estimator_requested':{'target_virtual':191,'max_seq_len':128,'skip_first':16,'reduction':'stock_sum_over_valid_target_positions_then_mean_valid_source_positions','source_coverage':'root to freeze after preflight; no fitted matrices produced here'},
          'numpy_version':np.__version__,'generator_sha256':digest(Path(__file__).read_bytes()),'fits':fits,
          'between_fit_overlap':{name:overlap(field) for name,field in [('exact_text','text_sha256'),('T128_model_input','token_prefix_sha256'),('article','article_row')]},
          'historical_overlap':[{'fit':f['fit'],'exact_text_first1200':len({r['text_sha256'] for r in f['rows']}&old_hash),
                                'T128_model_input_first1200':len({r['token_prefix_sha256'] for r in f['rows']}&set(old_tokens)),
                                'article_first100':len({r['article_row'] for r in f['rows']}&{r['article_row'] for r in old_map[:100]}),
                                'article_first1200':len({r['article_row'] for r in f['rows']}&{r['article_row'] for r in old_map})} for f in fits]}
    path=HERE/'calibration_plan.json';payload=json.dumps(plan,indent=2)+'\n'
    if path.exists():assert path.read_text()==payload
    else:path.write_text(payload)
    print(json.dumps({k:plan[k] for k in ['status','eligible_sampling_pool','between_fit_overlap','historical_overlap']},indent=2))
    print('distinct_articles',[f['distinct_articles'] for f in fits],'replacement_counts',[f['replacement_count'] for f in fits])


if __name__=='__main__':main()
