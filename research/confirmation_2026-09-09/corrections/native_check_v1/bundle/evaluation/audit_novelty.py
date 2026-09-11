#!/usr/bin/env python3
"""Blinded lexical overlap evidence; semantic identity remains an annotation review.

An old word inside a new entity description is not an identity collision. This
script reports literal matches without silently excluding or renaming concepts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
from artifacts import record, json_save
ROUND=Path(__file__).resolve().parents[1]
REPO=ROUND.parents[1]
OLD=ROUND.parent/'refit_round_2026-09-07'
sys.dont_write_bytecode=True
sys.path[:0]=[str(REPO),'/home/moloch/ouro_project/src']

def words(text):return re.findall(r"[a-z0-9]+",text.casefold())
def grams(text,n=8):
    seq=words(text);return {tuple(seq[i:i+n]) for i in range(len(seq)-n+1)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.benchmark.read_text())
    oldpath=REPO/'data/evaluations/lens-eval-multihop.json';olditems=json.loads(oldpath.read_text())['items']
    from ouro_jlens.evaldata import surface_forms
    oldlabels=sorted({v for row in olditems for v in row['intermediates']})
    oldforms={form.casefold():label for label in oldlabels for form in surface_forms(label)}
    concepts=data['concepts'];literal=[]
    for concept in concepts:
        for alias in [concept['intermediate'],*concept['semantic_aliases']]:
            if alias.casefold() in oldforms:literal.append({'new_concept':concept['intermediate'],'alias':alias,'old_label':oldforms[alias.casefold()]})
    discovery_pairs=[]
    for row in data['items']:
        new=grams(row['prompt']);tokens=set(words(row['prompt']))
        for old in olditems:
            shared=new&grams(old['prompt']);other=set(words(old['prompt']));jaccard=len(tokens&other)/len(tokens|other)
            if shared or jaccard>=.6:
                discovery_pairs.append({'new':row['name'],'discovery':old['name'],'shared_8grams':[' '.join(g) for g in sorted(shared)],'word_set_jaccard':jaccard})
    supplement=json.loads((ROUND/'artifacts/RETAINED_INPUTS_SUPPLEMENT.json').read_text())
    paths=sorted({entry['calibration_path'] for entry in supplement['upstream'].values()})
    texts=[];calibration_records={}
    for path in paths:
        rows=json.loads(Path(path).read_text());calibration_records[path]=record(path)
        for i,text in enumerate(rows):texts.append((path,i,text,grams(text)))
    prompt_matches=[];label_mentions=[]
    for row in data['items']:
        new=grams(row['prompt'])
        for path,i,text,oldgrams in texts:
            shared=new&oldgrams
            if shared:prompt_matches.append({'item':row['name'],'calibration_file':path,'paragraph':i,'shared_8grams':[' '.join(g) for g in sorted(shared)]})
    for concept in concepts:
        for alias in [concept['intermediate'],*concept['semantic_aliases']]:
            pattern=re.compile(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',re.I)
            for path,i,text,_ in texts:
                match=pattern.search(text)
                if match:label_mentions.append({'concept':concept['intermediate'],'matched_alias':alias,'calibration_file':path,'paragraph':i,'context':text[max(0,match.start()-75):match.end()+75]})
    output={'schema':'confirmation_blind_lexical_novelty.v1','new_confirmation_results_exposed':False,'benchmark_record':record(args.benchmark),
            'discovery_record':record(oldpath),'all_discovery_intermediate_labels':oldlabels,
            'exact_discovery_surface_form_collisions':literal,'discovery_prompt_similarity_flags':discovery_pairs,
            'calibration_input_records':calibration_records,'calibration_prompt_8gram_matches':prompt_matches,
            'calibration_literal_concept_mentions':label_mentions,
            'interpretation':'Literal flags for blinded semantic/factual review; shared words, labels or final answers alone do not establish the same fact or entity. No automated eligibility exclusions or new-pretraining-absence claim.'}
    json_save(args.out,output)
    print(json.dumps({'literal_old_forms':len(literal),'discovery_prompt_flags':len(discovery_pairs),'calibration_prompt_matches':len(prompt_matches),'calibration_concept_mentions':len(label_mentions)}))

if __name__=='__main__':main()
