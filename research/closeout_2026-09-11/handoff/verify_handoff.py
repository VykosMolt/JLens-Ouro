#!/usr/bin/env python3
"""Verify an extracted JLens_Ouro_Writer_Handoff_v2 directory without any file outside it: rehash the manifest,
rebuild the primary and 20 secondary contrasts from the included compact data with the included frozen code, and
compare with the included analysis.json. Requires only numpy (+ the standard library).
   python3 verify_handoff.py [handoff_dir]"""
import json, sys, hashlib
from pathlib import Path
H = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
man = json.load(open(H / 'HANDOFF_MANIFEST.json')); bad = []
for e in man['entries']:
    p = H / e['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != e['sha256']: bad.append(e['path'])
print(f"manifest: {man['files']} files, {man['bytes']/1e6:.1f} MB, mismatched/missing: {len(bad)}", bad[:5])
sys.dont_write_bytecode = True; sys.path[:0] = [str(H / '08_code/frozen_evaluation'), str(H / '08_code/frozen_analysis')]
import analyze
A = H / '05_data/accepted_payload'
rerun, arrays = analyze.analyze(A)
saved = json.load(open(H / '01_reports/confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json'))
same = json.loads(json.dumps(rerun, sort_keys=True)) == saved
print('frozen analysis rebuilt from included readouts == included analysis.json:', same)
print('primary:', rerun['estimates'][0], rerun['family_percentile_95_intervals'][0])
rv = json.load(open(H / '02_reviewer_checks/RESULTS.json')); print('reviewer anchor primary:', rv['anchor_primary_reproduced'])
import csv
n_fig = len(list((H / '06_manuscript/figures').glob('fig*.png'))); n_csv = len(list((H / '06_manuscript/figure_data').glob('*.csv')))
print(f'figures: {n_fig} png; figure data csv: {n_csv}; pdf present: {(H / "06_manuscript/v2/JLens_Ouro_Second_Draft.pdf").exists()}')
sys.exit(0 if same and not bad else 1)
