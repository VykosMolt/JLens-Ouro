#!/usr/bin/env python3
"""Build the anonymous TMLR supplementary zip from the writer handoff package.

Copies the reviewer-relevant records, data, frozen code and figure data; rewrites machine paths and
person/account identifiers in text files; refuses to package if any identifying string survives;
writes README_SUPPLEMENT.md, verify_supplement.py and MANIFEST.json; zips; then extracts the zip into a
temporary directory, runs the verifier there and rescans. TMLR allows at most 100 MB (PDF or ZIP)."""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

M = Path(__file__).resolve().parents[1]                      # manuscript/
H = M.parents[0] / 'handoff' / 'JLens_Ouro_Writer_Handoff_v2'  # closeout_2026-09-11/handoff/...
STAGE = M / 'tmlr' / 'supplement_stage' / 'JLens_Ouro_supplement'
ZIP = M / 'tmlr' / 'JLens_Ouro_TMLR_supplement.zip'
PY = sys.executable

# (handoff source, destination, exclude substrings)
OPERATIONAL = ['RUN_LOG.md', 'PURGE_MANIFEST', 'RECOVERY_LEDGER', 'preservation/', 'spending/', 'resources/', 'cloud_leases/',
               'monitoring/', 'BUDGET.md', 'COST_REVISION.md', 'GPU_AND_BUDGET', 'PROGRESS.md', 'RESEARCH_LOG.md']
PLAN = [
    ('01_reports/confirmation_2026-09-09', 'reports/confirmation_2026-09-09', OPERATIONAL),
    ('01_reports/verification_2026-09-11', 'reports/verification_2026-09-11', OPERATIONAL),
    ('01_reports/refit_round_2026-09-07', 'reports/refit_round_2026-09-07', OPERATIONAL),
    ('01_reports/followup_2026-09-07', 'reports/followup_2026-09-07', OPERATIONAL),
    ('01_reports/application_era', 'reports/initial_study', OPERATIONAL),
    ('02_reviewer_checks', 'reviewer_checks', []),
    ('03_local_exit', 'local_exit', []),
    ('04_methods', 'methods', []),
    ('05_data', 'data', []),
    ('08_code', 'code', []),
    ('06_manuscript/figure_data', 'manuscript/figure_data', []),
    ('06_manuscript/ledgers', 'manuscript/ledgers', []),
]
DROP_RENDERS_UNDER = 'reports/'   # .png renders of data that is present as csv/json
TEXT_SUFFIXES = {'.md', '.json', '.csv', '.py', '.txt', '.stdout', '.log', '.tex', '.svg', '.toml', '.yaml', '.yml', '.lock', '.sh', '.cfg', '.ini', ''}
SUBS = [  # order matters
    (re.compile(r'/home/moloch'), '/home/user'),
    (re.compile(r'moloch@arch-legion|arch-legion'), 'workstation'),
    (re.compile(r'moloch'), 'user'),
    (re.compile(r'models--Vykos--'), 'models--[anonymized]--'),
    (re.compile(r'VykosMolt/JLens-Ouro'), '[anonymized]/JLens-Ouro'),
    (re.compile(r'Vykos/ouro-jlens-results'), '[anonymized]/ouro-jlens-results'),
    (re.compile(r'VykosMolt|Vykos'), '[anonymized]'),
    (re.compile(r"Jan Kirin's|Jan's"), "the author's"),
    (re.compile(r'Jan Kirin|Kirin|\bJan\b'), 'the author'),
    (re.compile(r'NEEL_NOTE'), 'MENTOR_NOTE'),
    (re.compile(r'Neel Nanda|Neel'), '[mentor]'),
    (re.compile(r'\bMATS\b'), '[program]'),
    (re.compile(r'illjaesterhazy@gmail\.com'), '[email]'),
]
FORBIDDEN = re.compile(r'Kirin|\bJan\b|moloch|Vykos|illja|esterhazy|arch-legion|\bMATS\b|Neel', re.I)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def redact(text):
    for rx, rep in SUBS: text = rx.sub(rep, text)
    return text

def stage():
    if STAGE.parent.exists(): shutil.rmtree(STAGE.parent)
    STAGE.mkdir(parents=True)
    n_text = n_bin = 0
    for src_rel, dst_rel, excl in PLAN:
        src = H / src_rel
        for p in sorted(src.rglob('*')):
            if not p.is_file(): continue
            rel = p.relative_to(src).as_posix()
            if any(x in rel or x in (src_rel + '/' + rel) for x in excl): continue
            dst = STAGE / dst_rel / rel
            if dst_rel.startswith(DROP_RENDERS_UNDER) and p.suffix.lower() == '.png': continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix.lower() in TEXT_SUFFIXES:
                try: t = p.read_text(encoding='utf-8')
                except UnicodeDecodeError: shutil.copy2(p, dst); n_bin += 1; continue
                dst.write_text(redact(t), encoding='utf-8'); n_text += 1
            else: shutil.copy2(p, dst); n_bin += 1
    ae = STAGE / 'data' / 'application_era'
    if ae.exists(): ae.rename(STAGE / 'data' / 'initial_study')
    figs = STAGE / 'manuscript' / 'figures'; figs.mkdir(parents=True)
    for f in sorted((M / 'figures_v3').glob('fig*.pdf')): shutil.copy2(f, figs / f.name); n_bin += 1
    return n_text, n_bin

README = """# Supplementary material: Final-Target J-Lens in Ouro

Anonymous supplementary package for double-blind review. Every file is a copy of a project record; nothing here was
produced for the submission except this README, `verify_supplement.py` and `MANIFEST.json`. Machine paths were rewritten
to `/home/user/...`, and account, repository and person identifiers were replaced by bracketed placeholders. Multi-gigabyte
artifacts (lens banks, activation states, model weights) are not included; their sizes and SHA-256 hashes are recorded in
the provenance, freeze and inventory files. The paper calls the first round of work the initial study; the record files of that period call it the "application era" (`application_era` in file names and text). The model is `ByteDance/Ouro-2.6B` at the revision named in `data/accepted_payload/run_spec.json`.

## Verify

    python3 verify_supplement.py        # needs only numpy

Rehashes every file against `MANIFEST.json`, then rebuilds the primary endpoint and all twenty secondary contrasts of the
paper from the included compact readouts (`data/accepted_payload`) with the included frozen analysis code
(`code/frozen_evaluation`, `code/frozen_analysis`) and compares the result with the saved `analysis.json`.

## Where each part of the paper is supported

| Paper | Files |
|---|---|
| Sections 3, 4.2, Appendix C: model, estimators, prompts, scoring, fitting recipe, freeze | `reports/confirmation_2026-09-09/PROSPECTIVE_PLAN.md`, `FREEZE.json`, `corrections/`, `data/accepted_payload/run_spec.json`, `benchmark.json`, `population.json`, `methods/METHODS_COMPLETION.md`, `methods/EXAMPLES.md`, `code/frozen_evaluation`, `code/frozen_ouro_jlens`, `code/frozen_jlens` |
| Section 4.1, Figure 1: discovery and independent refits | `reports/followup_2026-09-07/`, `reports/refit_round_2026-09-07/`, `data/refit/`, `data/followup/`, `manuscript/figure_data/discovery_five_fit_curves.csv` |
| Sections 5.1-5.3, Tables 1-4, Figures 2-4: confirmation results | `reports/confirmation_2026-09-09/REPORT.md`, `CLAIMS.md`, `results/.../analysis/analysis.json`, `data/accepted_payload/readouts/*.npz`, `manuscript/figure_data/` |
| Section 5.4, Table 3, Figures 3-4: post-confirmation reviewer checks | `reviewer_checks/ANALYSIS_PLAN.md` (frozen before computation), `REVIEWER_CHECKS.md`, `RESULTS.json`, `PER_ITEM.csv`, `OVERLAP_LEDGER.csv`, `GROUP_MEMBERSHIP.csv`, `run_checks.py` |
| Section 6, Table 5, Appendix B: numerical verification | `reports/verification_2026-09-11/FINAL_VERIFICATION.md`, `report/`, `metrics/VERIFICATION_METRICS.json`, `checker/`, `reviews/`, `RUN_SPECIFICATION.json` |
| Section 7, Figure 5: local-exit targets on the discovery population | `local_exit/LOCAL_EXIT_STATUS.md`, `LOCAL_EXIT_BANK_INVENTORY.json`, `DISCOVERY_POPULATION_FIXED_BAND.json`, `discovery_population_curves.csv`, `data/initial_study/`, `reports/initial_study/` |
| Appendix A: exploratory findings and Huginn pilot | `reports/followup_2026-09-07/results/`, `probe/`, `reports/refit_round_2026-09-07/analysis/HUGINN_COMPARISON_CONTRACT.md`, `data/refit/huginn_run01/` |
| Every number in the paper | `manuscript/ledgers/SOURCE_TO_MANUSCRIPT_LEDGER.md` (value -> record path), `SOURCE_VALUES.json` |
| Figures | `manuscript/figures/*.pdf` (as in the paper), `manuscript/figure_data/*.csv` |

Record paths inside the ledgers and reports refer to the original project layout (`research/<round>/...`); the same files
are here under `reports/<round>/`, `data/`, `reviewer_checks/`, `local_exit/` and `methods/`.
"""

VERIFY = '''#!/usr/bin/env python3
"""Verify this supplementary package without any file outside it: rehash MANIFEST.json, then rebuild the
primary and 20 secondary contrasts from data/accepted_payload with the included frozen code and compare
with the included analysis.json. Requires numpy.      python3 verify_supplement.py [dir]"""
import json, sys, hashlib
from pathlib import Path
H = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
man = json.load(open(H / 'MANIFEST.json')); bad = []
for e in man['entries']:
    p = H / e['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != e['sha256']: bad.append(e['path'])
print(f"manifest: {man['files']} files, {man['bytes']/1e6:.1f} MB, mismatched/missing: {len(bad)}", bad[:5])
sys.dont_write_bytecode = True; sys.path[:0] = [str(H / 'code/frozen_evaluation'), str(H / 'code/frozen_analysis')]
import analyze
rerun, arrays = analyze.analyze(H / 'data/accepted_payload')
saved = json.load(open(H / 'reports/confirmation_2026-09-09/results/ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json'))
same = json.loads(json.dumps(rerun, sort_keys=True)) == saved
print('frozen analysis rebuilt from included readouts == included analysis.json:', same)
print('primary:', rerun['estimates'][0], rerun['family_percentile_95_intervals'][0])
rv = json.load(open(H / 'reviewer_checks/RESULTS.json')); print('reviewer anchor primary:', rv['anchor_primary_reproduced'])
sys.exit(0 if same and not bad else 1)
'''

def main():
    n_text, n_bin = stage()
    (STAGE / 'README_SUPPLEMENT.md').write_text(README, encoding='utf-8')
    (STAGE / 'verify_supplement.py').write_text(VERIFY, encoding='utf-8')
    # identity scan of every text file
    leaks = []
    for p in sorted(STAGE.rglob('*')):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            try: t = p.read_text(encoding='utf-8')
            except UnicodeDecodeError: continue
            for m in FORBIDDEN.finditer(t): leaks.append((p.relative_to(STAGE).as_posix(), t[max(0, m.start()-30):m.end()+30].replace('\n', ' ')))
    if leaks:
        print('IDENTIFYING STRINGS REMAIN:'); [print('  ', f, '|', c) for f, c in leaks[:20]]; sys.exit(1)
    entries = [{'path': p.relative_to(STAGE).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(STAGE.rglob('*')) if p.is_file()]
    (STAGE / 'MANIFEST.json').write_text(json.dumps({'schema': 'tmlr_supplement_manifest.v1', 'files': len(entries),
                                                     'bytes': sum(e['bytes'] for e in entries), 'entries': entries}, indent=1))
    if ZIP.exists(): ZIP.unlink()
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():
                zi = zipfile.ZipInfo(('JLens_Ouro_supplement/' + p.relative_to(STAGE).as_posix()), date_time=(2026, 9, 12, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
                z.writestr(zi, p.read_bytes())
    size = ZIP.stat().st_size
    # independent test: extract, verify, rescan
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(ZIP) as z:
            assert all(not (i.filename.startswith('/') or '..' in i.filename) for i in z.infolist()); z.extractall(td)
        root = Path(td) / 'JLens_Ouro_supplement'
        r = subprocess.run([PY, '-B', 'verify_supplement.py'], cwd=root, capture_output=True, text=True)
        print(r.stdout.strip()); 
        if r.returncode != 0: print(r.stderr[-2000:]); sys.exit('verifier failed on the extracted zip')
        rescan = [str(p) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and FORBIDDEN.search(p.read_text(encoding='utf-8', errors='ignore'))]
        assert not rescan, rescan
    print(f'zip: {ZIP} {size:,} bytes ({size/1e6:.1f} MB; TMLR limit 100 MB); files {len(entries)} ({n_text} text redacted, {n_bin} binary); sha256 {sha(ZIP)}')
    if size > 100_000_000: sys.exit('zip exceeds the TMLR limit')

if __name__ == '__main__': main()
