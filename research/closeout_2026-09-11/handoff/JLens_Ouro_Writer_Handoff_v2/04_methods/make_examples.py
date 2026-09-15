#!/usr/bin/env python3
"""Real ID-selected benchmark examples with token IDs (selection rule: the three fixed numerical-sample items 0,1,2 used by the
frozen scorer readouts.SAMPLE_ITEMS, plus the first zero-drop boundary item and the W-collision item confirmation-027)."""
import json, csv
from pathlib import Path
import transformers
C = Path('/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09')
A = C / 'cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results'
snap = (C / '../verification_2026-09-11/preservation/bundle/model_snapshot').resolve()
tok = transformers.AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True, local_files_only=True)
pop = json.load(open(A / 'population.json')); rows = pop['rows']; bench = json.load(open(A / 'benchmark.json')); by = {it['name']: it for it in bench['items']}
bc = {b['name']: b['dropped_prompt_tokens'] for b in pop['boundary_checks']}
first_zero = next(r['name'] for r in rows if bc[r['name']] == 0)
names = ['confirmation-001', 'confirmation-002', 'confirmation-003', first_zero, 'confirmation-027']
out, md = [], ['# Real benchmark examples (selection rule stated; no example chosen by outcome)\n', 'Selection: the three fixed numerical-sample items of the frozen scorer (`readouts.SAMPLE_ITEMS = (0, 1, 2)` = confirmation-001..003), the first item in ID order with a zero-token boundary drop, and the W-collision item confirmation-027. Token IDs are the exact model input (`token_ids`); the first ID is BOS. Readout position is the last token.\n']
for nm in names:
    r = next(x for x in rows if x['name'] == nm); it = by[nm]; ids = r['token_ids']; nmn = r['intermediates'][0]
    forms = {f: tok.decode([f]) for f in r['intermediate_tokens'][nmn]}
    e = {'item': nm, 'domain': next(c['domain'] for c in bench['concepts'] if c['intermediate'] == nmn), 'dependency_group': r['dependency_group_id'], 'prompt': r['prompt'], 'target': r['target'], 'intermediate': nmn,
         'scored_token_ids_and_forms': forms, 'token_ids': ids, 'tokens': [tok.decode([t]) for t in ids], 'readout_token': r['readout_token'], 'dropped_prompt_tokens_at_boundary': bc[nm], 'n_tokens': r['n_tokens'],
         'relation_type': it['relation_type'], 'relation_status': it['relation_type_status'], 'n_controls': len(r['control_indices'][0])}
    out.append(e)
    md += [f"## {nm} ({e['domain']}, {e['dependency_group']})\n", f"- Prompt: `{r['prompt']}`", f"- Target (answer, never shown to the model): `{r['target']}`", f"- Annotated intermediate: **{nmn}**; scored single-token forms (ID → text): {', '.join(f'{k} → `{v}`' for k, v in forms.items())}",
           f"- Model input ({e['n_tokens']} tokens incl. BOS; boundary drop {bc[nm]}): IDs `{ids}`", f"- Decoded tokens: {' | '.join(repr(t) for t in e['tokens'])}", f"- Readout token (position −1): `{r['readout_token']}`; controls: {e['n_controls']} other names; relation: {it['relation_type']} ({it['relation_type_status']})\n"]
Path(__file__).with_name('EXAMPLES.md').write_text('\n'.join(md)); Path(__file__).with_name('examples.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print('\n'.join(md[:12]))
