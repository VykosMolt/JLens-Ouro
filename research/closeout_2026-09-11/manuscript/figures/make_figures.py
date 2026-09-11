#!/usr/bin/env python3
"""Render manuscript figures from the exported arrays only (no new statistics)."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; D = HERE.parent / 'figure_data'; L = HERE.parents[1] / 'local_exit'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
C_FIT, C_RAW, C_LOCAL, C_FINAL, BAND, THIRD = '#1f5fa8', '#c8552f', '#2a9d4a', '#1f5fa8', '#f2d16b', '#e6e6e6'
def rd(name): return list(csv.DictReader(open(D / name)))
def save(fig, stem):
    for ext in ('png', 'pdf', 'svg'): fig.savefig(HERE / f'{stem}.{ext}', bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
def shade(ax, band=True, third=False):
    if third: ax.axvspan(32.5, 48.5, color=THIRD, lw=0, zorder=0)
    if band: ax.axvspan(25.5, 37.5, color=BAND, alpha=0.55, lw=0, zorder=0)
    ax.axhline(0, color='k', lw=0.6)

# F1: five-fit discovery curves (multihop), all four passes
rows = rd('discovery_five_fit_curves.csv')
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for p, ax in enumerate(axes, 1):
    r = [x for x in rows if int(x['pass']) == p]; x = [int(v['physical']) for v in r]
    if p == 4: shade(ax, True, True)
    else: ax.axhline(0, color='k', lw=0.6)
    ax.fill_between(x, [float(v['crossed_fit_item_pointwise_low']) for v in r], [float(v['crossed_fit_item_pointwise_high']) for v in r], color=C_FIT, alpha=0.18, lw=0, label='crossed fit/item pointwise 95%')
    for k in range(1, 6): ax.plot(x, [float(v[f'fit{k}']) for v in r], color=C_FIT, lw=0.6, alpha=0.6)
    ax.plot(x, [float(v['fit_mean']) for v in r], color=C_FIT, lw=1.6, label='five-fit mean')
    if p == 4: ax.plot([48], [float(r[-1]['fit_mean'])], marker='o', mfc='white', mec='k', ms=4, lw=0, label='identity endpoint (virtual 191)')
    ax.set_title(f'pass {p}'); ax.set_xlabel('physical layer'); ax.set_xlim(1, 48)
axes[0].set_ylabel('J-Lens − raw excess hit@10\n(discovery, 90 items)'); axes[3].legend(fontsize=7, loc='upper left', frameon=False)
fig.suptitle('Discovery population: five independent N100 final-target fits (fit01–fit05) minus raw, fixed layers. Yellow: band 26–37 selected here and later frozen; grey: final third 33–48.', fontsize=8, y=1.04)
save(fig, 'fig1_discovery_five_fit_curves')

# F2: confirmation curves: paired excess per layer with descriptive group band (top), own recovery (bottom)
pe = rd('confirmation_paired_excess_by_layer.csv'); cv = rd('confirmation_layer_curves.csv')
fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex='col', sharey='row')
for p in range(1, 5):
    r = [x for x in pe if int(x['pass']) == p]; c = [x for x in cv if int(x['pass']) == p]; x = [int(v['physical']) for v in r]
    ax = axes[0, p - 1]; shade(ax, p == 4, False)
    ax.fill_between(x, [float(v['group_boot_2.5_descriptive']) for v in r], [float(v['group_boot_97.5_descriptive']) for v in r], color=C_FIT, alpha=0.18, lw=0, label='group-bootstrap pointwise 95% (descriptive)')
    ax.plot(x, [float(v['fit01_minus_raw']) for v in r], color=C_FIT, lw=1.5, label='fit01 − raw'); ax.plot(x, [float(v['fit02_minus_raw']) for v in r], color=C_FIT, lw=0.8, ls='--', label='fit02 − raw')
    if p == 4: ax.plot([48], [float(r[-1]['fit01_minus_raw'])], marker='o', mfc='white', mec='k', ms=4, lw=0)
    ax.set_title(f'pass {p}')
    ax = axes[1, p - 1]; shade(ax, p == 4, False)
    ax.plot(x, [float(v['fit01_own']) for v in c], color=C_FIT, lw=1.5, label='fit01 intended recovery'); ax.plot(x, [float(v['raw_own']) for v in c], color=C_RAW, lw=1.5, label='raw intended recovery')
    ax.plot(x, [float(v['fit01_control']) for v in c], color=C_FIT, lw=0.8, ls=':', label='fit01 control recovery'); ax.plot(x, [float(v['raw_control']) for v in c], color=C_RAW, lw=0.8, ls=':', label='raw control recovery')
    ax.set_xlabel('physical layer'); ax.set_xlim(1, 48)
axes[0, 0].set_ylabel('paired excess difference\n(160 items)'); axes[1, 0].set_ylabel('hit@10 rate'); axes[0, 3].legend(fontsize=7, frameon=False, loc='upper left'); axes[1, 3].legend(fontsize=7, frameon=False, loc='upper left')
fig.suptitle('Confirmation population (160 new items): fixed-layer curves. Yellow: the frozen primary band, pass 4 layers 26–37. Curves are descriptive; only the band mean was inferential.', fontsize=8, y=1.0)
save(fig, 'fig2_confirmation_curves')

# F3: same band in all four passes
sb = rd('same_band_by_pass.csv')
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
ax = axes[0]; ps = [int(r['pass']) for r in sb]; est = [float(r['paired_excess_diff']) for r in sb]; lo = [float(r['interval_low']) for r in sb]; hi = [float(r['interval_high']) for r in sb]
ax.axhline(0, color='k', lw=0.6)
ax.errorbar(ps[:3], est[:3], yerr=[np.subtract(est[:3], lo[:3]), np.subtract(hi[:3], est[:3])], fmt='o', color=C_FIT, capsize=3, label='passes 1–3: post-confirmation, simultaneous 95% (family N1)')
ax.errorbar(ps[3:], est[3:], yerr=[np.subtract(est[3:], lo[3:]), np.subtract(hi[3:], est[3:])], fmt='s', color='k', capsize=3, label='pass 4: original prospective primary, group percentile 95%')
ax.set_xticks(ps); ax.set_xlabel('pass'); ax.set_ylabel('fit01 − raw excess hit@10,\nphysical layers 26–37'); ax.legend(fontsize=6.5, frameon=False, loc='lower left')
ax = axes[1]; w = 0.35
ax.bar(np.array(ps) - w / 2, [float(r['fit01_intended']) for r in sb], w, color=C_FIT, label='fit01 intended'); ax.bar(np.array(ps) + w / 2, [float(r['raw_intended']) for r in sb], w, color=C_RAW, label='raw intended')
ax.set_xticks(ps); ax.set_xlabel('pass'); ax.set_ylabel('intended recovery, layers 26–37'); ax.legend(fontsize=7, frameon=False)
fig.suptitle('Same physical band in every pass (confirmation population). Control recovery is below 0.004 in every cell.', fontsize=8)
save(fig, 'fig3_same_band_by_pass')

# F4: contrast forest: estimator controls (original simultaneous family) + within-domain (N1)
sec = {r['id']: r for r in rd('confirmation_secondary_contrasts.csv')}; rv = json.load(open(HERE.parents[1] / 'reviewer_checks/RESULTS.json'))
items = [('fit01 − raw (primary, all 79 controls)', 0.23188291139240508, rv['A_within_domain']['paired']['excess_all79_diff_ORIGINAL_PRIMARY']['original_group_percentile_95'], 'k'),
         ('fit01 − raw, within-domain controls (7) [N1]', rv['family_N1']['contrasts']['A_within_domain_excess_diff_pass4']['estimate'], rv['family_N1']['contrasts']['A_within_domain_excess_diff_pass4']['simultaneous_95_maxt'], C_FIT)]
for k, lab in (('fit02_local', 'fit02 − raw'), ('fit01_minus_fit02_local', 'fit01 − fit02'), ('penultimate_local', 'penultimate target − raw'), ('sampled_sum_local', 'sampled-sum − raw'), ('diagonal_local', 'diagonal − raw'), ('fit01_minus_penultimate_local', 'fit01 − penultimate target'), ('sampled_sum_minus_diagonal_local', 'sampled-sum − diagonal')):
    s = sec[k]; items.append((lab + ' [original family]', float(s['estimate']), [float(s['simultaneous_low']), float(s['simultaneous_high'])], '#555555'))
fig, ax = plt.subplots(figsize=(7.2, 3.6)); y = np.arange(len(items))[::-1]
for yi, (lab, e, iv, col) in zip(y, items): ax.errorbar([e], [yi], xerr=[[e - iv[0]], [iv[1] - e]], fmt='o', color=col, capsize=3)
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=7.5); ax.set_xlabel('difference in excess hit@10, pass 4 layers 26–37')
fig.suptitle('Pass-4 band contrasts. Black: prospective primary (percentile). Grey: original 20-contrast simultaneous family. Blue: post-confirmation within-domain control (family N1).', fontsize=7.5)
save(fig, 'fig4_band_contrasts')

# F5: local-exit descriptive (discovery population)
dj = json.load(open(L / 'DISCOVERY_POPULATION_FIXED_BAND.json')); cur = list(csv.DictReader(open(L / 'discovery_population_curves.csv')))
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0), gridspec_kw={'wspace': 0.35})
for p in (1, 2, 3):
    ax = axes[p - 1]; r = [x for x in cur if int(x['pass']) == p]; x = [int(v['physical']) for v in r]; shade(ax)
    ax.plot(x, [float(v['raw']) for v in r], color=C_RAW, lw=1.4, label='raw'); ax.plot(x, [float(v['final']) for v in r], color=C_FINAL, lw=1.4, label='J-Lens, final-pass target (exit3)')
    loc = [float(v[f'local{p}']) if v[f'local{p}'] else np.nan for v in r]; ax.plot(x, loc, color=C_LOCAL, lw=1.6, label=f"J-Lens, pass-{p} own exit (exit{p-1})")
    ax.set_title(f'pass {p}'); ax.set_xlabel('physical layer'); ax.set_xlim(1, 48)
axes[0].set_ylabel('excess hit@10 (discovery, 90 items)'); axes[0].legend(fontsize=6.5, frameon=False, loc='upper left')
ax = axes[3]; ax.axhline(0, color='k', lw=0.6)
for j, (k, col, lab) in enumerate((('local_minus_final', C_LOCAL, 'local − final'), ('local_minus_raw', '#7a4fa8', 'local − raw'), ('final_minus_raw', C_FINAL, 'final − raw'))):
    for p in (1, 2, 3):
        row = dj['passes'][p - 1][k]; e = row['excess_diff']; iv = row['concept_cluster_bootstrap_95_descriptive']
        ax.errorbar([p + (j - 1) * 0.22], [e], yerr=[[e - iv[0]], [iv[1] - e]], fmt='o', color=col, capsize=2.5, ms=4, label=lab if p == 1 else None)
ax.set_xticks([1, 2, 3]); ax.set_xlabel('pass'); ax.set_ylabel('band 26–37 difference'); ax.legend(fontsize=6.5, frameon=False, loc='lower left'); ax.set_title('band means, cluster 95% (descr.)', fontsize=9)
fig.suptitle('Application-era matched local/final banks (same recipe, target differs), evaluated on the DISCOVERY population. Descriptive and post hoc; not the confirmation population.', fontsize=7.5, y=1.04)
save(fig, 'fig5_local_exit_discovery_descriptive')
print('figures:', sorted(p.name for p in HERE.glob('fig*.png')))
