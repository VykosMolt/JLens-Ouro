#!/usr/bin/env python3
"""Figures in the Kirin visual system (Documents/Research/paper_style/figure_style.py). Data only from the exported arrays
in ../figure_data and ../../local_exit; no new statistics. Vector PDF for the paper, PNG for the markdown preview."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/moloch/Documents/Research')
from paper_style import figure_style as st
st.apply(paper=1)
HERE = Path(__file__).resolve().parent; D = HERE.parent / 'figure_data'; L = HERE.parents[1] / 'local_exit'
# gentle-red scheme: J-Lens in a muted brick red, raw lens in the baseline grey, own-exit lens in teal, contrasts in violet
RED, RED_DARK, RED_PALE = '#B0544F', '#7E3835', '#F7ECEA'
JL, RAW, LOCAL, FINAL = RED, st.BASELINE, st.POSITIVE, RED
BAND, THIRD = RED_PALE, '#F1F3F5'
mpl_cycle = __import__('matplotlib').cycler(color=[RED, st.POSITIVE, st.VIOLET, st.BASELINE]); __import__('matplotlib').rcParams['axes.prop_cycle'] = mpl_cycle
def rd(name): return list(csv.DictReader(open(D / name)))
def save(fig, stem):
    st.save(fig, HERE / f'{stem}.pdf'); fig.savefig(HERE / f'{stem}.png', dpi=220, bbox_inches='tight', pad_inches=0.03); plt.close(fig); print('wrote', stem)
def band(ax, third=False):
    if third: ax.axvspan(32.5, 48.5, color=THIRD, lw=0, zorder=0)
    ax.axvspan(25.5, 37.5, color=BAND, lw=0, zorder=0)
def zero(ax): ax.axhline(0, color=st.INK, lw=0.6, zorder=1)
def passaxes(ax, p, ylabel=None):
    ax.set_xlim(0.4, 48.9); ax.set_xticks([1, 12, 24, 36, 48]); ax.set_title(f'({"abcd"[p-1]}) pass {p}', loc='left'); ax.set_xlabel('physical layer')
    if ylabel: ax.set_ylabel(ylabel)

# ---- Figure 1: discovery, five independent fits minus raw, all passes ----
rows = rd('discovery_five_fit_curves.csv')
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
for p, ax in enumerate(axes, 1):
    r = [x for x in rows if int(x['pass']) == p]; x = [int(v['physical']) for v in r]
    band(ax, third=(p == 4)); zero(ax)
    ax.fill_between(x, [float(v['crossed_fit_item_pointwise_low']) for v in r], [float(v['crossed_fit_item_pointwise_high']) for v in r], color=JL, alpha=0.16, lw=0, label='crossed fit/item pointwise 95%')
    for k in range(1, 6): ax.plot(x, [float(v[f'fit{k}']) for v in r], color=JL, lw=0.55, alpha=0.55, label='individual fits (5)' if k == 1 else None)
    ax.plot(x, [float(v['fit_mean']) for v in r], color=JL, lw=1.7, label='five-fit mean')
    if p == 4: ax.plot([48], [float(r[-1]['fit_mean'])], marker='o', mfc='white', mec=st.INK, ms=4.2, lw=0, label='identity endpoint', clip_on=False)
    passaxes(ax, p)
axes[0].set_ylabel('J-Lens − raw, excess hit@10')
h, l = axes[3].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=7.6)
fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.30, wspace=0.12); save(fig, 'fig1_discovery')

# ---- Figure 2: confirmation curves ----
pe = rd('confirmation_paired_excess_by_layer.csv'); cv = rd('confirmation_layer_curves.csv')
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex='col', sharey='row')
for p in range(1, 5):
    r = [x for x in pe if int(x['pass']) == p]; c = [x for x in cv if int(x['pass']) == p]; x = [int(v['physical']) for v in r]
    ax = axes[0, p - 1]; (band(ax) if p == 4 else None); zero(ax)
    ax.fill_between(x, [float(v['group_boot_2.5_descriptive']) for v in r], [float(v['group_boot_97.5_descriptive']) for v in r], color=JL, alpha=0.16, lw=0, label='group-bootstrap pointwise 95% (descriptive)')
    ax.plot(x, [float(v['fit01_minus_raw']) for v in r], color=JL, lw=1.6, label='fit01 − raw'); ax.plot(x, [float(v['fit02_minus_raw']) for v in r], color=JL, lw=0.8, ls='--', label='fit02 − raw')
    if p == 4: ax.plot([48], [float(r[-1]['fit01_minus_raw'])], marker='o', mfc='white', mec=st.INK, ms=4.2, lw=0, clip_on=False)
    ax.set_title(f'({"abcd"[p-1]}) pass {p}', loc='left'); ax.set_xlim(0.4, 48.9); ax.set_xticks([1, 12, 24, 36, 48])
    ax = axes[1, p - 1]; (band(ax) if p == 4 else None)
    ax.plot(x, [float(v['fit01_own']) for v in c], color=JL, lw=1.6, label='J-Lens fit01, intended'); ax.plot(x, [float(v['raw_own']) for v in c], color=RAW, lw=1.6, label='raw lens, intended')
    ax.plot(x, [float(v['fit01_control']) for v in c], color=JL, lw=0.8, ls=':', label='J-Lens, controls'); ax.plot(x, [float(v['raw_control']) for v in c], color=RAW, lw=0.8, ls=':', label='raw, controls')
    ax.set_xlabel('physical layer'); ax.set_xlim(0.4, 48.9); ax.set_xticks([1, 12, 24, 36, 48]); ax.set_ylim(-0.02, 0.66)
axes[0, 0].set_ylabel('paired excess difference'); axes[1, 0].set_ylabel('hit@10 rate')
h1, l1 = axes[0, 3].get_legend_handles_labels(); h2, l2 = axes[1, 3].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=7.4)
fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.2, wspace=0.12, hspace=0.30); save(fig, 'fig2_confirmation')

# ---- Figure 3: the same band in every pass ----
sb = rd('same_band_by_pass.csv'); ps = [int(r['pass']) for r in sb]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={'width_ratios': [1.15, 1]})
ax = axes[0]; zero(ax)
for r in sb:
    p = int(r['pass']); e = float(r['paired_excess_diff']); lo, hi = float(r['interval_low']), float(r['interval_high'])
    col = RED_DARK if p == 4 else JL; mk = 's' if p == 4 else 'o'
    ax.errorbar([p], [e], yerr=[[e - lo], [hi - e]], fmt=mk, color=col, mfc=(col if p == 4 else 'white'), mec=col, mew=1.4, capsize=3, ms=5.5, lw=1.4)
    ax.annotate(f'{e:+.3f}', (p, e), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8, color=st.INK)
ax.set_xticks(ps); ax.set_xlim(0.5, 4.8); ax.set_xlabel('pass'); ax.set_ylabel('fit01 − raw, excess hit@10\nlayers 26–37')
ax.set_title('(a) paired difference in the same band', loc='left')
ax = axes[1]; w = 0.36
b1 = ax.bar(np.array(ps) - w / 2, [float(r['fit01_intended']) for r in sb], w, color=JL, label='J-Lens fit01'); b2 = ax.bar(np.array(ps) + w / 2, [float(r['raw_intended']) for r in sb], w, color=RAW, label='raw lens')
st.bar_label(ax, b1, fmt='{:.3f}', dy=0.006, fontsize=7.2); st.bar_label(ax, b2, fmt='{:.3f}', dy=0.006, fontsize=7.2)
ax.set_xticks(ps); ax.set_xlabel('pass'); ax.set_ylabel('intended recovery, layers 26–37'); ax.set_ylim(0, 0.45); ax.set_title('(b) intended-concept recovery', loc='left'); ax.legend(loc='upper left')
fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.2, wspace=0.32); save(fig, 'fig3_same_band')

# ---- Figure 4: band contrasts (forest) ----
sec = {r['id']: r for r in rd('confirmation_secondary_contrasts.csv')}; rv = json.load(open(HERE.parents[1] / 'reviewer_checks/RESULTS.json'))
items = [('fit01 − raw, 79 controls (primary)', 0.23188291139240508, rv['A_within_domain']['paired']['excess_all79_diff_ORIGINAL_PRIMARY']['original_group_percentile_95'], st.INK, 's'),
         ('fit01 − raw, 7 within-domain controls', rv['family_N1']['contrasts']['A_within_domain_excess_diff_pass4']['estimate'], rv['family_N1']['contrasts']['A_within_domain_excess_diff_pass4']['simultaneous_95_maxt'], st.POSITIVE, 'D')]
for k, lab in (('fit02_local', 'fit02 − raw'), ('fit01_minus_fit02_local', 'fit01 − fit02'), ('penultimate_local', 'penultimate target − raw'), ('sampled_sum_local', 'sampled-sum − raw'), ('diagonal_local', 'diagonal − raw'), ('fit01_minus_penultimate_local', 'fit01 − penultimate target'), ('sampled_sum_minus_diagonal_local', 'sampled-sum − diagonal')):
    s = sec[k]; items.append((lab, float(s['estimate']), [float(s['simultaneous_low']), float(s['simultaneous_high'])], JL, 'o'))
fig, ax = plt.subplots(figsize=(7.2, 3.3)); y = np.arange(len(items))[::-1]
ax.axvline(0, color=st.INK, lw=0.6)
for yi, (lab, e, iv, col, mk) in zip(y, items):
    ax.errorbar([e], [yi], xerr=[[e - iv[0]], [iv[1] - e]], fmt=mk, color=col, capsize=2.5, ms=5, lw=1.3)
    ax.annotate(f'{e:+.3f}', (iv[1], yi), xytext=(6, 0), textcoords='offset points', va='center', fontsize=7.6, color=st.INK)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=8.2); ax.set_xlabel('difference in excess hit@10, pass 4, layers 26–37'); ax.set_xlim(-0.05, 0.42); ax.grid(axis='y', visible=False)
ax.plot([], [], 's', color=st.INK, label='prospective primary (percentile 95%)'); ax.plot([], [], 'o', color=JL, label='original 20-contrast simultaneous family'); ax.plot([], [], 'D', color=st.POSITIVE, label='post-confirmation family N1 (simultaneous)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.3); ax.set_xlim(-0.05, 0.46); fig.subplots_adjust(left=0.36, right=0.98, top=0.97, bottom=0.27); save(fig, 'fig4_contrasts')

# ---- Figure 5: local-exit targets on the discovery population (descriptive) ----
dj = json.load(open(L / 'DISCOVERY_POPULATION_FIXED_BAND.json')); cur = list(csv.DictReader(open(L / 'discovery_population_curves.csv')))
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45), gridspec_kw={'width_ratios': [1, 1, 1, 1.15], 'wspace': 0.5})
for p in (1, 2, 3):
    ax = axes[p - 1]; r = [x for x in cur if int(x['pass']) == p]; x = [int(v['physical']) for v in r]; band(ax); zero(ax)
    ax.plot(x, [float(v['raw']) for v in r], color=RAW, lw=1.5, label='raw lens'); ax.plot(x, [float(v['final']) for v in r], color=FINAL, lw=1.5, label='J-Lens, final-pass target')
    ax.plot(x, [float(v[f'local{p}']) if v[f'local{p}'] else np.nan for v in r], color=LOCAL, lw=1.7, label="J-Lens, the pass's own exit")
    passaxes(ax, p); ax.set_ylim(-0.05, 0.42)
axes[0].set_ylabel('excess hit@10 (90 items)')
ax = axes[3]; zero(ax)
for j, (k, col, lab, mk) in enumerate((('local_minus_final', LOCAL, 'own exit − final target', 'o'), ('local_minus_raw', st.VIOLET, 'own exit − raw', 'D'), ('final_minus_raw', FINAL, 'final target − raw', 's'))):
    for p in (1, 2, 3):
        row = dj['passes'][p - 1][k]; e = row['excess_diff']; iv = row['concept_cluster_bootstrap_95_descriptive']
        ax.errorbar([p + (j - 1) * 0.24], [e], yerr=[[e - iv[0]], [iv[1] - e]], fmt=mk, color=col, capsize=2.2, ms=4.2, lw=1.2, label=lab if p == 1 else None)
ax.set_xticks([1, 2, 3]); ax.set_xlim(0.5, 3.5); ax.set_xlabel('pass'); ax.set_ylabel('band difference', labelpad=2); ax.set_title('(d) band 26–37', loc='left')
h, l = axes[0].get_legend_handles_labels(); h2, l2 = ax.get_legend_handles_labels()
fig.legend(h + h2, l + l2, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7.3)
fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.36); save(fig, 'fig5_local_exit')
