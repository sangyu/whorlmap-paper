"""Regenerate all panel SVGs for Figure 1. Run from the nbs/ directory.

Outputs:
  images/panel_ai.svg      — Gardner-Altman dabest plot (new Ai)
  images/panel_a.svg       — whorl cell + histogram (Aii, Aiii)
  images/panel_b.svg       — cell taxonomy, 5 scenarios (Bi–Bv)
  images/panel_ci.svg      — GTEx whorlmap only (Ci)
  images/panel_cii.svg     — GTEx scalar heatmap only (Cii)
"""
import matplotlib
matplotlib.use('Agg')
import pathlib, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import (LinearSegmentedColormap, Normalize,
                                TwoSlopeNorm)
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import seaborn as sns
import dabest
from dabest.multi import combine

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
IMAGES = pathlib.Path('images')
IMAGES.mkdir(exist_ok=True)

# ── Shared colourmaps ─────────────────────────────────────────────────────────
CMAP_SWOR = LinearSegmentedColormap.from_list(
    'steelwhiteorangered',
    [(0.00, (70/255, 130/255, 180/255)),
     (0.50, (1.0,   1.0,   1.0  )),
     (0.75, (1.0,   0.55,  0.0  )),
     (1.00, (0.86,  0.08,  0.24 ))],
    N=256
)
CMAP  = LinearSegmentedColormap.from_list(
    'steelwhiteorangered2',
    ['steelblue', 'white', 'orangered'], N=256
)
DNORM = TwoSlopeNorm(vcenter=0, vmin=-3.5, vmax=3.5)


# ── Spiral helpers (used in panel A) ─────────────────────────────────────────
def _spiralize(fill, m, n):
    i = 0; j = 0; k = 0; array = np.zeros((m, n))
    while m > 0 and k < len(fill):
        jj = j; ii = i
        for j in range(j, n):
            if k >= len(fill): break
            array[i, j] = fill[k]; k += 1
        for i in range(ii + 1, m):
            if k >= len(fill): break
            array[i, j] = fill[k]; k += 1
        for j in range(n - 2, jj - 1, -1):
            if k >= len(fill): break
            array[i, j] = fill[k]; k += 1
        for i in range(m - 2, ii, -1):
            if k >= len(fill): break
            array[i, j] = fill[k]; k += 1
        m -= 1; n -= 1; j += 1
    return array

def _spiral_path(n):
    path = []
    top, bot, lft, rgt = 0, n - 1, 0, n - 1
    while top <= bot and lft <= rgt:
        for c in range(lft, rgt + 1): path.append((top, c))
        top += 1
        for r in range(top, bot + 1): path.append((r, rgt))
        rgt -= 1
        if top <= bot:
            for c in range(rgt, lft - 1, -1): path.append((bot, c))
            bot -= 1
        if lft <= rgt:
            for r in range(bot, top - 1, -1): path.append((r, lft))
            lft += 1
    return path

def _ring_structure(n):
    rings, pos, k = [], 0, 0
    while True:
        sz = n - 2 * k
        if sz <= 0: break
        nc = 1 if sz == 1 else 4 * (sz - 1)
        rings.append({'k': k, 'start': pos, 'end': pos + nc, 'sz': sz})
        pos += nc; k += 1
        if sz == 1: break
    return rings


# ── Shared bootstrap data (used for both Ai and A) ───────────────────────────
_rng    = np.random.default_rng(42)
noise_a = _rng.normal(0, 3., 20); noise_b = _rng.normal(0, 3., 20)
group_a = noise_a - noise_a.mean()
group_b = noise_b - noise_b.mean() + 0.5
_df_a   = pd.DataFrame({'group': ['A'] * 20 + ['B'] * 20,
                         'value': np.concatenate([group_a, group_b])})
_dobj_a = dabest.load(_df_a, x='group', y='value', idx=('A', 'B'))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel Ai — Gardner-Altman plot
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating panel_ai …")
fig_ai = _dobj_a.mean_diff.plot(
    fig_size=(4.5, 5),
    float_contrast=True,   # classic Gardner-Altman floating contrast
    show_pairs=False,
)
fig_ai.savefig(IMAGES / 'panel_ai.svg', bbox_inches='tight')
fig_ai.savefig(IMAGES / 'panel_ai.png', dpi=600, bbox_inches='tight')
plt.close(fig_ai)
print("  saved panel_ai.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A (Aii + Aiii) — whorl cell + histogram, no suptitle
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating panel_a (Aii+Aiii) …")
_multi_a = combine([[_dobj_a]], ['gene'], row_labels=['region'],
                   effect_size='mean_diff')
bs_raw   = sorted(_multi_a.bootstraps[0])
chop     = int(np.ceil(len(bs_raw) * 0.025))
bs_c     = bs_raw[chop:-chop]
n        = 7; N = n * n
ranks    = np.linspace(0, len(bs_c), N, dtype=int); ranks[0] = 1
fv       = np.array([bs_c[r - 1] for r in ranks])
if sum(v > 0 for v in bs_c) < len(bs_c) / 2:
    fv = fv[::-1]

XLIM     = (-3.5, 3.5)
rings    = _ring_structure(n)
fv_sorted  = np.clip(np.sort(fv), *XLIM)
cell_vals  = _spiralize(fv.tolist(), n, n)
bound      = max(2.0, np.ceil(max(abs(fv_sorted[0]), abs(fv_sorted[-1])) * 2) / 2)
VALUE_NORM = Normalize(vmin=-bound, vmax=bound)
spiral_pos = _spiral_path(n)

fig = plt.figure(figsize=(11, 5.4))
ax_cell = fig.add_axes([0.03, 0.18, 0.32, 0.68])
ax_cbar = fig.add_axes([0.03, 0.08, 0.32, 0.05])
ax_hist = fig.add_axes([0.43, 0.12, 0.54, 0.74])

ax_cell.imshow(cell_vals, cmap=CMAP_SWOR, norm=VALUE_NORM,
               origin='upper', interpolation='nearest', aspect='equal')

def _ring_arrows(ax, ring_k, sz, avg_val):
    mid = ring_k + sz // 2
    bg  = CMAP_SWOR(VALUE_NORM(avg_val))
    col = 'w' if (0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]) < 0.52 else 'k'
    ap  = dict(arrowstyle='->', color=col, lw=1.0, mutation_scale=9)
    d   = 0.25
    ax.annotate('', xy=(mid+0.6, ring_k-d), xytext=(mid-0.6, ring_k-d),
                arrowprops=ap, zorder=8, annotation_clip=False)
    ax.annotate('', xy=(ring_k+sz-1+d, mid+0.6), xytext=(ring_k+sz-1+d, mid-0.6),
                arrowprops=ap, zorder=8, annotation_clip=False)
    ax.annotate('', xy=(mid-0.6, ring_k+sz-1+d), xytext=(mid+0.6, ring_k+sz-1+d),
                arrowprops=ap, zorder=8, annotation_clip=False)
    ax.annotate('', xy=(ring_k-d, mid-0.6), xytext=(ring_k-d, mid+0.6),
                arrowprops=ap, zorder=8, annotation_clip=False)

for ring in rings[:-1]:
    avg_val = float(np.mean(fv_sorted[ring['start']:ring['end']]))
    _ring_arrows(ax_cell, ring['k'], ring['sz'], avg_val)

for k, (r, c) in enumerate(spiral_pos):
    val = float(fv[k])
    bg  = CMAP_SWOR(VALUE_NORM(val))
    lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
    tc  = 'w' if lum < 0.52 else 'k'
    ax_cell.text(c, r, f'q{k}\n{val:.1f}',
                 ha='center', va='center', fontsize=4.5, color=tc,
                 zorder=10, linespacing=1.1, multialignment='center')

ax_cell.set_xlim(-0.5, n - 0.5); ax_cell.set_ylim(n - 0.5, -0.5)
ax_cell.axis('off')
ax_cell.set_title('7×7 whorl cell — outside-in spiral fill\n(pixel colour = effect size)', fontsize=9)

cb = ColorbarBase(ax_cbar, cmap=CMAP_SWOR, norm=VALUE_NORM, orientation='horizontal')
cb_ticks = np.arange(-bound, bound + 0.01, 0.5)
cb.set_ticks(cb_ticks)
cb.set_ticklabels([f'{t:g}' for t in cb_ticks])
cb.ax.tick_params(labelsize=7.5)
cb.set_label('Effect size (bootstrap mean difference)', fontsize=8)

BIN_W  = 0.20
bins   = np.arange(XLIM[0], XLIM[1] + BIN_W, BIN_W)
counts, _ = np.histogram(np.clip(bs_c, *XLIM), bins=bins, density=True)

seg_bounds = np.empty(N + 1)
seg_bounds[0] = XLIM[0]; seg_bounds[-1] = XLIM[1]
for k in range(1, N):
    seg_bounds[k] = (fv_sorted[k - 1] + fv_sorted[k]) / 2.0
seg_cols = [CMAP_SWOR(VALUE_NORM(float(v))) for v in fv_sorted]

best_piece = {}
for lo, hi, ht in zip(bins[:-1], bins[1:], counts):
    if ht == 0: continue
    y_bot = 0.0
    for k in range(N):
        overlap = max(0., min(hi, seg_bounds[k + 1]) - max(lo, seg_bounds[k]))
        if overlap <= 0: continue
        h = ht * overlap / (hi - lo)
        ax_hist.bar(lo, h, width=hi - lo, bottom=y_bot,
                    align='edge', color=seg_cols[k], edgecolor='none')
        if k not in best_piece or h > best_piece[k][2]:
            best_piece[k] = (lo + (hi - lo) / 2, y_bot, h)
        y_bot += h

for k, (xc, yb, h) in best_piece.items():
    if h < counts.max() * 0.005: continue
    val = float(fv_sorted[k])
    col = CMAP_SWOR(VALUE_NORM(val))
    lum = 0.299*col[0] + 0.587*col[1] + 0.114*col[2]
    tc  = 'w' if lum < 0.52 else 'k'
    ax_hist.text(xc, yb + h / 2, f'q{k}\n{val:.1f}',
                 ha='center', va='center', fontsize=3.5, color=tc,
                 zorder=11, linespacing=1.0, multialignment='center')

margin = (fv_sorted[-1] - fv_sorted[0]) * 0.08
ax_hist.set_xlim(fv_sorted[0] - margin, fv_sorted[-1] + margin)
ax_hist.set_ylim(0, counts.max() * 1.03)
ax_hist.spines['left'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.spines['top'].set_visible(False)
ax_hist.set_yticks([])
ax_hist.tick_params(labelsize=8)
ax_hist.set_xlabel('Bootstrap mean difference', fontsize=9)
ax_hist.set_title('Bootstrap distribution, segmented by quantile rank\n'
                  '(bar colours match spiral position on left)', fontsize=9)

# No suptitle
fig.savefig(IMAGES / 'panel_a.svg', bbox_inches='tight')
fig.savefig(IMAGES / 'panel_a.png', dpi=600, bbox_inches='tight')
plt.close(fig)
print("  saved panel_a.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — cell taxonomy (5 scenarios), no suptitle
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating panel_b …")

def _make_1x1_multi(ga, gb):
    df   = pd.DataFrame({'group': ['A'] * len(ga) + ['B'] * len(gb),
                         'value': np.concatenate([ga, gb])})
    dobj = dabest.load(df, x='group', y='value', idx=('A', 'B'))
    return combine([[dobj]], [''], row_labels=[''], effect_size='mean_diff')

_rng2 = np.random.default_rng(1234)
_ga1  = _rng2.normal(0, 1.5, 22);  _gb1 = _rng2.normal( 3.0, 1.5, 22)
_ga2  = _rng2.normal(0, 1.5, 22);  _gb2 = _rng2.normal(-3.0, 1.5, 22)
_ga3  = _rng2.normal(0, 0.4, 5);   _gb3 = _rng2.normal( 0.05, 0.4, 5)
_ga4  = _rng2.normal(0, 0.5,  15); _gb4 = _rng2.normal( 0.3,  6.0,  15)
_ga5  = np.array([0, 0, 0, 0, 0])
_gb5  = np.array([2.1, 2, 2.05, -3, -3.05])

scenarios = [
    ('(i) Large positive,\nnarrow',      _make_1x1_multi(_ga1, _gb1)),
    ('(ii) Large negative,\nnarrow',     _make_1x1_multi(_ga2, _gb2)),
    ('(iii) Near-zero\neffect',          _make_1x1_multi(_ga3, _gb3)),
    ('(iv) Broad / uncertain\n(wide CI)',_make_1x1_multi(_ga4, _gb4)),
    ('(v) Bimodal\n(discrete, N=2)',     _make_1x1_multi(_ga5, _gb5)),
]

XLIM_B = (-5, 5)
BIN_W_B = 0.14
BINS_B  = np.arange(XLIM_B[0], XLIM_B[1] + BIN_W_B, BIN_W_B)
BINCX   = 0.5 * (BINS_B[:-1] + BINS_B[1:])

hists_b = []
for _, multi_s in scenarios:
    bs = np.clip(multi_s.bootstraps[0], *XLIM_B)
    counts_s, _ = np.histogram(bs, bins=BINS_B, density=True)
    hists_b.append(counts_s)
ymax_b = max(h.max() for h in hists_b) * 1.05

fig_b, axes_b = plt.subplots(
    2, 5, figsize=(14, 5),
    gridspec_kw={'hspace': 0.08, 'wspace': 0.35, 'height_ratios': [1.8, 1]}
)

for j, ((label, multi_s), counts_s) in enumerate(zip(scenarios, hists_b)):
    multi_s.whorlmap(
        n=21, chop_tail=2.5, cmap=CMAP, vmin=-3.5, vmax=3.5,
        ax=axes_b[0, j],
        heatmap_kwargs={'cbar': False, 'xticklabels': [''], 'yticklabels': [''],
                        'linewidths': 0},
    )
    axes_b[0, j].set_aspect('equal')
    for _c in axes_b[0, j].collections:
        _c.set_rasterized(True)

    ax = axes_b[1, j]
    cols = CMAP(DNORM(BINCX))
    for lo, hi, ht, c in zip(BINS_B[:-1], BINS_B[1:], counts_s, cols):
        ax.bar(lo, ht, width=hi - lo, align='edge', color=c,
               edgecolor='lightgray', linewidth=.5)
    ax.set_xlim(*XLIM_B)
    ax.set_ylim(0, ymax_b)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([-3, 0, 3])
    ax.tick_params(labelsize=7)
    ax.text(0.5, -0.42, label, transform=ax.transAxes,
            ha='center', fontsize=8.5, va='top', linespacing=1.3)

fig_b.subplots_adjust(top=0.92, right=0.88)
# white spacer so bbox_inches='tight' preserves a top margin for sub-panel labels
fig_b.text(0.5, 0.99, ' ', fontsize=14, va='top', color='white')
cbar_ax_b = fig_b.add_axes([0.90, 0.18, 0.015, 0.60])
sm_b = ScalarMappable(cmap=CMAP, norm=DNORM)
sm_b.set_array([])
cbar_b = fig_b.colorbar(sm_b, cax=cbar_ax_b)
cbar_b.set_label('Effect size', fontsize=8)
cbar_b.set_ticks([-3, 0, 3])
cbar_b.ax.tick_params(labelsize=7)

# No suptitle
fig_b.savefig(IMAGES / 'panel_b.svg', dpi=600, bbox_inches='tight')
fig_b.savefig(IMAGES / 'panel_b.png', dpi=600, bbox_inches='tight')
plt.close(fig_b)
print("  saved panel_b.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# GTEx data rebuild (from pickles)
# ═══════════════════════════════════════════════════════════════════════════════
print("Rebuilding GTEx multi object from pickles …")

BRAIN_REGIONS = {
    "Hypothalamus":      "Brain - Hypothalamus",
    "Amygdala":          "Brain - Amygdala",
    "Hippocampus":       "Brain - Hippocampus",
    "Ant. Cing. Ctx":    "Brain - Anterior cingulate cortex (BA24)",
    "Frontal Cortex":    "Brain - Frontal Cortex (BA9)",
    "Cortex":            "Brain - Cortex",
    "Caudate":           "Brain - Caudate (basal ganglia)",
    "Putamen":           "Brain - Putamen (basal ganglia)",
    "Nucleus Accumbens": "Brain - Nucleus accumbens (basal ganglia)",
    "Cerebellum":        "Brain - Cerebellum",
    "Cerebellar Hemi.":  "Brain - Cerebellar Hemisphere",
    "Substantia Nigra":  "Brain - Substantia nigra",
    "Spinal Cord":       "Brain - Spinal cord (cervical c-1)",
}
LEFT_GENES  = ["RPS4Y1", "DDX3Y", "KDM5D", "XIST"]
RIGHT_GENES = [
    "TPH2", "CHRNA7", "ESR1",
    "TH", "SLC6A3", "DDC", "AGRP",
    "SST", "PENK", "GAD1", "GAD2", "CRH",
    "DRD5", "CHRM1", "BDNF", "CYP19A1",
    "AIF1", "MAOA", "FKBP5", "GFAP",
]
FINAL_GENES  = RIGHT_GENES   # trimmed: left anchor genes dropped
region_order = list(BRAIN_REGIONS.keys())

with open('gtex_data/gene_expr.pkl', 'rb') as f:
    gene_expr = pickle.load(f)
sample_meta = pd.read_csv('gtex_data/sample_meta.csv', index_col=0)

dabest_grid = []
for region in region_order:
    row = []
    for gene in FINAL_GENES:
        gd    = gene_expr[gene]
        mask  = sample_meta['region'] == region
        sids  = sample_meta.index[mask]
        recs  = [{'expr': gd[s], 'sex': sample_meta.loc[s, 'sex']}
                 for s in sids if s in gd]
        df    = pd.DataFrame(recs)
        dobj  = dabest.load(df, x='sex', y='expr', idx=('Male', 'Female'))
        row.append(dobj)
    dabest_grid.append(row)
    print(f"  row: {region}", flush=True)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    multi_gtex = combine(dabest_grid, FINAL_GENES,
                         row_labels=region_order, effect_size='mean_diff')
print("  multi_gtex ready")


# ═══════════════════════════════════════════════════════════════════════════════
# Panels Ci + Cii — identical figure size, same colorbar & colour scale
# ═══════════════════════════════════════════════════════════════════════════════

# Step 1: quick run to get mean_df and set the shared colour scale
print("Computing shared colour scale …")
_fig_tmp, _ax_tmp = plt.subplots()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    _, mean_df = multi_gtex.whorlmap(
        cmap=CMAP, chop_tail=2.5, ax=_ax_tmp,
        heatmap_kwargs={'cbar': False},
    )
plt.close(_fig_tmp)

vabs = 1.5   # fixed colorbar range ±1.5
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle as _Rect
from matplotlib.lines import Line2D as _L2D
from scipy.stats import gaussian_kde as _gkde
shared_norm = Normalize(vmin=-vabs, vmax=vabs)

def _add_shared_cbar(cax, orientation='horizontal'):
    cb = ColorbarBase(cax, cmap=CMAP, norm=shared_norm, orientation=orientation)
    cb.set_label('Mean diff F−M (log₂TPM+1)', fontsize=8)
    cb.set_ticks([-1.5, -0.5, 0, 0.5, 1.5])
    cb.ax.tick_params(labelsize=7)

# ── Highlighted pairs ─────────────────────────────────────────────────────────
# Pair A (blue):   a1=SST/Spinal Cord row=12 col=7,  a2=FKBP5/Putamen row=7 col=18
# Pair B (maroon): b1=AIF1/SN row=11 col=16,         b2=CYP19A1/Caudate row=6 col=15
# Whorlmap: each cell = 21 data units; Cell(r,c) → x=[c*21,(c+1)*21], y=[r*21,(r+1)*21]
_N_COLS_G  = len(RIGHT_GENES)
_PAIR_COL  = ['#1A5276', '#6E2F1A']   # blue=A, maroon=B

# (row, col, pair_idx, within_pair_idx, label)  — display order: a1, a2, [gap], b1, b2
_ORDERED_CELLS = [
    (12,  7, 0, 0, 'a1'),   # SST/SC          → top sparkline
    ( 7, 18, 0, 1, 'a2'),   # FKBP5/Putamen   → 2nd
    (11, 16, 1, 0, 'b1'),   # AIF1/SN         → 3rd (after pair gap)
    ( 6, 15, 1, 1, 'b2'),   # CYP19A1/Caudate → bottom
]

# Bootstrap arrays: pair A=[a1, a2], pair B=[b1, b2]
_pair_bs = {
    0: [np.asarray(multi_gtex.bootstraps[12*_N_COLS_G +  7]),   # a1 SST/SC
        np.asarray(multi_gtex.bootstraps[ 7*_N_COLS_G + 18])],  # a2 FKBP5/Put
    1: [np.asarray(multi_gtex.bootstraps[11*_N_COLS_G + 16]),   # b1 AIF1/SN
        np.asarray(multi_gtex.bootstraps[ 6*_N_COLS_G + 15])],  # b2 CYP19A1/Caud
}
_pair_xlims = {}
_pair_ymax  = {}
for _pi, _blist in _pair_bs.items():
    _lo = min(_b.min() - 0.3*np.std(_b) for _b in _blist)
    _hi = max(_b.max() + 0.3*np.std(_b) for _b in _blist)
    _pair_xlims[_pi] = (_lo, _hi)
    _xr = np.linspace(_lo, _hi, 300)
    _pair_ymax[_pi] = max(_gkde(_b, bw_method='silverman')(_xr).max() for _b in _blist) * 1.1

# ── Shared figure geometry (Ci and Cii use the same dimensions & heatmap rect) ─
FIGSIZE_BOTH = (16.5, 8)
HAX_RECT     = [0.09, 0.18, 0.55, 0.77]
CBAR_RECT    = [0.09, 0.035, 0.275, 0.038]

# Sparklines: a1, a2 | gap | b1, b2 — top to bottom
_H_SPARK    = 0.17;  _SPARK_X = 0.685;  _SPARK_W = 0.26
_SG         = 0.02   # small gap within pair
_LG         = 0.06   # large gap between pairs
_SB         = HAX_RECT[1]   # 0.18 — bottom of heatmap = bottom of bottom sparkline
SPARK_RECTS = [
    [_SPARK_X, _SB + 3*_H_SPARK + 2*_SG + _LG, _SPARK_W, _H_SPARK],  # a1 (top)
    [_SPARK_X, _SB + 2*_H_SPARK + 1*_SG + _LG, _SPARK_W, _H_SPARK],  # a2
    [_SPARK_X, _SB + 1*_H_SPARK + 1*_SG,        _SPARK_W, _H_SPARK],  # b1
    [_SPARK_X, _SB,                              _SPARK_W, _H_SPARK],  # b2 (bottom)
]

# ── Panel Ci — whorlmap with sparkline insets ────────────────────────────────
print("Generating panel_ci …")

fig_ci     = plt.figure(figsize=FIGSIZE_BOTH)
ax_ci      = fig_ci.add_axes(HAX_RECT)
cbar_ax_ci = fig_ci.add_axes(CBAR_RECT)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    _, _ = multi_gtex.whorlmap(
        cmap=CMAP, chop_tail=2.5, vmin=-vabs, vmax=vabs,
        ax=ax_ci, heatmap_kwargs={'cbar': False, 'linewidths': 0},
    )
ax_ci.set_aspect('equal')
for _c in ax_ci.collections:
    _c.set_rasterized(True)
_add_shared_cbar(cbar_ax_ci, orientation='horizontal')

_mn = mean_df.values

for _idx, (_ri, _ci, _pi, _wi, _lbl) in enumerate(_ORDERED_CELLS):
    _mean_val  = float(_mn[_ri, _ci])
    _bs        = _pair_bs[_pi][_wi]
    _hl_col    = _PAIR_COL[_pi]
    _cell_rgba = CMAP(shared_norm(_mean_val))

    # Highlight rectangle on whorlmap
    _cx0, _cy0 = _ci * 21, _ri * 21
    ax_ci.add_patch(_Rect((_cx0, _cy0), 21, 21,
                          fill=False, edgecolor=_hl_col, linewidth=2.0, zorder=12))
    ax_ci.text(_cx0 + 2.5, _cy0 + 5, _lbl,
               fontsize=5.5, fontweight='bold', color=_hl_col, zorder=13,
               va='top', ha='left')

    # Sparkline axes
    _sr  = SPARK_RECTS[_idx]
    _sax = fig_ci.add_axes(_sr)

    _xlo, _xhi = _pair_xlims[_pi]
    _xr = np.linspace(_xlo, _xhi, 300)
    _yr = _gkde(_bs, bw_method='silverman')(_xr)
    _sax.fill_between(_xr, _yr, color=_cell_rgba, alpha=0.65)
    _sax.plot(_xr, _yr, color=_hl_col, lw=0.9)
    _sax.axvline(0,          color='#aaaaaa', lw=0.6, ls=(0,(3,3)))
    _sax.axvline(_mean_val,  color=_hl_col,   lw=1.2)
    _sax.set_xlim(_xlo, _xhi);  _sax.set_ylim(0, _pair_ymax[_pi])
    _sax.set_yticks([]);  _sax.set_xticks([0])
    _sax.set_xticklabels(['0'], fontsize=5, color='#888')
    for _sp in ['top', 'right', 'left', 'bottom']:
        _sax.spines[_sp].set_visible(False)
    _sax.tick_params(length=2, width=0.5, pad=1)
    # Label (top-left) + gene/region caption (below)
    _sax.text(0.02, 0.97, _lbl, transform=_sax.transAxes,
              fontsize=7, fontweight='bold', ha='left', va='top', color=_hl_col)
    _sax.text(0.5, -0.15, f'{RIGHT_GENES[_ci]} · {region_order[_ri]}',
              transform=_sax.transAxes, fontsize=5.5, ha='center', va='top',
              style='italic', color='#444')

# Orthogonal connections: horizontal along row → short segment to sparkline
fig_ci.canvas.draw()

def _d2f(_xd, _yd):
    _disp = ax_ci.transData.transform((_xd, _yd))
    return tuple(fig_ci.transFigure.inverted().transform(_disp))

_xlim_r = ax_ci.get_xlim()[1]

for _idx, (_ri, _ci, _pi, _wi, _lbl) in enumerate(_ORDERED_CELLS):
    _hl_col = _PAIR_COL[_pi]
    _sr = SPARK_RECTS[_idx]
    _p1 = _d2f((_ci + 1) * 21, _ri * 21 + 10.5)   # cell right-middle
    _p2 = _d2f(_xlim_r,         _ri * 21 + 10.5)   # heatmap right edge
    _p3 = (_sr[0], _sr[1] + _sr[3] / 2)             # sparkline left-middle
    for _xs, _ys in [([_p1[0], _p2[0]], [_p1[1], _p2[1]]),
                     ([_p2[0], _p3[0]], [_p2[1], _p3[1]])]:
        fig_ci.add_artist(_L2D(_xs, _ys, transform=fig_ci.transFigure,
                               color=_hl_col, lw=0.85, zorder=15, solid_capstyle='round'))

fig_ci.savefig(IMAGES / 'panel_ci.svg', dpi=600)
fig_ci.savefig(IMAGES / 'panel_ci.png', dpi=600)
plt.close(fig_ci)
print("  saved panel_ci.svg")

# ── Panel Cii — scalar heatmap, same geometry as Ci, with matching labels ────
print("Generating panel_cii …")

fig_cii     = plt.figure(figsize=FIGSIZE_BOTH)
ax_cii      = fig_cii.add_axes(HAX_RECT)
cbar_ax_cii = fig_cii.add_axes(CBAR_RECT)
sns.heatmap(
    mean_df, ax=ax_cii,
    cmap=CMAP, vmin=-vabs, vmax=vabs, center=0,
    annot=True, fmt='.2f',
    linewidths=0, linecolor='none',
    annot_kws={'fontsize': 6},
    square=True,
    cbar=False,
)
for _c in ax_cii.collections:
    _c.set_rasterized(True)
ax_cii.set_title('Scalar mean (distribution information lost)', fontsize=10)
ax_cii.tick_params(axis='x', rotation=45, labelsize=7)
ax_cii.tick_params(axis='y', rotation=0,  labelsize=7)
_add_shared_cbar(cbar_ax_cii, orientation='horizontal')

# Same highlight boxes + labels as Ci (seaborn coords: cell (r,c) at x=[c,c+1], y=[r,r+1])
for _ri, _ci, _pi, _wi, _lbl in _ORDERED_CELLS:
    _hl_col = _PAIR_COL[_pi]
    ax_cii.add_patch(_Rect((_ci, _ri), 1, 1,
                           fill=False, edgecolor=_hl_col, linewidth=2.0, zorder=12))
    ax_cii.text(_ci + 0.06, _ri + 0.18, _lbl,
                fontsize=6, fontweight='bold', color=_hl_col, zorder=13,
                va='top', ha='left')

fig_cii.savefig(IMAGES / 'panel_cii.svg', dpi=600)
fig_cii.savefig(IMAGES / 'panel_cii.png', dpi=600)
plt.close(fig_cii)
print("  saved panel_cii.svg")

print("\nAll panels done.")
