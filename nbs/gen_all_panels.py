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

vabs = float(mean_df.abs().max().max()) * 1.05
# Use simple Normalize for the colorbar — TwoSlopeNorm + empty ScalarMappable
# has a known rendering bug in mpl 3.10 that produces a blank gradient.
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
shared_norm = Normalize(vmin=-vabs, vmax=vabs)

# Shared layout constants (fractions of figure): 13 rows × 20 cols grid
FIGSIZE   = (14, 7)
HAX_RECT  = [0.14, 0.21, 0.71, 0.72]
CBAR_RECT = [0.87, 0.21, 0.022, 0.72]

def _add_shared_cbar(cax):
    cb = ColorbarBase(cax, cmap=CMAP, norm=shared_norm, orientation='vertical')
    cb.set_label('Mean diff F−M (log₂TPM+1)', fontsize=8)
    cb.set_ticks([-round(vabs, 1), 0, round(vabs, 1)])
    cb.ax.tick_params(labelsize=7)

# ── Panel Ci — whorlmap ───────────────────────────────────────────────────────
print("Generating panel_ci …")
fig_ci = plt.figure(figsize=FIGSIZE)
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
_add_shared_cbar(cbar_ax_ci)
fig_ci.savefig(IMAGES / 'panel_ci.svg', dpi=600)
fig_ci.savefig(IMAGES / 'panel_ci.png', dpi=600)
plt.close(fig_ci)
print("  saved panel_ci.svg")

# ── Panel Cii — scalar heatmap ────────────────────────────────────────────────
print("Generating panel_cii …")
fig_cii = plt.figure(figsize=FIGSIZE)
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
_add_shared_cbar(cbar_ax_cii)
fig_cii.savefig(IMAGES / 'panel_cii.svg', dpi=600)
fig_cii.savefig(IMAGES / 'panel_cii.png', dpi=600)
plt.close(fig_cii)
print("  saved panel_cii.svg")

print("\nAll panels done.")
