"""Regenerate Panel A — run from the nbs/ directory."""
import matplotlib
matplotlib.use('Agg')
import pathlib, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
import dabest
from dabest.multi import combine

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
IMAGES = pathlib.Path('images')

# steelblue → white (at midpoint 0.5) → orange → crimson
# White MUST be at position 0.5 so Normalize(-bound, bound) places white at value 0
CMAP_SWOR = LinearSegmentedColormap.from_list(
    'steelwhiteorangered',
    [(0.00, (70/255, 130/255, 180/255)),
     (0.50, (1.0,   1.0,   1.0  )),
     (0.75, (1.0,   0.55,  0.0  )),
     (1.00, (0.86,  0.08,  0.24 ))],
    N=256
)

def _spiralize(fill, m, n):
    i=0; j=0; k=0; array=np.zeros((m,n))
    while m>0 and k<len(fill):
        jj=j; ii=i
        for j in range(j,n):
            if k>=len(fill): break
            array[i,j]=fill[k]; k+=1
        for i in range(ii+1,m):
            if k>=len(fill): break
            array[i,j]=fill[k]; k+=1
        for j in range(n-2,jj-1,-1):
            if k>=len(fill): break
            array[i,j]=fill[k]; k+=1
        for i in range(m-2,ii,-1):
            if k>=len(fill): break
            array[i,j]=fill[k]; k+=1
        m-=1; n-=1; j+=1
    return array

def _spiral_path(n):
    path = []
    top, bot, lft, rgt = 0, n-1, 0, n-1
    while top <= bot and lft <= rgt:
        for c in range(lft, rgt+1): path.append((top, c))
        top += 1
        for r in range(top, bot+1): path.append((r, rgt))
        rgt -= 1
        if top <= bot:
            for c in range(rgt, lft-1, -1): path.append((bot, c))
            bot -= 1
        if lft <= rgt:
            for r in range(bot, top-1, -1): path.append((r, lft))
            lft += 1
    return path

def _ring_structure(n):
    rings, pos, k = [], 0, 0
    while True:
        sz = n - 2*k
        if sz <= 0: break
        nc = 1 if sz == 1 else 4*(sz-1)
        rings.append({'k': k, 'start': pos, 'end': pos+nc, 'sz': sz})
        pos += nc; k += 1
        if sz == 1: break
    return rings

# Bootstrap
n = 7; N = n*n
_rng    = np.random.default_rng(42)
noise_a = _rng.normal(0, 3., 20); noise_b = _rng.normal(0, 3., 20)
group_a = noise_a - noise_a.mean(); group_b = noise_b - noise_b.mean() + 0.5
_df_a   = pd.DataFrame({'group': ['A']*20+['B']*20,
                         'value': np.concatenate([group_a, group_b])})
_dobj_a = dabest.load(_df_a, x='group', y='value', idx=('A', 'B'))
_multi_a = combine([[_dobj_a]], ['gene'], row_labels=['region'], effect_size='mean_diff')
bs_raw  = sorted(_multi_a.bootstraps[0])
chop    = int(np.ceil(len(bs_raw)*0.025)); bs_c = bs_raw[chop:-chop]
ranks   = np.linspace(0, len(bs_c), N, dtype=int); ranks[0] = 1
fv      = np.array([bs_c[r-1] for r in ranks])
if sum(v>0 for v in bs_c) < len(bs_c)/2: fv = fv[::-1]

XLIM  = (-3.5, 3.5)
rings = _ring_structure(n)

fv_sorted  = np.clip(np.sort(fv), *XLIM)
cell_vals  = _spiralize(fv.tolist(), n, n)

# Symmetric value-based norm — same for panel AND colorbar so colours match exactly
bound      = max(2.0, np.ceil(max(abs(fv_sorted[0]), abs(fv_sorted[-1])) * 2) / 2)
VALUE_NORM = Normalize(vmin=-bound, vmax=bound)

spiral_pos = _spiral_path(n)

# Figure
fig = plt.figure(figsize=(11, 5.4))
ax_cell = fig.add_axes([0.03, 0.18, 0.32, 0.68])
ax_cbar = fig.add_axes([0.03, 0.08, 0.32, 0.05])
ax_hist = fig.add_axes([0.43, 0.12, 0.54, 0.74])

# Left: whorl cell coloured by effect size value (same norm as colorbar)
ax_cell.imshow(cell_vals, cmap=CMAP_SWOR, norm=VALUE_NORM,
               origin='upper', interpolation='nearest', aspect='equal')

def _ring_arrows(ax, ring_k, sz, avg_val):
    mid = ring_k + sz // 2
    bg  = CMAP_SWOR(VALUE_NORM(avg_val))
    col = 'w' if (0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]) < 0.52 else 'k'
    ap  = dict(arrowstyle='->', color=col, lw=1.0, mutation_scale=9)
    d   = 0.25  # shift outward from pixel centre by this fraction of a cell
    ax.annotate('', xy=(mid+0.6, ring_k-d),         xytext=(mid-0.6, ring_k-d),         arrowprops=ap, zorder=8, annotation_clip=False)
    ax.annotate('', xy=(ring_k+sz-1+d, mid+0.6),    xytext=(ring_k+sz-1+d, mid-0.6),    arrowprops=ap, zorder=8, annotation_clip=False)
    ax.annotate('', xy=(mid-0.6, ring_k+sz-1+d),    xytext=(mid+0.6, ring_k+sz-1+d),    arrowprops=ap, zorder=8, annotation_clip=False)
    ax.annotate('', xy=(ring_k-d, mid-0.6),             xytext=(ring_k-d, mid+0.6),             arrowprops=ap, zorder=8, annotation_clip=False)

for ring in rings[:-1]:
    avg_val = float(np.mean(fv_sorted[ring['start']:ring['end']]))
    _ring_arrows(ax_cell, ring['k'], ring['sz'], avg_val)

# Annotate every pixel: quantile rank + effect size value (luminance from VALUE_NORM)
for k, (r, c) in enumerate(spiral_pos):
    val = float(fv[k])
    bg  = CMAP_SWOR(VALUE_NORM(val))
    lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
    tc  = 'w' if lum < 0.52 else 'k'
    ax_cell.text(c, r, f'q{k}\n{val:.1f}',
                 ha='center', va='center', fontsize=4.5, color=tc,
                 zorder=10, linespacing=1.1, multialignment='center')

ax_cell.set_xlim(-0.5, n-0.5); ax_cell.set_ylim(n-0.5, -0.5)
ax_cell.axis('off')
ax_cell.set_title('7×7 whorl cell — outside-in spiral fill\n(pixel colour = effect size)', fontsize=9)

# Colorbar: same VALUE_NORM — colours identical between panel and bar
cb = ColorbarBase(ax_cbar, cmap=CMAP_SWOR, norm=VALUE_NORM, orientation='horizontal')
cb_ticks = np.arange(-bound, bound + 0.01, 0.5)
cb.set_ticks(cb_ticks)
cb.set_ticklabels([f'{t:g}' for t in cb_ticks])
cb.ax.tick_params(labelsize=7.5)
cb.set_label('Effect size (bootstrap mean difference)', fontsize=8)

# Right: stacked quantile histogram (value-coloured segments, matching panel + colorbar)
BIN_W = 0.20
bins  = np.arange(XLIM[0], XLIM[1]+BIN_W, BIN_W)
counts, _ = np.histogram(np.clip(bs_c, *XLIM), bins=bins, density=True)

seg_bounds = np.empty(N+1)
seg_bounds[0] = XLIM[0]; seg_bounds[-1] = XLIM[1]
for k in range(1, N):
    seg_bounds[k] = (fv_sorted[k-1] + fv_sorted[k]) / 2.0
seg_cols = [CMAP_SWOR(VALUE_NORM(float(v))) for v in fv_sorted]

best_piece = {}  # k -> (x_center, y_bottom, h)

for lo, hi, ht in zip(bins[:-1], bins[1:], counts):
    if ht == 0: continue
    y_bot = 0.0
    for k in range(N):
        overlap = max(0., min(hi, seg_bounds[k+1]) - max(lo, seg_bounds[k]))
        if overlap <= 0: continue
        h = ht * overlap / (hi - lo)
        ax_hist.bar(lo, h, width=hi-lo, bottom=y_bot,
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
    ax_hist.text(xc, yb + h/2, f'q{k}\n{val:.1f}',
                 ha='center', va='center', fontsize=3.5, color=tc,
                 zorder=11, linespacing=1.0, multialignment='center')

margin = (fv_sorted[-1] - fv_sorted[0]) * 0.08
ax_hist.set_xlim(fv_sorted[0] - margin, fv_sorted[-1] + margin)
ax_hist.set_ylim(0, counts.max()*1.03)
ax_hist.spines['left'].set_visible(False)
ax_hist.spines['right'].set_visible(False); ax_hist.spines['top'].set_visible(False)
ax_hist.set_yticks([]); ax_hist.tick_params(labelsize=8)
ax_hist.set_xlabel('Bootstrap mean difference', fontsize=9)
ax_hist.set_title('Bootstrap distribution, segmented by quantile rank\n'
                  '(bar colours match spiral position on left)', fontsize=9)

fig.suptitle('A   From bootstrap distribution to whorl cell: '
             'quantile rank encodes spatial position',
             fontsize=11, fontweight='bold', x=0.02, ha='left', y=0.99)

fig.savefig(IMAGES/'panel_a.svg', bbox_inches='tight')
fig.savefig(IMAGES/'panel_a.png', dpi=600, bbox_inches='tight')
print("Saved panel_a.svg / panel_a.png")
