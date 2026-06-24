"""Figure 1 assembly.

SVG: svgutils compositing (for editing).
PNG: PIL stitch of source PNGs (correct colorbars, no SVG artefacts) with
     matplotlib sub-panel labels rendered on top at 600 DPI.

Run from the nbs/ directory.
"""
import svgutils.transform as sg

IMAGES   = "images"
GAP      = 24    # vertical gap between rows (pt)
LABEL_SZ = 13
FONT     = "Arial"


def pt(s):
    return float(str(s).replace("pt", ""))


def load(name):
    fig = sg.fromfile(f"{IMAGES}/{name}")
    return fig, pt(fig.width), pt(fig.height)


# ── Load all panels ────────────────────────────────────────────────────────────
fig_ai,  w_ai,  h_ai  = load("panel_ai.svg")
fig_a,   w_a,   h_a   = load("panel_a.svg")
fig_b,   w_b,   h_b   = load("panel_b.svg")
fig_ci,  w_ci,  h_ci  = load("panel_ci.svg")
fig_cii, w_cii, h_cii = load("panel_cii.svg")

print("Raw dimensions (pt):")
for name, w, h in [("panel_ai",w_ai,h_ai),("panel_a",w_a,h_a),
                    ("panel_b",w_b,h_b),("panel_ci",w_ci,h_ci),
                    ("panel_cii",w_cii,h_cii)]:
    print(f"  {name:12s}: {w:.1f} × {h:.1f}")

# ── Row A: Ai + A side by side, normalised to a common height ────────────────
# Pick target height = panel_a's natural height (keeps Aii/Aiii at 1:1 scale)
TARGET_H_A = h_a

scale_ai_row = TARGET_H_A / h_ai
scale_a_row  = 1.0   # panel_a at natural size

w_ai_scaled = w_ai * scale_ai_row
w_a_scaled  = w_a  * scale_a_row
ROW_A_W     = w_ai_scaled + w_a_scaled   # total row-A width

print(f"\nRow A total width: {ROW_A_W:.1f} pt  height: {TARGET_H_A:.1f} pt")

# ── Target width = max of all row widths ─────────────────────────────────────
TARGET_W = max(ROW_A_W, w_b, w_ci, w_cii)
print(f"Target width (all rows scaled to): {TARGET_W:.1f} pt")

# Rows B, Ci, Cii: each scales uniformly to TARGET_W
scale_b   = TARGET_W / w_b
scale_ci  = TARGET_W / w_ci
scale_cii = TARGET_W / w_cii

# Row A: scale together so the combined width hits TARGET_W
row_a_factor = TARGET_W / ROW_A_W
s_ai = scale_ai_row * row_a_factor
s_a  = scale_a_row  * row_a_factor
H_A  = TARGET_H_A  * row_a_factor

H_B   = h_b   * scale_b
H_CI  = h_ci  * scale_ci
H_CII = h_cii * scale_cii

# Y offsets
y_A   = 0
y_B   = H_A  + GAP
y_CI  = y_B  + H_B  + GAP
y_CII = y_CI + H_CI + GAP
TOTAL_H = y_CII + H_CII

print(f"\nRow A  : y={y_A:.1f}  h={H_A:.1f}")
print(f"Row B  : y={y_B:.1f}  h={H_B:.1f}")
print(f"Row Ci : y={y_CI:.1f}  h={H_CI:.1f}")
print(f"Row Cii: y={y_CII:.1f}  h={H_CII:.1f}")
print(f"Total  : {TARGET_W:.1f} × {TOTAL_H:.1f} pt")

# ── Position roots ────────────────────────────────────────────────────────────
root_ai  = fig_ai.getroot();  root_ai.moveto(0,                        y_A,  s_ai)
root_a   = fig_a.getroot();   root_a.moveto(w_ai * s_ai,               y_A,  s_a)
root_b   = fig_b.getroot();   root_b.moveto(0,                         y_B,  scale_b)
root_ci  = fig_ci.getroot();  root_ci.moveto(0,                        y_CI, scale_ci)
root_cii = fig_cii.getroot(); root_cii.moveto(0,                       y_CII,scale_cii)

# ── Sub-panel labels ──────────────────────────────────────────────────────────
LY = LABEL_SZ + 2   # y offset from row top

def lbl(x, y, text):
    return sg.TextElement(x, y, text, size=LABEL_SZ, weight="bold", font=FONT)

# Row A
x_Ai   = 4
x_Aii  = w_ai * s_ai + 4               # starts where panel_a begins (left = whorl cell)
x_Aiii = w_ai * s_ai + w_a * 0.41 * s_a   # histogram ax at ~43% of panel_a width

# Row B — 5 equally wide columns
col_w_b = TARGET_W / 5
labels_B = [lbl(col_w_b * j + 4, y_B + LY, f"B{'i'*1 + ('i'*(j))}")
            for j, _ in enumerate(['i','ii','iii','iv','v'])]
# Nicer: explicit roman
roman = ['Bi', 'Bii', 'Biii', 'Biv', 'Bv']
labels_B = [lbl(col_w_b * j + 4, y_B + LY, roman[j]) for j in range(5)]

# Row Ci / Cii
x_Ci  = 4
x_Cii = 4

all_labels = [
    lbl(x_Ai,   y_A + LY,   "Ai"),
    lbl(x_Aii,  y_A + LY,   "Aii"),
    lbl(x_Aiii, y_A + LY,   "Aiii"),
    lbl(x_Ci,   y_CI  + LY, "Ci"),
    lbl(x_Cii,  y_CII + LY, "Cii"),
] + labels_B

# ── Assemble ──────────────────────────────────────────────────────────────────
fig_out = sg.SVGFigure(f"{TARGET_W:.2f}pt", f"{TOTAL_H:.2f}pt")
fig_out.append([root_ai, root_a, root_b, root_ci, root_cii] + all_labels)
out_path = f"{IMAGES}/figure1_assembly.svg"
fig_out.save(out_path)
print(f"\nSaved: {out_path}")

# ── Patch SVG root width/height (needed by some SVG viewers) ─────────────────
from lxml import etree as _et
_tree = _et.parse(out_path)
_root = _tree.getroot()
_root.set("width",  f"{TARGET_W:.2f}pt")
_root.set("height", f"{TOTAL_H:.2f}pt")
_tree.write(out_path, xml_declaration=True, encoding="UTF-8", standalone=True)
print("  patched SVG width/height")

# ═══════════════════════════════════════════════════════════════════════════════
# PNG @ 600 DPI: PIL stitch source PNGs, then render labels with matplotlib
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DPI        = 600
LABEL_PT   = 13          # sub-panel label font size (points)
GAP_PX     = int(round(0.07 * DPI))   # ~42 px gap between rows

def _load(name):
    return Image.open(f"{IMAGES}/{name}").convert('RGB')

def _scale_to_w(img, target_w):
    h = int(round(img.height * target_w / img.width))
    return img.resize((target_w, h), Image.LANCZOS)

# Row A: Ai and A side by side at the same height, combined → TARGET_W_PX
p_ai = _load("panel_ai.png");  p_a = _load("panel_a.png")
_h_a = p_a.height
_w_ai_scaled = int(round(p_ai.width * _h_a / p_ai.height))
_p_ai_h = p_ai.resize((_w_ai_scaled, _h_a), Image.LANCZOS)
_raw_w  = _w_ai_scaled + p_a.width
TARGET_W_PX = max(_raw_w, _load("panel_b.png").width,
                  _load("panel_ci.png").width, _load("panel_cii.png").width)

_f = TARGET_W_PX / _raw_w
_row_a_h = int(round(_h_a * _f))
_p_ai_f  = _p_ai_h.resize((int(round(_w_ai_scaled * _f)), _row_a_h), Image.LANCZOS)
_p_a_f   = p_a.resize((int(round(p_a.width * _f)), _row_a_h), Image.LANCZOS)
row_a = Image.new('RGB', (TARGET_W_PX, _row_a_h), (255, 255, 255))
row_a.paste(_p_ai_f, (0, 0));  row_a.paste(_p_a_f, (_p_ai_f.width, 0))

row_b   = _scale_to_w(_load("panel_b.png"),   TARGET_W_PX)
row_ci  = _scale_to_w(_load("panel_ci.png"),  TARGET_W_PX)
row_cii = _scale_to_w(_load("panel_cii.png"), TARGET_W_PX)

gap = Image.new('RGB', (TARGET_W_PX, GAP_PX), (255, 255, 255))
_rows = [row_a, gap, row_b, gap, row_ci, gap, row_cii]
TOTAL_H_PX = sum(r.height for r in _rows)
combined = Image.new('RGB', (TARGET_W_PX, TOTAL_H_PX), (255, 255, 255))
_y = 0
for _r in _rows:
    combined.paste(_r, (0, _y));  _y += _r.height

# Y offsets of each row top in pixel space
_y_a   = 0
_y_b   = row_a.height + GAP_PX
_y_ci  = _y_b  + row_b.height  + GAP_PX
_y_cii = _y_ci + row_ci.height + GAP_PX

# Label x positions (pixels)
_LOFF = int(round(0.012 * DPI))    # ~7 px left margin per label
_YOFF = int(round(0.03  * DPI))    # ~18 px down from row top
_x_Ai   = _LOFF
_x_Aii  = _p_ai_f.width + _LOFF
_x_Aiii = _p_ai_f.width + int(round(_p_a_f.width * 0.41)) + _LOFF
_col_b  = TARGET_W_PX / 5

_labels = {
    "Ai":   (_x_Ai,              _y_a   + _YOFF),
    "Aii":  (_x_Aii,             _y_a   + _YOFF),
    "Aiii": (_x_Aiii,            _y_a   + _YOFF),
    "Bi":   (int(_col_b*0)+_LOFF, _y_b  + _YOFF),
    "Bii":  (int(_col_b*1)+_LOFF, _y_b  + _YOFF),
    "Biii": (int(_col_b*2)+_LOFF, _y_b  + _YOFF),
    "Biv":  (int(_col_b*3)+_LOFF, _y_b  + _YOFF),
    "Bv":   (int(_col_b*4)+_LOFF, _y_b  + _YOFF),
    "Ci":   (_LOFF,               _y_ci  + _YOFF),
    "Cii":  (_LOFF,               _y_cii + _YOFF),
}

# Render combined image in matplotlib and draw labels on top
_fig_w = TARGET_W_PX / DPI
_fig_h = TOTAL_H_PX  / DPI
_fig, _ax = plt.subplots(figsize=(_fig_w, _fig_h))
_fig.subplots_adjust(0, 0, 1, 1)
_ax.imshow(np.asarray(combined), aspect='auto')
_ax.axis('off')
for _txt, (_px, _py) in _labels.items():
    _ax.text(_px, _py, _txt,
             fontsize=LABEL_PT, fontweight='bold', fontfamily='Arial',
             color='black', va='top', ha='left',
             transform=_ax.transData)

_png_out = f"{IMAGES}/figure1_assembly.png"
_fig.savefig(_png_out, dpi=DPI, bbox_inches='tight', pad_inches=0)
plt.close(_fig)
print(f"  PNG @ {DPI} Dpi → {_png_out}")
