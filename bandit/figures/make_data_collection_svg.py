"""Generate the 'data collection with Llama' figure for the bandit experiment.

Prompt-centric linear layout (top -> bottom):
  1. environment strip   : restless 4-armed bandit (drifting payouts) + arm->letter map
  2. the prompt (hero)   : chat-templated prompt Llama actually sees, with role tags and
                           callouts (instruction / raw history / decision token)
  3. readout + loop      : Llama -> next-token distribution over arm letters (+other) ->
                           sample -> reward -> append line -> repeat -> dataset

Faithful to llm_data/prompt.py + rollout.py. Plain, transform-free, Inkscape-editable SVG.

    python bandit/figures/make_data_collection_svg.py     # -> bandit/figures/data_collection.svg
"""
from __future__ import annotations

import math
import pathlib

# ----------------------------------------------------------------------------- task constants
# arm index -> (letter, color). Matches the prompt.py self-test example history below.
ARMS = [("B", "#ec4a3b"), ("C", "#1ec9da"), ("D", "#57bf36"), ("F", "#9b4dc4")]
HISTORY = [(0, 58), (0, 61), (2, 22), (1, 47)]          # (arm, reward), from prompt.py _selftest
DIST = [0.55, 0.15, 0.08, 0.12]                          # P over arms B,C,D,F
DIST_OTHER = 0.10
SAMPLED = 1                                              # history line whose reward we 'append' (the 61)

# ----------------------------------------------------------------------------- palette
INK, MUT, FAINT = "#202124", "#5f6368", "#9aa0a6"
CARD_BG, CARD_BD = "#fbfaf7", "#d9d5cc"
DELIM = "#c98a00"                                        # << >> delimiter accent
DARK = "#2b2b33"
W, H = 1000, 880

MONO = "'SF Mono','Menlo','Consolas','Roboto Mono',monospace"
SANS = "'Roboto','Helvetica Neue',Arial,sans-serif"

out: list[str] = []
def add(s: str) -> None: out.append(s)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# =============================================================================== header
add(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{SANS}">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L7.5,3 L0,6 Z" fill="{INK}"/></marker>
    <marker id="arM" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L7.5,3 L0,6 Z" fill="{MUT}"/></marker>
    <filter id="sh" x="-10%" y="-10%" width="120%" height="125%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="b"/>
      <feOffset in="b" dx="0" dy="3" result="o"/>
      <feComponentTransfer in="o" result="s"><feFuncA type="linear" slope="0.16"/></feComponentTransfer>
      <feMerge><feMergeNode in="s"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <text x="40" y="46" font-size="24" font-weight="700" fill="{INK}">Data collection: Llama plays the restless bandit</text>
  <text x="40" y="70" font-size="15" fill="{MUT}">Each episode, the model reads its own raw history and picks the next arm. We log every choice.</text>
''')

# =============================================================================== 1. ENVIRONMENT STRIP
# --- left: drifting reward curves
bx0, bx1, by0, by1 = 48, 360, 116, 196
add(f'  <text x="48" y="106" font-size="15" font-weight="600" fill="{INK}">Restless 4-armed bandit</text>')
add(f'  <rect x="{bx0}" y="{by0}" width="{bx1-bx0}" height="{by1-by0}" rx="6" fill="#fcfcfb" stroke="{CARD_BD}"/>')
for a, (letter, col) in enumerate(ARMS):
    base = [58, 46, 64, 38][a]
    drift = [12, -13, -20, 15][a]
    pts = []
    for k in range(41):
        t = k / 40
        val = base + 13 * math.sin(6.28 * (0.8 + 0.25 * a) * t + 1.7 * a) + drift * t
        val = max(6, min(94, val))
        px = bx0 + t * (bx1 - bx0)
        py = by1 - (val / 100) * (by1 - by0)
        pts.append(f"{px:.1f},{py:.1f}")
    add(f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
add(f'  <text x="{(bx0+bx1)/2:.0f}" y="{by1+16:.0f}" font-size="12" fill="{MUT}" text-anchor="middle">payouts drift slowly &#8594; the best arm changes</text>')

# --- right: arm -> letter map
mx = 470
add(f'  <text x="{mx}" y="106" font-size="15" font-weight="600" fill="{INK}">Arms &#8594; letters <tspan fill="{MUT}" font-weight="400">(reshuffled every episode)</tspan></text>')
for a, (letter, col) in enumerate(ARMS):
    x = mx + a * 92
    add(f'  <text x="{x+24}" y="132" font-size="12" fill="{MUT}" text-anchor="middle">arm {a+1}</text>')
    add(f'  <rect x="{x}" y="138" width="48" height="48" rx="9" fill="{col}" filter="url(#sh)"/>')
    add(f'  <text x="{x+24}" y="171" font-size="26" font-weight="700" fill="#fff" text-anchor="middle" font-family="{MONO}">{letter}</text>')
add(f'  <text x="{mx}" y="200" font-size="12" fill="{MUT}">consonants only (no &#8220;A&#8221;), non-adjacent &#8594; no order / position / token bias</text>')

# =============================================================================== 2. PROMPT CARD (hero)
cx, cy, cw = 48, 224, 590
ch = 286
add(f'  <rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="10" fill="{CARD_BG}" stroke="{CARD_BD}" stroke-width="1.4" filter="url(#sh)"/>')
add(f'  <text x="{cx+18}" y="{cy+24}" font-size="12.5" font-weight="600" fill="{MUT}" letter-spacing="0.04em">PROMPT GIVEN TO LLAMA 3.1 8B-INSTRUCT</text>')
add(f'  <line x1="{cx+14}" y1="{cy+34}" x2="{cx+cw-14}" y2="{cy+34}" stroke="{CARD_BD}"/>')

tag_x, txt_x, fs, lh = cx + 96, cx + 110, 14.5, 22.5
y = cy + 60

def tag(label, yy):
    add(f'  <text x="{tag_x}" y="{yy}" font-size="11" fill="{FAINT}" text-anchor="end" font-style="italic">{label}</text>')

def mono(yy, spans):
    s = "".join(spans)
    add(f'  <text x="{txt_x}" y="{yy}" font-size="{fs}" font-family="{MONO}" fill="{INK}">{s}</text>')

def t(txt, fill=INK, bold=False, italic=False):
    w = ' font-weight="700"' if bold else ''
    i = ' font-style="italic"' if italic else ''
    f = f' fill="{fill}"' if fill != INK else ''
    return f'<tspan{f}{w}{i}>{esc(txt)}</tspan>'

# --- system
tag("system", y)
mono(y,           [t("You are playing a game with 4 slot machines")])
y += lh
mono(y, [t("labelled "),
         t("B", ARMS[0][1], bold=True), t(", "),
         t("C", ARMS[1][1], bold=True), t(", "),
         t("D", ARMS[2][1], bold=True), t(", "),
         t("F", ARMS[3][1], bold=True), t(".  Payouts drift over time,")])
y += lh
mono(y, [t("so the best machine changes.  Reply as "),
         t("<<", DELIM, bold=True), t("X"), t(">>", DELIM, bold=True), t(".")])

# --- user (raw history)
y += lh + 10
hist_top = y - 14
tag("user", y)
for (arm, reward) in HISTORY:
    letter, col = ARMS[arm]
    mono(y, [t("You press "),
             t("<<", DELIM, bold=True), t(letter, col, bold=True), t(">>", DELIM, bold=True),
             t(" and get "), t(str(reward), INK, bold=True), t(" points.")])
    y += lh
hist_bot = y - lh + 6
mono(y, [t("Which machine do you press next?")])

# --- assistant (decision stub + cursor)
y += lh + 10
tag("assistant", y)
mono(y, [t("You press "), t("<<", DELIM, bold=True)])
cur_x = txt_x + 11 * (len("You press ") + 2)        # approx end of the stub (monospace ~ fs*0.6)
cur_x = txt_x + fs * 0.60 * (len("You press <<"))
add(f'  <rect x="{cur_x:.0f}" y="{y-12:.0f}" width="8" height="15" fill="{INK}"/>')
stub_y = y

# =============================================================================== CALLOUTS (right of card)
ann_x = cx + cw + 26
# (A) instruction
add(f'  <text x="{ann_x}" y="{cy+58}" font-size="12.5" fill="{INK}" font-weight="600">instruction</text>')
add(f'  <text x="{ann_x}" y="{cy+76}" font-size="12" fill="{MUT}">labels, drifting payouts,</text>')
add(f'  <text x="{ann_x}" y="{cy+92}" font-size="12" fill="{MUT}">and the answer format.</text>')
add(f'  <line x1="{cx+cw}" y1="{cy+64}" x2="{ann_x-6}" y2="{cy+64}" stroke="{MUT}" stroke-width="1" marker-end="url(#arM)"/>')
# (B) raw-history bracket
brx = cx + cw + 8
midh = (hist_top + hist_bot) / 2
add(f'  <path d="M{brx-6},{hist_top} L{brx},{hist_top} L{brx},{hist_bot} L{brx-6},{hist_bot}" fill="none" stroke="{MUT}" stroke-width="1.3"/>')
add(f'  <line x1="{brx}" y1="{midh:.0f}" x2="{ann_x-6}" y2="{midh:.0f}" stroke="{MUT}" stroke-width="1"/>')
add(f'  <text x="{ann_x}" y="{midh-8:.0f}" font-size="12.5" fill="{INK}" font-weight="600">raw per-trial history</text>')
add(f'  <text x="{ann_x}" y="{midh+9:.0f}" font-size="12" fill="{MUT}">never summarized &#8212; value</text>')
add(f'  <text x="{ann_x}" y="{midh+25:.0f}" font-size="12" fill="{MUT}">integration must live inside</text>')
add(f'  <text x="{ann_x}" y="{midh+41:.0f}" font-size="12" fill="{MUT}">the network.</text>')

# =============================================================================== 3. READOUT + LOOP
# (C) decision-token callout + arrow from stub down to Llama
llx, lly, llw, llh = 92, 560, 210, 48
add(f'  <line x1="{cur_x+18:.0f}" y1="{stub_y+8:.0f}" x2="{llx+llw/2:.0f}" y2="{lly-2:.0f}" stroke="{INK}" stroke-width="1.6" marker-end="url(#ar)"/>')
add(f'  <text x="{cur_x+28:.0f}" y="{stub_y+2:.0f}" font-size="12.5" fill="{INK}" font-weight="600">decision token</text>')
add(f'  <text x="{cur_x+28:.0f}" y="{stub_y+19:.0f}" font-size="12" fill="{MUT}">prompt ends at <tspan font-family="{MONO}" fill="{DELIM}" font-weight="700">&lt;&lt;</tspan> so the next</text>')
add(f'  <text x="{cur_x+28:.0f}" y="{stub_y+35:.0f}" font-size="12" fill="{MUT}">token <tspan font-style="italic">is</tspan> the chosen arm.</text>')

# Llama box
add(f'  <rect x="{llx}" y="{lly}" width="{llw}" height="{llh}" rx="9" fill="{DARK}" filter="url(#sh)"/>')
add(f'  <text x="{llx+llw/2}" y="{lly+30}" font-size="16" font-weight="600" fill="#fff" text-anchor="middle">Llama 3.1 8B-Instruct</text>')

# distribution panel
dpx = 360
add(f'  <text x="{dpx}" y="{lly-2}" font-size="13.5" font-weight="600" fill="{INK}">next-token distribution <tspan fill="{MUT}" font-weight="400">(arm letters + &#8220;other&#8221;)</tspan></text>')
add(f'  <line x1="{llx+llw}" y1="{lly+llh/2}" x2="{dpx-8}" y2="{lly+llh/2}" stroke="{INK}" stroke-width="1.6" marker-end="url(#ar)"/>')
bar_x, bar_max, bh, gap = dpx + 34, 230, 15, 8
by = lly + 16
rows = [(ARMS[i][0], ARMS[i][1], DIST[i]) for i in range(4)] + [("other", FAINT, DIST_OTHER)]
for i, (lab, col, p) in enumerate(rows):
    yy = by + i * (bh + gap)
    add(f'  <text x="{bar_x-8}" y="{yy+bh-3}" font-size="13" font-family="{MONO}" fill="{INK}" text-anchor="end" font-weight="{700 if i==SAMPLED else 400}">{lab}</text>')
    add(f'  <rect x="{bar_x}" y="{yy}" width="{p*bar_max:.0f}" height="{bh}" rx="2" fill="{col}"/>')
    add(f'  <text x="{bar_x+p*bar_max+8:.0f}" y="{yy+bh-3}" font-size="12" fill="{MUT}">{p:.2f}</text>')
    if i == SAMPLED:
        add(f'  <rect x="{bar_x-1}" y="{yy-1}" width="{p*bar_max+2:.0f}" height="{bh+2}" rx="2" fill="none" stroke="{INK}" stroke-width="1.6"/>')
        add(f'  <text x="{bar_x+p*bar_max+38:.0f}" y="{yy+bh-3}" font-size="12.5" font-weight="600" fill="{INK}">&#8592; sampled</text>')

# dataset box (terminal)
gx, gy, gw, gh = 720, 548, 236, 132
add(f'  <line x1="{bar_x+bar_max+78:.0f}" y1="{by+60}" x2="{gx-8}" y2="{by+60}" stroke="{INK}" stroke-width="1.6" marker-end="url(#ar)"/>')
add(f'  <rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="9" fill="#f4f6f8" stroke="{CARD_BD}"/>')
add(f'  <text x="{gx+16}" y="{gy+26}" font-size="15" font-weight="700" fill="{INK}">Dataset</text>')
add(f'  <text x="{gx+16}" y="{gy+50}" font-size="12.5" fill="{MUT}">&#8776;150 trials &#215; &#8776;400 episodes</text>')
add(f'  <text x="{gx+16}" y="{gy+74}" font-size="12.5" fill="{INK}">logged per trial:</text>')
add(f'  <text x="{gx+16}" y="{gy+93}" font-size="12.5" fill="{MUT}">&#8226; action &amp; reward</text>')
add(f'  <text x="{gx+16}" y="{gy+111}" font-size="12.5" fill="{MUT}">&#8226; choice distribution</text>')

# =============================================================================== LOOP RIBBON + return arrow
ry = 728
def chip(x, w, label, sub=None):
    add(f'  <rect x="{x}" y="{ry}" width="{w}" height="40" rx="8" fill="#fff" stroke="{CARD_BD}"/>')
    add(f'  <text x="{x+w/2:.0f}" y="{ry+(18 if sub else 25):.0f}" font-size="13" font-weight="600" fill="{INK}" text-anchor="middle">{esc(label)}</text>')
    if sub:
        add(f'  <text x="{x+w/2:.0f}" y="{ry+33:.0f}" font-size="11" fill="{MUT}" text-anchor="middle" font-family="{MONO}">{esc(sub)}</text>')

chips = [(150, "sample an arm"), (300, "environment reward"), (470, "append history line")]
xc = 92
widths = [128, 150, 250]
labels = ["sample an arm", "scheduled reward", "append the new line"]
subs = [None, "61 points", "You press <<B>> and get 61…"]
positions = []
for wd, lab, sub in zip(widths, labels, subs):
    chip(xc, wd, lab, sub)
    positions.append((xc, wd))
    xc += wd + 30
# arrows between chips
for i in range(len(positions) - 1):
    x0 = positions[i][0] + positions[i][1]
    x1 = positions[i + 1][0]
    add(f'  <line x1="{x0+4}" y1="{ry+20}" x2="{x1-6}" y2="{ry+20}" stroke="{INK}" stroke-width="1.5" marker-end="url(#ar)"/>')
# repeat label after last chip
lastx = positions[-1][0] + positions[-1][1]
add(f'  <text x="{lastx+16}" y="{ry+26}" font-size="13" font-weight="600" fill="{INK}">repeat &#8635;</text>')

# closed-loop return: the appended line re-enters next trial's prompt. Routed down and along
# the left margin (clear of the dataset / readout) back into the prompt's history region.
app_cx = positions[-1][0] + positions[-1][1] / 2
add(f'  <path d="M{app_cx:.0f},{ry+40} L{app_cx:.0f},814 L28,814 L28,372 L{cx-3},372" '
    f'fill="none" stroke="{MUT}" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#arM)"/>')
add(f'  <text x="320" y="808" font-size="12" fill="{MUT}" text-anchor="middle">the appended line becomes part of next trial&#8217;s prompt</text>')

add('</svg>\n')

dst = pathlib.Path(__file__).parent / "data_collection.svg"
dst.write_text("\n".join(out))
print(f"wrote {dst}")
