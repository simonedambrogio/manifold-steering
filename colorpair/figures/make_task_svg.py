"""Generate the behavioural-task schematic SVG for the color-pair comparison task.

Aesthetic mirrors the sequential-saccade monkey-task figure: a subject + monitor at the left with
dashed gaze lines, then a FLAT horizontal row of screens, each with the epoch name above and its
duration below, dashed saccade/response arrows, and panel letters. Output is plain, transform-free
SVG (explicit coordinates) so it stays fully editable in Inkscape; Inkscape-safe drop shadows.

    python colorpair/figures/make_task_svg.py        # writes colorpair/figures/task.svg
"""
from __future__ import annotations

import colorsys
import math
import pathlib

# ----------------------------------------------------------------------------- task constants
N = 24                                   # latent ring size (env.N_COLORS)
PAIR0 = (5, 7)                           # the CLOSER pair (circular distance 2)
PAIR1 = (14, 23)                         # the farther pair (circular distance 9); 23 sits by the seam

# ----------------------------------------------------------------------------- palette / geometry
SCR_TOP, SCR_BOT, SCR_EDGE = "#dcd9d1", "#cac6bd", "#a7a097"
INK, MUT, GREEN = "#2b2b2b", "#5b564f", "#3a9d5d"
W, H = 1240, 740
SW, SH = 150, 160                        # flat screen size
ROW_Y = 150                              # screen top
XS = [300 + i * (SW + 28) for i in range(5)]


def hue_hex(i: int, sat: float = 0.95, val: float = 0.98) -> str:
    r, g, b = colorsys.hsv_to_rgb((i % N) / N, sat, val)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


out: list[str] = []
def add(s: str) -> None: out.append(s)


# ----------------------------------------------------------------------------- screen primitives
def screen(x: float, label: str, dur: str, body: str, y: float = ROW_Y,
           w: float = SW, h: float = SH) -> str:
    cx = x + w / 2
    return f'''  <g>
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="url(#screenfill)"
          stroke="{SCR_EDGE}" stroke-width="1.2" filter="url(#cardshadow)"/>
    <text x="{cx}" y="{y-12:.0f}" font-size="14.5" font-weight="600" fill="{INK}" text-anchor="middle">{label}</text>
    <text x="{cx}" y="{y+h+22:.0f}" font-size="13" fill="{MUT}" text-anchor="middle">{dur}</text>
{body}  </g>'''


def fixdot(x: float, y: float = ROW_Y) -> str:
    return f'    <circle cx="{x+SW/2:.0f}" cy="{y+SH/2:.0f}" r="3.4" fill="{INK}"/>\n'


def sq(x: float, y: float, s: float, color: str, stroke: str = "#ffffff") -> str:
    return (f'    <rect x="{x:.0f}" y="{y:.0f}" width="{s}" height="{s}" rx="3" fill="{color}" '
            f'stroke="{stroke}" stroke-width="1.6"/>\n')


def two_pairs(x: float, y: float, s: float = 26) -> str:
    """Four color squares: pair 0 (closer, left group) and pair 1 (farther, right group)."""
    cy = y + SH / 2 - s / 2
    b = sq(x + 22, cy, s, hue_hex(PAIR0[0])) + sq(x + 22 + s + 4, cy, s, hue_hex(PAIR0[1]))
    b += sq(x + SW - 22 - 2 * s - 4, cy, s, hue_hex(PAIR1[0])) + sq(x + SW - 22 - s, cy, s, hue_hex(PAIR1[1]))
    return b


# ----------------------------------------------------------------------------- ring primitives
def ring(cx, cy, R, dr=6.0):
    s = ""
    for i in range(N):
        a = math.radians(-90 + i * 360 / N)
        s += (f'    <circle cx="{cx+R*math.cos(a):.1f}" cy="{cy+R*math.sin(a):.1f}" r="{dr}" '
              f'fill="{hue_hex(i)}" stroke="#ffffff" stroke-width="1.1"/>\n')
    return s


def ring_pt(cx, cy, R, i):
    a = math.radians(-90 + i * 360 / N)
    return cx + R * math.cos(a), cy + R * math.sin(a)


def arc(cx, cy, R, i, j, color, width):
    x0, y0 = ring_pt(cx, cy, R, i)
    x1, y1 = ring_pt(cx, cy, R, j)
    return (f'    <path d="M {x0:.1f},{y0:.1f} A {R},{R} 0 0 1 {x1:.1f},{y1:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>\n')


# =============================================================================== build
add(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="'Helvetica Neue', Arial, sans-serif">
  <defs>
    <linearGradient id="screenfill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{SCR_TOP}"/><stop offset="1" stop-color="{SCR_BOT}"/>
    </linearGradient>
    <filter id="cardshadow" x="-15%" y="-15%" width="135%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="b"/>
      <feOffset in="b" dx="2" dy="4" result="o"/>
      <feComponentTransfer in="o" result="s"><feFuncA type="linear" slope="0.26"/></feComponentTransfer>
      <feMerge><feMergeNode in="s"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L7.5,3 L0,6 Z" fill="{INK}"/></marker>
    <marker id="arG" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L7.5,3 L0,6 Z" fill="{GREEN}"/></marker>
  </defs>

  <text x="26" y="40" font-size="25" font-weight="700" fill="{INK}">A</text>
  <text x="56" y="40" font-size="18" font-weight="600" fill="{INK}">Color-pair comparison task</text>
''')

# ----------------------------------------------------------------------------- subject + monitor
HX, HY = 70, 230
add(f'''  <g id="subject">
    <!-- stylized observer head in profile, facing the monitor -->
    <path d="M {HX+18},{HY-60}
             C {HX-22},{HY-58} {HX-28},{HY+6} {HX+2},{HY+30}
             C {HX+8},{HY+36} {HX+6},{HY+52} {HX+14},{HY+58}
             L {HX+14},{HY+86} L {HX+58},{HY+86}
             C {HX+58},{HY+64} {HX+92},{HY+60} {HX+96},{HY+30}
             C {HX+116},{HY+24} {HX+118},{HY-2} {HX+98},{HY-6}
             C {HX+96},{HY-40} {HX+66},{HY-62} {HX+18},{HY-60} Z"
          fill="#eceae5" stroke="{INK}" stroke-width="1.6"/>
    <circle cx="{HX+74}" cy="{HY-12}" r="3.6" fill="{INK}"/>
    <!-- monitor -->
    <rect x="{HX+120}" y="{HY-44}" width="96" height="78" rx="6" fill="url(#screenfill)" stroke="{SCR_EDGE}" stroke-width="1.4"/>
    <rect x="{HX+161}" y="{HY+34}" width="14" height="14" fill="#bdb8af"/>
    <rect x="{HX+150}" y="{HY+48}" width="36" height="5" rx="2" fill="#bdb8af"/>
    <rect x="{HX+132}" y="{HY-22}" width="15" height="15" rx="2" fill="{hue_hex(PAIR0[0])}"/>
    <rect x="{HX+149}" y="{HY-22}" width="15" height="15" rx="2" fill="{hue_hex(PAIR0[1])}"/>
    <rect x="{HX+174}" y="{HY-22}" width="15" height="15" rx="2" fill="{hue_hex(PAIR1[0])}"/>
    <rect x="{HX+191}" y="{HY-22}" width="15" height="15" rx="2" fill="{hue_hex(PAIR1[1])}"/>
    <!-- gaze lines -->
    <line x1="{HX+78}" y1="{HY-12}" x2="{HX+120}" y2="{HY-40}" stroke="{MUT}" stroke-width="1" stroke-dasharray="4 3"/>
    <line x1="{HX+78}" y1="{HY-12}" x2="{HX+120}" y2="{HY+30}" stroke="{MUT}" stroke-width="1" stroke-dasharray="4 3"/>
  </g>
''')

# ----------------------------------------------------------------------------- the trial row
add(screen(XS[0], "Fixation", "0.5 s", fixdot(XS[0])))
add(screen(XS[1], "Stimuli", "0.5 s", two_pairs(XS[1], ROW_Y)))
add(screen(XS[2], "Delay", "0.5–1.0 s", fixdot(XS[2])))

# response: two target placeholders (where the pairs were) + dashed saccade arrow to chosen (left) pair
bx = XS[3]
cyc = ROW_Y + SH / 2
body = ""
for tx in (bx + 35, bx + SW - 35):
    body += (f'    <circle cx="{tx:.0f}" cy="{cyc:.0f}" r="17" fill="none" stroke="#8e877b" '
             f'stroke-width="1.5" stroke-dasharray="4 3"/>\n')
body += (f'    <line x1="{bx+SW/2:.0f}" y1="{cyc:.0f}" x2="{bx+35+15:.0f}" y2="{cyc:.0f}" '
         f'stroke="{GREEN}" stroke-width="2.2" stroke-dasharray="5 3" marker-end="url(#arG)"/>\n')
add(screen(bx, "Response: choose closer pair", "until response", body))

# feedback: chosen (left) pair marked correct
fx = XS[4]
body = (f'    <circle cx="{fx+35:.0f}" cy="{cyc:.0f}" r="20" fill="none" stroke="{GREEN}" stroke-width="2.8"/>\n'
        + sq(fx + 35 - 13, cyc - 13, 26, hue_hex(PAIR0[0]))
        + f'    <path d="M {fx+35-8:.0f},{cyc:.0f} l 5,6 l 11,-14" fill="none" stroke="{GREEN}" '
          f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>\n')
add(screen(fx, "Feedback", "0.3 s", body))

# time arrow under the row
add(f'''  <line x1="{XS[0]}" y1="{ROW_Y+SH+44}" x2="{XS[4]+SW}" y2="{ROW_Y+SH+44}" stroke="{MUT}" stroke-width="1.4" marker-end="url(#ar)"/>
  <text x="{XS[0]}" y="{ROW_Y+SH+60}" font-size="12" fill="{MUT}">time</text>
''')

# =============================================================================== Panel B: hidden structure
BY = 470
add(f'''  <text x="26" y="{BY}" font-size="25" font-weight="700" fill="{INK}">B</text>
  <text x="56" y="{BY}" font-size="18" font-weight="600" fill="{INK}">The hidden structure: a latent color ring</text>
''')

# latent ring (left) with the seam
cx, cy, R = 320, BY + 130, 80
add('  <g id="latent-ring">\n')
add(ring(cx, cy, R))
sx, sy = cx, cy - R
add(f'    <line x1="{sx}" y1="{sy-16:.0f}" x2="{sx}" y2="{sy+16:.0f}" stroke="{INK}" stroke-width="2" stroke-dasharray="3 3"/>\n')
add(f'    <text x="{sx+9}" y="{sy-8:.0f}" font-size="12" font-style="italic" fill="{INK}">seam</text>\n')
add(f'    <text x="{cx}" y="{cy+R+30:.0f}" font-size="13" font-weight="600" fill="{INK}" text-anchor="middle">latent ring, never shown</text>\n')
add(f'    <text x="{cx}" y="{cy+R+48:.0f}" font-size="12" fill="{MUT}" text-anchor="middle">input is one-hot: all 24 colors equidistant</text>\n')
add('  </g>\n')

# decision ring (right): the 4 stimulus colors + within-pair arcs
cx2, cy2, R2 = 760, BY + 130, 80
add('  <g id="decision-ring">\n')
add(ring(cx2, cy2, R2, dr=5.5))
for idx in (*PAIR0, *PAIR1):
    x, y = ring_pt(cx2, cy2, R2, idx)
    add(f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="none" stroke="{INK}" stroke-width="2"/>\n')
add(arc(cx2, cy2, R2 + 13, PAIR0[0], PAIR0[1], GREEN, 4))
add(arc(cx2, cy2, R2 + 26, PAIR1[0], PAIR1[1], "#8d8579", 4))
mx, my = ring_pt(cx2, cy2, R2 + 24, (PAIR0[0] + PAIR0[1]) / 2)
add(f'    <text x="{mx+8:.1f}" y="{my:.1f}" font-size="12" font-weight="600" fill="{GREEN}">pair 0 (closer)</text>\n')
mx, my = ring_pt(cx2, cy2, R2 + 36, (PAIR1[0] + PAIR1[1]) / 2)
add(f'    <text x="{mx-12:.1f}" y="{my:.1f}" font-size="12" fill="#6f685d" text-anchor="end">pair 1</text>\n')
add(f'    <text x="{cx2}" y="{cy2+R2+30:.0f}" font-size="13" font-weight="600" fill="{INK}" text-anchor="middle">choose the smaller-arc pair</text>\n')
add(f'    <text x="{cx2}" y="{cy2+R2+48:.0f}" font-size="12" fill="{MUT}" text-anchor="middle">circular distance, not linear</text>\n')
add('  </g>\n')

# caption tying it together
add(f'''  <text x="980" y="{BY+70}" font-size="12.5" fill="{MUT}">
    <tspan x="980" dy="0">Seam pairs (across the cut) are</tspan>
    <tspan x="980" dy="18">circularly <tspan font-style="italic">near</tspan> but linearly <tspan font-style="italic">far</tspan>.</tspan>
    <tspan x="980" dy="26">Only a model that folds the colors</tspan>
    <tspan x="980" dy="18">into a ring judges them correctly.</tspan>
  </text>
''')

add('</svg>\n')

dst = pathlib.Path(__file__).parent / "task.svg"
dst.write_text("\n".join(out))
print(f"wrote {dst}")
