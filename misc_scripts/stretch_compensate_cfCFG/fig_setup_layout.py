"""Table layout of the pre-stretcher and the usCFG shaper, drawn to scale.

Writes fig_setup_layout.svg to config.OUTPUT_DIR, then exports the PDF with Inkscape (when it
is installed). Positions are read off config.LAYOUT_PNG, whose optical table is 56 in wide.
Optics symbols are from Alexander Franzen's ComponentLibrary (gwoptics.org), CC BY-NC 3.0;
symbols, beam primitives and band colours follow
manuscript_plotting_scripts/shaped_usCFG_paper/fig_setup_schematic.py.

The pulse is chirped in a modified 4f (G1 L1 L2 G2) folded by a roof retroreflector, which
returns it displaced vertically onto the pick-off mirror PO1. PBS1 splits it into the shaper
arm, a second modified 4f (G3 L3 L4 G4, picked off at PO2), and the delay arm, folded by the
retroreflector RR; PBS2 recombines the two into the centrifuge. The 4fs are drawn with chief
rays: they fan out from the first grating, run parallel between the lenses, head for the focal
plane of the second lens (f = half the lens spacing) and leave the second grating collimated.
"""
import math
import os
import subprocess
import tempfile
from pathlib import Path

from misc_scripts.stretch_compensate_cfCFG import config

SVG = config.OUTPUT_DIR / "fig_setup_layout.svg"
PDF = config.OUTPUT_DIR / "fig_setup_layout.pdf"

FONT = 10.0                       # ~4 pt at 3.375 in wide
PURPLE, BLUE, RED = "#d42ad4", "#2a6cff", "#ff2a2a"
DIM = "#000000"

# band colours and opacities, red (s = 0) to violet (s = 64)
N_BANDS = 64
BAND_COLORS = """e8493b e94e3a ea5438 eb5937 ec5f35 ec6433 ed6a32 ee6f30 ef742f ef7a2d f07f2c f1852a
f28a28 f18f27 f19327 f09826 ef9c25 eea024 eda523 eca922 ebad21 eab220 eab620 e9ba1f e5be1e d5bc21
c6ba24 b6b827 a6b62a 96b42c 86b32f 76b132 67af35 57ad37 47ab3a 3ca940 3ca54c 3ca058 3b9c64 3b9870
3b947d 3b9089 3b8c95 3b88a1 3b83ad 3a7fba 3a7bc6 3a77d2 3d74d8 4272d8 4771d8 4c6fd8 516dd7 566cd7
5b6ad7 6068d7 6667d7 6b65d7 7063d7 7562d7 7a60d6 7f5ed6 845dd6 895bd6""".split()
RAMP = [0.125, 0.190, 0.254, 0.319, 0.384, 0.448, 0.513, 0.577]
BAND_OPACITY = RAMP + [0.620] * (N_BANDS - 2 * len(RAMP)) + RAMP[::-1]

# ---- optics symbols (ComponentLibrary) ------------------------------------
STOPS = ('<stop offset="0.0056" style="stop-color:#89A4B6"/><stop offset="0.5" style="stop-color:#FFFFFF"/>'
         '<stop offset="1" style="stop-color:#89A4B6"/>')
SYMBOLS = {
    "mir": ((-0.375, -14.548),       # origin on the reflecting face; black strip is the back
            '<linearGradient id="{id}" gradientUnits="userSpaceOnUse" x1="350.0874" y1="43.0298" x2="350.0874" y2="14.6831" '
            'gradientTransform="matrix(0.35 0 0 1 -117.1952 -14.3086)">' + STOPS + '</linearGradient>'
            '<rect x="0.375" y="0.375" fill="url(#{id})" stroke="#000000" stroke-width="0.75" width="9.921" height="28.346"/>'
            '<rect x="7.46" y="0.375" stroke="#000000" stroke-width="0.75" width="2.836" height="28.346"/>'),
    "grat": ((-0.375, -14.549),
             '<linearGradient id="{id}" gradientUnits="userSpaceOnUse" x1="339.7324" y1="43.0312" x2="339.7324" y2="14.6846" '
             'gradientTransform="matrix(0.35 0 0 1 -113.5716 -14.3096)">' + STOPS + '</linearGradient>'
             '<polygon fill="url(#{id})" stroke="#000000" stroke-width="0.75" points="0.375,0.375 0.375,2.855 3.918,2.855 '
             '3.918,4.981 0.375,4.981 0.375,7.108 3.918,7.108 3.918,9.234 0.375,9.234 0.375,11.36 3.918,11.36 3.918,13.486 '
             '0.375,13.486 0.375,15.611 3.918,15.611 3.918,17.738 0.375,17.738 0.375,19.863 3.918,19.863 3.918,21.989 '
             '0.375,21.989 0.375,24.115 3.918,24.115 3.918,26.241 0.375,26.241 0.375,28.722 10.295,28.722 10.295,0.375 "/>'
             '<rect x="7.461" y="0.375" stroke="#000000" stroke-width="0.75" width="2.834" height="28.347"/>'),
    "lens": ((-4.627, -11.713),
             '<linearGradient id="{id}" gradientUnits="userSpaceOnUse" x1="4.627" y1="23.0518" x2="4.627" y2="0.3745">'
             + STOPS + '</linearGradient>'
             '<path fill="url(#{id})" stroke="#000000" stroke-width="0.75" d="M7.17,0.375H2.084C1.049,2.96,0.375,7.07,0.375,11.714'
             'c0,4.644,0.674,8.754,1.709,11.338H7.17c1.037-2.584,1.709-6.694,1.709-11.338C8.879,7.07,8.205,2.96,7.17,0.375z"/>'),
    "bsp": ((-11.713, -11.713),
            '<linearGradient id="{id}" gradientUnits="userSpaceOnUse" x1="44.0039" y1="105.2021" x2="72.3501" y2="76.856" '
            'gradientTransform="matrix(0.8 0 0 0.8 -34.8284 -61.1108)"><stop offset="0" style="stop-color:#FFFFFF"/>'
            '<stop offset="1" style="stop-color:#89A4B6"/></linearGradient>'
            '<rect x="0.375" y="0.375" fill="url(#{id})" stroke="#000000" stroke-width="0.75" width="22.677" height="22.677"/>'
            '<line fill="none" stroke="#000000" stroke-width="0.75" x1="0.375" y1="0.375" x2="23.052" y2="23.052"/>'),
}
_count = {}


def symbol(kind, x, y, rot, scale):
    _count[kind] = _count.get(kind, 0) + 1
    (tx, ty), body = SYMBOLS[kind]
    gid = f"b-{kind}-{_count[kind]}"
    return (f'<g transform="translate({x:.3f},{y:.3f}) rotate({rot:.3f}) scale({scale}) translate({tx},{ty})">'
            f'<g id="{gid}">{body.format(id=gid + "-grad")}</g></g>')


def unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def mirror_face(center, normal, scale):
    """Mirror whose reflecting face is centred on `center`, facing along `normal`."""
    return symbol("mir", center[0], center[1], math.degrees(math.atan2(-normal[1], -normal[0])), scale)


def mirror(pt, d_in, d_out, scale):
    """Fold mirror at the beam point `pt`, taking direction d_in to d_out."""
    d_in, d_out = unit(d_in), unit(d_out)
    return mirror_face(pt, unit((d_out[0] - d_in[0], d_out[1] - d_in[1])), scale)


# ---- primitives -----------------------------------------------------------
def pts(points):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in points)


def line(points, color, width=1.3, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{pts(points)}" fill="none" stroke="{color}" stroke-width="{width}"'
            f' stroke-linejoin="round" stroke-linecap="butt"{d}{extra}/>')


def head(tip, direction, color, length=9.0, half=3.6):
    ux, uy = unit(direction)
    bx, by = tip[0] - length * ux, tip[1] - length * uy
    return (f'<polygon points="{pts([tip, (bx - half * uy, by + half * ux), (bx + half * uy, by - half * ux)])}"'
            f' fill="{color}" stroke="none"/>')


def text(x, y, s, color="#000000", anchor="middle"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" style="font-size:{FONT}px;font-family:\'CMU Serif\';'
            f'text-anchor:{anchor};fill:{color}">{s}</text>')


def sub(s, i):
    return f'{s}<tspan style="font-size:65%;baseline-shift:sub">{i}</tspan>'

# ---- layout (table inches: x from the left edge, y up from the near edge) ----
S = 10.0                          # user units per inch
CM = 1 / 2.54                     # inches per cm

IN_START, M_IN1, M_IN2 = (24.5, 39.0), (24.5, 32.8), (29.6, 32.8)

# The gratings are used near Littrow at the centre wavelength: the beam onto the first grating
# of each 4f comes in DEVIATION off the 4f axis (seen from above; positive from the far side of
# the table), so sin(alpha) + sin(beta) = lambda / d with alpha - beta = DEVIATION. Beams into
# and out of a 4f overlap from above, at different heights: the 4f is entered by a mirror on
# that line between the first grating and lens and left by a pick-off mirror nearer the
# grating. The second grating and the roof retroreflector mirror this about the Fourier plane.
LAMBDA_NM = 800.0
DEVIATION = 25                   # deg
ORDER = -1                        # first order used, +1 or -1: which way the first grating is turned off the axis

# Each modified 4f: grating at f before the first lens, lenses 2 f apart (f1 = f2), and the
# second grating at f + dL after the second lens, i.e. dL from its focal plane.
# pre-stretcher
F1, DL1 = 25.0 * CM, -15.0 * CM
Y_S = 21.8
X_G1 = 17.2
X_L1, X_L2 = X_G1 + F1, X_G1 + 3 * F1
X_G2 = X_L2 + F1 + DL1
GROOVES1 = 1800.0                 # G1, G2 (per mm)
RR1_DIST = 7.5                    # G2 to its roof retroreflector (in), out of the beam above the 8020
X_IM1, X_PO1 = 32.0, 29.5         # input mirror and pick-off, on the line onto G1 (past L1)
M_S = (X_PO1, 16.5)               # straight below PO1, so its beam clears L1 on the L2 side
PBS1 = (19.6, 16.5)

# shaper arm
F3, DL3 = 20.0 * CM, 5.0 * CM
X_A = 14.0                        # the two fold mirrors after PBS1
M_A1, M_A2, M_A3 = (X_A, 16.5), (X_A, 13.7), (17.1, 13.7)
Y_A = 3.6
X_G3 = 8.2
X_L3, X_L4 = X_G3 + F3, X_G3 + 3 * F3
X_G4 = X_L4 + F3 + DL3
GROOVES3 = 1500.0                 # G3, G4 (per mm)
RR2_DIST = 4.1
X_IM2, X_PO2 = 17.0, 14.5         # past L3, like IM1 and PO1
M_A4 = (7.4, 9.3)
PBS2 = (21.5, 9.3)

# delay arm: from below PBS1 (Y_D) up to the mirror at (X_D, Y_D_OUT), down to the retroreflector,
# up to (X_D_RET, Y_D_BACK) and back to above PBS2 (Y_D_RET). The outgoing leg passes above
# the (X_D_RET, Y_D_BACK) mirror.
Y_D, Y_D_RET, Y_RR = 13.2, 11.7, 2.0
X_D, X_D_RET = 52.6, 49.7
Y_D_OUT, Y_D_BACK = 20.0, 18.0
# path compensator between (X_D_RET, Y_D_BACK) and the mirror above PBS2: the beam zigzags left to
# DC1, back right to DC2 and on; DC1's x (X_DC1, below) is solved so that DA is as long as TA
Y_DC1, DC2 = 14.8, (47.0, 12.6)

M_OUT = (38.3, 9.3)
Y_CORK0, Y_CORK1 = 30.0, 39.0     # output corkscrew

# table and the rectangles on it the beams must avoid (x0, y0, x1, y1). The png ends 51.6 in up
# the table, so that is where TABLE and the rectangles running off the picture stop.
TABLE = (0.0, 0.0, 56.0, 51.6)
OBSTACLES = [
    (2.0, 8.0, 4.0, 38.0), (17.0, 25.0, 19.0, 51.6),    # black
    (1.0, 13.0, 9.0, 21.0), (15.0, 28.0, 23.0, 36.0),   # pink
    (26.0, 36.0, 36.0, 46.0),                           # teal
    (44.0, 34.0, 56.0, 51.6),                           # beige
    (43.0, 24.0, 56.0, 26.0), (54.0, 0.0, 56.0, 26.0),  # dark grey
]
TABLE_EDGE = "#808080"
OBST_EDGE, OBST_FILL, OBST_TEXT = "#f0a030", "#fff4e3", "#b06a00"
# what the obstacles are: centre (table in), lines of text, rotation (deg)
OBSTACLE_LABELS = [
    ((5.0, 17.0), ["fCFG", "ARM"], 0),
    ((19.0, 32.0), ["fCFG", "ARM"], 0),
    ((31.0, 41.0), ["fCFG", "OUTPUT"], 0),
    ((50.0, 42.8), ["TOPAS"], 0),
    ((55.0, 12.0), ["8020"], -90),
]

X0, X1, Y0, Y1 = -8.0, 568.0, -524.0, 8.0
WIDTH_IN = 3.375
HALF_SPREAD = 7.0                 # chief-ray half height between the lenses (user units)
MIR, GRAT, BSP, LENS = 0.5, 0.6, 0.55, 0.9
MIR_HALF = 28.346 * MIR / 2


def p(x, y):
    """Table inches to user units."""
    return (S * x, -S * y)


def along(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


# ---- modified 4f ----------------------------------------------------------
class FourF:
    def __init__(self, x_g1, x_l1, x_l2, x_g2, y, grooves_per_mm, rr_dist):
        self.g1, self.g2 = p(x_g1, y), p(x_g2, y)
        self.x_l1, self.x_l2, self.y = S * x_l1, S * x_l2, -S * y
        self.focus = (self.x_l2 + (self.x_l2 - self.x_l1) / 2, self.y)
        # the beam comes in from table angle dev and leaves along +x, so the G1 normal is at
        # dev / 2 + ORDER * theta, with 2 sin(theta) cos(dev / 2) = lambda / d (counter-clockwise
        # on the table); G2 mirrors G1 about the Fourier plane
        dev = math.radians(DEVIATION)
        th = math.asin(LAMBDA_NM * 1e-6 * grooves_per_mm / (2 * math.cos(dev / 2)))
        phi = dev / 2 + ORDER * th
        self.alpha, self.beta = abs(math.degrees(dev - phi)), abs(math.degrees(phi))
        self.n1 = (math.cos(phi), -math.sin(phi))
        self.n2 = (-self.n1[0], self.n1[1])
        self.u_in = (math.cos(dev), -math.sin(dev))                # from G1 towards the input mirror
        self.rr_dir = (-self.u_in[0], self.u_in[1])               # from G2 towards the RR
        self.rr = (self.g2[0] + S * rr_dist * self.rr_dir[0], self.g2[1] + S * rr_dist * self.rr_dir[1])

    def on_input(self, x):
        """The point at table x on the line into and out of G1."""
        t = (S * x - self.g1[0]) / self.u_in[0]
        return (self.g1[0] + t * self.u_in[0], self.g1[1] + t * self.u_in[1])

    def ray(self, s):
        """Chief ray from G1 off the G2 face onto the RR (it retraces itself from there, a level up)."""
        y = self.y - HALF_SPREAD + 2 * HALF_SPREAD * s / N_BANDS
        a = (self.x_l2, y)
        u = _dot(_sub(self.g2, a), self.n2) / _dot(_sub(self.focus, a), self.n2)
        hit = along(a, self.focus, u)
        t = _dot(_sub(self.rr, hit), self.rr_dir)
        return [self.g1, (self.x_l1, y), a, hit, (hit[0] + t * self.rr_dir[0], hit[1] + t * self.rr_dir[1])]

    def fan(self):
        out = []
        for k in range(N_BANDS):
            sa, sb = max(k - 0.25, 0.0), min(k + 1.25, float(N_BANDS))
            poly = self.ray(sa) + self.ray(sb)[::-1]
            out.append(f'<polygon points="{pts(poly)}" fill="#{BAND_COLORS[k]}" fill-opacity="{BAND_OPACITY[k]:.3f}" stroke="none"/>')
        return out

    def optics(self):
        return [
            symbol("grat", *self.g1, math.degrees(math.atan2(-self.n1[1], -self.n1[0])), GRAT),
            symbol("grat", *self.g2, math.degrees(math.atan2(-self.n2[1], -self.n2[0])), GRAT),
            symbol("lens", self.x_l1, self.y, 0, LENS),
            symbol("lens", self.x_l2, self.y, 0, LENS),
            mirror_face(self.rr, (-self.rr_dir[0], -self.rr_dir[1]), MIR),
        ]


STRETCHER = FourF(X_G1, X_L1, X_L2, X_G2, Y_S, GROOVES1, RR1_DIST)
SHAPER = FourF(X_G3, X_L3, X_L4, X_G4, Y_A, GROOVES3, RR2_DIST)
IM1, PO1 = STRETCHER.on_input(X_IM1), STRETCHER.on_input(X_PO1)
IM2, PO2 = SHAPER.on_input(X_IM2), SHAPER.on_input(X_PO2)


def fold(pts_in):
    """Fold mirrors at the inner points of a beam polyline (SVG units)."""
    return [mirror(b, _sub(b, a), _sub(c, b), MIR) for a, b, c in zip(pts_in, pts_in[1:], pts_in[2:])]


def over_mirror(pt, a, b, color):
    """The incoming beam a -> b drawn across the pick-off mirror at pt (it passes above it)."""
    d = unit(_sub(b, a))
    r = 1.3 * MIR_HALF
    return line([(pt[0] - r * d[0], pt[1] - r * d[1]), (pt[0] + r * d[0], pt[1] + r * d[1])], color)


def union_outline(rects):
    """Outline of the union of axis-aligned rectangles, as closed loops of corners (table inches)."""
    xs = sorted({v for r in rects for v in (r[0], r[2])})
    ys = sorted({v for r in rects for v in (r[1], r[3])})
    edges = set()                 # directed cell edges, counter-clockwise; shared ones cancel
    for xa, xb in zip(xs, xs[1:]):
        for ya, yb in zip(ys, ys[1:]):
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            if not any(r[0] < xm < r[2] and r[1] < ym < r[3] for r in rects):
                continue
            c = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
            for e in zip(c, c[1:] + c[:1]):
                if e[::-1] in edges:
                    edges.remove(e[::-1])
                else:
                    edges.add(e)
    nxt = dict(edges)
    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, pt = [start], nxt.pop(start)
        while pt != start:
            loop.append(pt)
            pt = nxt.pop(pt)
        # keep the corners only
        loops.append([b for a, b, c in zip(loop[-1:] + loop[:-1], loop, loop[1:] + loop[:1])
                      if (b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0])])
    return loops


def block_label(x, y, lines, rot):
    """Lines of text centred on the table point (x, y), turned by rot."""
    cx, cy = p(x, y)
    lh = 1.1 * FONT
    y0 = cy - lh * (len(lines) - 1) / 2 + 0.35 * FONT
    items = "".join(text(cx, y0 + i * lh, s, OBST_TEXT) for i, s in enumerate(lines))
    return f'<g transform="rotate({rot} {cx:.3f} {cy:.3f})">{items}</g>'


def corkscrew_up(x, ya, yb):
    """The corkscrew of fig_setup_schematic.py, turned to run up the table from ya to yb."""
    import base64
    import struct
    data = config.CORKSCREW_PNG.read_bytes()
    w_px, h_px = struct.unpack(">II", data[16:24])
    w = yb - ya
    h = w * h_px / w_px
    href = "data:image/png;base64," + base64.b64encode(data).decode()
    return (f'<g transform="translate({x:.3f},{(ya + yb) / 2:.3f}) rotate(90)">'
            f'<image x="{-w / 2:.3f}" y="{-h / 2:.3f}" width="{w:.3f}" height="{h:.3f}" '
            f'preserveAspectRatio="none" xlink:href="{href}"/></g>')


def beam_paths(x_dc1=None):
    """Beam polylines (SVG units), each starting and ending on a non-mirror element."""
    x_dc1 = X_DC1 if x_dc1 is None else x_dc1
    path_in = [p(*IN_START), p(*M_IN1), p(*M_IN2), IM1, STRETCHER.g1]
    path_pre = [STRETCHER.g1, PO1, p(*M_S), p(*PBS1)]
    path_a_in = [p(*PBS1), p(*M_A1), p(*M_A2), p(*M_A3), IM2, SHAPER.g1]
    path_a_out = [SHAPER.g1, PO2, p(*M_A4), p(*PBS2)]
    path_d = [p(*PBS1), p(PBS1[0], Y_D), p(X_D, Y_D_OUT), p(X_D, Y_RR), p(X_D_RET, Y_RR),
              p(X_D_RET, Y_D_BACK), p(x_dc1, Y_DC1), p(*DC2), p(PBS2[0], Y_D_RET), p(*PBS2)]
    path_out = [p(*PBS2), p(*M_OUT), p(M_OUT[0], Y_CORK0 + 0.15)]
    return path_in, path_pre, path_a_in, path_a_out, path_d, path_out


def arm_lengths(x_dc1=None):
    """Geometric PBS1 -> PBS2 lengths (in) of TA and DA, along the central ray in air."""
    def length(path):
        return sum(math.dist(a, b) for a, b in zip(path, path[1:])) / S
    _, _, path_a_in, path_a_out, path_d, _ = beam_paths(x_dc1)
    ta = length(path_a_in) + 2 * (X_G4 - X_G3 + RR2_DIST) + length(path_a_out)
    return ta, length(path_d)


def _solve_x_dc1():
    """x of DC1 at which DA = TA (DA gets shorter as DC1 moves right, towards DC2)."""
    lo, hi = PBS2[0] + 2.0, DC2[0] - 2.0
    if not arm_lengths(hi)[1] <= arm_lengths(hi)[0] <= arm_lengths(lo)[1]:
        raise ValueError("the compensator cannot match DA to TA: move Y_DC1 or DC2")
    for _ in range(50):
        mid = (lo + hi) / 2
        ta, da = arm_lengths(mid)
        lo, hi = (mid, hi) if da > ta else (lo, mid)
    return (lo + hi) / 2


X_DC1 = _solve_x_dc1()


# ---- assemble -------------------------------------------------------------
def build():
    table, dims, beams, optics, over, labels = [], [], [], [], [], []

    # table edge, and the outline of everything on it that is in the way
    tx0, ty0 = p(TABLE[0], TABLE[3])
    tx1, ty1 = p(TABLE[2], TABLE[1])
    table.append(f'<rect x="{tx0:.3f}" y="{ty0:.3f}" width="{tx1 - tx0:.3f}" height="{ty1 - ty0:.3f}"'
                 f' fill="none" stroke="{TABLE_EDGE}" stroke-width="1.3"/>')
    d = " ".join("M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in (p(*c) for c in loop)) + " Z"
                 for loop in union_outline(OBSTACLES))
    table.append(f'<path d="{d}" fill="{OBST_FILL}" stroke="{OBST_EDGE}" stroke-width="1.0" stroke-linejoin="miter"/>')
    table += [block_label(*xy, lines, rot) for xy, lines, rot in OBSTACLE_LABELS]

    path_in, path_pre, path_a_in, path_a_out, path_d, path_out = beam_paths()

    beams += STRETCHER.fan() + SHAPER.fan()
    beams += [line(path_in, PURPLE), line(path_pre, PURPLE), line(path_a_in, RED), line(path_a_out, RED),
              line(path_d, BLUE), line(path_out, PURPLE)]
    beams.append(head(p(IN_START[0], IN_START[1] - 3.0), (0, 1), PURPLE))
    over.append(over_mirror(PO1, IM1, STRETCHER.g1, PURPLE))
    over.append(over_mirror(PO2, IM2, SHAPER.g1, RED))
    over.append(corkscrew_up(S * M_OUT[0], -S * Y_CORK1, -S * Y_CORK0))

    # optics
    optics += STRETCHER.optics() + SHAPER.optics()
    optics += fold(path_in) + fold(path_pre) + fold(path_a_in) + fold(path_a_out)
    optics += fold(path_d) + fold(path_out)
    optics += [symbol("bsp", *p(*PBS1), 90, BSP), symbol("bsp", *p(*PBS2), 0, BSP)]

    # 10 in scale bar, in the free corner at top left
    xa, xb, yb = S * 5.0, S * 15.0, -S * 45.0
    dims.append(line([(xa, yb - 4), (xa, yb), (xb, yb), (xb, yb - 4)], DIM, 0.9))
    labels.append(text((xa + xb) / 2, yb - 5, "10 in", DIM))
    ta, da = arm_lengths()
    labels += [text(xa, yb + 1.6 * FONT, f"TA, PBS{sub('', 1)} to PBS{sub('', 2)}: {ta:.1f} in", RED, "start"),
               text(xa, yb + 2.9 * FONT, f"DA, PBS{sub('', 1)} to PBS{sub('', 2)}: {da:.1f} in", BLUE, "start")]

    # labels
    g1, g2, g3, g4 = STRETCHER.g1, STRETCHER.g2, SHAPER.g1, SHAPER.g2
    labels += [
        text(S * IN_START[0] - 6, -S * IN_START[1] + 14, "in", PURPLE, "end"),
        text(S * M_OUT[0], -S * Y_CORK1 - 6, "out", PURPLE),
        text(g1[0] - 10, g1[1] + 4, sub("G", 1), anchor="end"),
        text(g2[0] + 10, g2[1] + 2, sub("G", 2), anchor="start"),
        text(STRETCHER.x_l1, STRETCHER.y + 20, sub("L", 1)),
        text(STRETCHER.x_l2, STRETCHER.y + 20, sub("L", 2)),
        text(STRETCHER.rr[0], STRETCHER.rr[1] - 12, "RR"),
        text(IM1[0] + 10, IM1[1] + 6, sub("IM", 1), anchor="start"),
        text(PO1[0] - 8, PO1[1] - 8, sub("PO", 1), anchor="end"),
        text(g3[0] - 10, g3[1] + 4, sub("G", 3), anchor="end"),
        text(g4[0] + 8, g4[1] + 12, sub("G", 4), anchor="start"),
        text(SHAPER.x_l1, SHAPER.y + 20, sub("L", 3)),
        text(SHAPER.x_l2, SHAPER.y + 20, sub("L", 4)),
        text(SHAPER.rr[0], SHAPER.rr[1] - 12, "RR"),
        text(IM2[0] + 8, IM2[1] + 3, sub("IM", 2), anchor="start"),
        text(PO2[0] - 22, PO2[1] + 2.5, sub("PO", 2), anchor="end"),
        text(S * PBS1[0] - 6, -S * PBS1[1] - 12, sub("PBS", 1)),
        text(S * PBS2[0], -S * PBS2[1] + 18, sub("PBS", 2)),
        text(S * X_D_RET - 12, -S * Y_RR + 6, "RR", BLUE, "end"),
        text(S * 36.0, -S * 17.2, "DA", BLUE),
        text(S * X_A - 12, -S * (M_A1[1] + M_A2[1]) / 2 + 7, "TA", RED, "end"),
        text(S * 33.8, -S * 19.6, "stretcher", DIM),
    ]

    def layer(i, name, items):
        return f'<g inkscape:groupmode="layer" inkscape:label="{name}" id="layer{i}">\n' + "\n".join(items) + "\n</g>"

    w, h = X1 - X0, Y1 - Y0
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Generated by Office_Apps/misc_scripts/stretch_compensate_cfCFG/fig_setup_layout.py.\n"
        "     Optics from Alexander Franzen's ComponentLibrary (gwoptics.org), CC BY-NC 3.0. -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{WIDTH_IN}in" '
        f'height="{WIDTH_IN * h / w:.4f}in" viewBox="{X0} {Y0} {w} {h}">',
        layer(0, "Table", table),
        layer(1, "Dimensions", dims),
        layer(2, "Beams", beams),
        layer(3, "Optics", optics),
        layer(4, "Beams over optics", over),
        layer(5, "Labels", labels),
        "</svg>\n",
    ])


if __name__ == "__main__":
    SVG.write_text(build(), encoding="utf-8")
    print(f"wrote {SVG}")
    for name, f4, grooves in (("G1/G2", STRETCHER, GROOVES1), ("G3/G4", SHAPER, GROOVES3)):
        print(f"{name} ({grooves:.0f}/mm at {LAMBDA_NM:.0f} nm, {DEVIATION:.0f} deg deviation): "
              f"incidence {f4.alpha:.1f} deg, diffraction {f4.beta:.1f} deg")
    ta, da = arm_lengths()
    print(f"TA {ta:.1f} in, DA {da:.1f} in: TA - DA = {ta - da:.1f} in = {(ta - da) * 2.54:.1f} cm")
    if config.INKSCAPE.exists():
        with tempfile.TemporaryDirectory() as tmp:
            conf = Path(tmp) / "fonts.conf"
            conf.write_text(config.FONTS_CONF, encoding="utf-8")
            subprocess.run([str(config.INKSCAPE), str(SVG), "--export-type=pdf", f"--export-filename={PDF}"],
                           check=True, env=dict(os.environ, FONTCONFIG_FILE=str(conf)))
        print(f"wrote {PDF}")
    else:
        print(f"Inkscape not found at {config.INKSCAPE}: PDF not exported")
