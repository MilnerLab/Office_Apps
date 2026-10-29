"""Setup schematic of the modified usCFG shaper (fig:cad).

Writes figures/fig_setup_schematic.svg in the paper repo, then exports the PDF
with Inkscape. Optics symbols are from Alexander Franzen's ComponentLibrary
(gwoptics.org), CC BY-NC 3.0.

The 4f is drawn with chief rays: they fan out from G1, run parallel to the
axis between L1 and L2, and refocus on the focal plane of L2 (L = 0).
"""
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from manuscript_plotting_scripts.shaped_usCFG_paper import config

SVG = config.FIGURES_DIR / "fig_setup_schematic.svg"
PDF = config.FIGURES_DIR / "fig_setup_schematic.pdf"
INKSCAPE = Path(shutil.which("inkscape") or r"C:\Program Files\Inkscape\bin\inkscape.exe")


def _cmu_dir() -> Path:
    """The TeX distribution's CMU fonts, found with kpsewhich wherever TeX is installed."""
    try:
        hit = subprocess.run(["kpsewhich", "cmunrm.otf"], capture_output=True, text=True).stdout.strip()
    except OSError:
        hit = ""
    return Path(hit).parent if hit else Path(r"C:\texlive\2025\texmf-dist\fonts\opentype\public\cm-unicode")


# Computer Modern, to match the manuscript and the matplotlib figures. CMU Serif ships with
# TeX Live, so Inkscape is pointed at it through a fontconfig file rather than an install.
CMU_DIR = _cmu_dir()
FONTS_CONF = ('<?xml version="1.0"?><fontconfig>'
              f'<include ignore_missing="yes">{INKSCAPE.parents[1] / "etc" / "fonts" / "fonts.conf"}</include>'
              f'<dir>{CMU_DIR}</dir></fontconfig>')

# ---- layout (user units) -------------------------------------------------
X0, X1, Y0, Y1 = -4.0, 536.0, -280.0, 18.0
WIDTH_IN = 3.375
FONT = 20.0                       # ~9 pt at 3.375 in
PURPLE, BLUE, RED = "#d42ad4", "#2a6cff", "#ff2a2a"
DIM = "#000000"
GREEN_EDGE, GREEN_FILL = "#3c9a3c", "#e6f4e1"
ORANGE_EDGE, ORANGE_FILL = "#f0a030", "#fff4e3"

Y_IN, Y_OUT = -110.0, -150.0      # input line, output line
Y_DA, Y_DA_RET = -222.0, -205.0   # delay arm out and back (mirrors meet at the RR corner)
X_START = 0.0                     # left end of in/out beams
X_HWP, X_PBS1, X_QWP, X_PBS2 = 38.0, 80.0, 110.0, 140.0
X_PICK, X_FOLD = 190.0, 245.0     # TA pick-off mirror, fold mirror to G1
X_CORK0, X_CORK1 = 2.0, 76.0      # output corkscrew

F = 70.0
Y4F = -60.0
X_G1 = 210.0
X_L1, X_FP, X_L2, X_FOC = X_G1 + F, X_G1 + 2 * F, X_G1 + 3 * F, X_G1 + 4 * F
L_DRAWN = 26.0
X_G2 = X_FOC + L_DRAWN
HALF_SPREAD = 10.0                # chief-ray half height between L1 and L2
G2_TILT = 15.0                    # deg
_d = (-23.6, -37.1)                 # collimated beam from G2 to its retroreflector
RR_DIR = (_d[0] / math.hypot(*_d), _d[1] / math.hypot(*_d))
RR_DIST = 42.0                      # G2 to where the central ray meets the RR
RR_CORNER = 0.0                     # hit point to the roof corner: the beam lands on the corner
X_RR_DA = X_G2 - 14.071             # delay-arm retroreflector corner

# prism: the bluest bands (s >= S_CUT of 64) are sent to the beam dump
N_BANDS = 64
S_CUT = 48.0
PRISM_H, PRISM_W = 16.0, 9.0
DEFLECT = math.radians(32.6)
BD_DIST = 34.0

MIR, GRAT, BSP, LENS, DUMP = 0.85, 0.9, 0.8, 1.1, 0.65

# band colours and opacities, red (s = 0) to violet (s = 64)
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
    "dump": ((0.0, -16.566),
             '<linearGradient id="{id}" gradientUnits="userSpaceOnUse" x1="55.9932" y1="110.8086" x2="55.9932" y2="81.2568" '
             'gradientTransform="matrix(0.8 0 0 0.8 -33.8884 -60.2056)"><stop offset="0" style="stop-color:#FFFFFF"/>'
             '<stop offset="0.1314" style="stop-color:#C4C4C4"/><stop offset="0.3752" style="stop-color:#5C5C5C"/>'
             '<stop offset="0.5444" style="stop-color:#1A1A1A"/><stop offset="0.6236" style="stop-color:#000000"/>'
             '<stop offset="0.6657" style="stop-color:#151515"/><stop offset="0.7529" style="stop-color:#4C4C4C"/>'
             '<stop offset="0.8765" style="stop-color:#A2A2A2"/><stop offset="1" style="stop-color:#FFFFFF"/></linearGradient>'
             '<rect x="1.177" y="4.801" fill="url(#{id})" stroke="#000000" stroke-width="0.75" width="19.463" height="23.642"/>'
             '<rect x="0.5" y="0.5" stroke="#000000" width="2.57" height="32.131"/>'
             '<rect x="6.785" y="0.5" stroke="#000000" width="3.815" height="32.131"/>'
             '<rect x="14.206" y="0.5" stroke="#000000" width="7.61" height="32.131"/>'),
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


def mirror_face(center, normal, scale=MIR):
    """Mirror whose reflecting face is centred on `center`, facing along `normal`."""
    return symbol("mir", center[0], center[1], math.degrees(math.atan2(-normal[1], -normal[0])), scale)


def mirror(pt, d_in, d_out, scale=MIR):
    """Fold mirror at the beam point `pt`, taking direction d_in to d_out."""
    d_in, d_out = unit(d_in), unit(d_out)
    return mirror_face(pt, unit((d_out[0] - d_in[0], d_out[1] - d_in[1])), scale)


MIR_HALF = 28.346 * MIR / 2


# ---- primitives -----------------------------------------------------------
def pts(points):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in points)


def line(points, color, width=1.3, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{pts(points)}" fill="none" stroke="{color}" stroke-width="{width}"'
            f' stroke-linejoin="round" stroke-linecap="butt"{d}{extra}/>')


def head(tip, direction, color, length=9.0, half=3.6):
    ux, uy = direction
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    bx, by = tip[0] - length * ux, tip[1] - length * uy
    return (f'<polygon points="{pts([tip, (bx - half * uy, by + half * ux), (bx + half * uy, by - half * ux)])}"'
            f' fill="{color}" stroke="none"/>')


def dimension(xa, xb, y, color, length=9.0, half=3.6, width=0.9):
    return (line([(xa + length - 0.5, y), (xb - length + 0.5, y)], color, width)
            + head((xb, y), (1, 0), color, length, half) + head((xa, y), (-1, 0), color, length, half))


def text(x, y, s, color="#000000", anchor="middle"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" style="font-size:{FONT}px;font-family:\'CMU Serif\';'
            f'text-anchor:{anchor};fill:{color}">{s}</text>')


def sub(s, i):
    return f'{s}<tspan style="font-size:65%;baseline-shift:sub">{i}</tspan>'


def it(s):
    return f'<tspan style="font-style:italic">{s}</tspan>'


def plate(x, y):
    return f'<rect x="{x - 1.5:.3f}" y="{y - 12:.3f}" width="3.000" height="24.000" fill="url(#glass)" stroke="#000000" stroke-width="0.75"/>'


# ---- 4f ray fan -----------------------------------------------------------
def y_chief(s):
    return Y4F - HALF_SPREAD + 2 * HALF_SPREAD * s / N_BANDS


def _rot(v, deg):
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (c * v[0] - s_ * v[1], s_ * v[0] + c * v[1])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


# lower (grating) retroreflector: a roof whose corner sits RR_CORNER beside the hit point
RR_HIT = (X_G2 + RR_DIST * RR_DIR[0], Y4F + RR_DIST * RR_DIR[1])
RR_F1 = _rot(RR_DIR, -45)            # along the first mirror, towards the corner
RR_F2 = _rot(RR_DIR, 45)
RR_C = (RR_HIT[0] + RR_CORNER * RR_F1[0], RR_HIT[1] + RR_CORNER * RR_F1[1])


def _roof_mirror(f):
    center = (RR_C[0] - MIR_HALF * f[0], RR_C[1] - MIR_HALF * f[1])
    n = (-f[1], f[0])
    if n[0] * -RR_DIR[0] + n[1] * -RR_DIR[1] < 0:
        n = (-n[0], -n[1])
    return mirror_face(center, n)


PRISM_TOP = y_chief(S_CUT)
D_HAT = (math.cos(DEFLECT), math.sin(DEFLECT))
S_MID_CUT = (S_CUT + N_BANDS) / 2
_y_mid = y_chief(S_MID_CUT)
_hyp_mid = (X_FP + (_y_mid - PRISM_TOP) * PRISM_W / PRISM_H, _y_mid)
BD_POS = (_hyp_mid[0] + BD_DIST * D_HAT[0], _hyp_mid[1] + BD_DIST * D_HAT[1])


def ray(s, deflected):
    y = y_chief(s)
    if deflected:
        hyp = (X_FP + (y - PRISM_TOP) * PRISM_W / PRISM_H, y)
        t = (BD_POS[0] - hyp[0]) * D_HAT[0] + (BD_POS[1] - hyp[1]) * D_HAT[1]
        return [(X_G1, Y4F), (X_L1, y), (X_FP, y), hyp, (hyp[0] + t * D_HAT[0], hyp[1] + t * D_HAT[1])]
    # through the focus onto the tilted G2 face, then collimated to the lower retroreflector
    m = (Y4F - y) / (X_FOC - X_L2)
    sn, cs = math.sin(math.radians(G2_TILT)), math.cos(math.radians(G2_TILT))
    tg = L_DRAWN * m / (cs - sn * m)
    g2 = (X_G2 + sn * tg, Y4F + cs * tg)
    rel = (RR_HIT[0] - g2[0], RR_HIT[1] - g2[1])
    t = _cross(rel, RR_F1) / _cross(RR_DIR, RR_F1)
    return [(X_G1, Y4F), (X_L1, y), (X_L2, y), (X_FOC, Y4F), g2, (g2[0] + t * RR_DIR[0], g2[1] + t * RR_DIR[1])]


def ray_fan():
    out = []
    for k in range(N_BANDS):
        sa, sb = max(k - 0.25, 0.0), min(k + 1.25, float(N_BANDS))
        deflected = k >= S_CUT
        poly = ray(sa, deflected) + ray(sb, deflected)[::-1]
        out.append(f'<polygon points="{pts(poly)}" fill="#{BAND_COLORS[k]}" fill-opacity="{BAND_OPACITY[k]:.3f}" stroke="none"/>')
    return out


# ---- output corkscrew -----------------------------------------------------
# Cut out of Fig. 2 of the usCFG paper (PRA 113, 033107), rotated to point along the output
# beam. Its axis runs through the middle row of the image.
CORKSCREW_PNG = config.FIGURES_DIR / "corkscrew_uscfg.png"


def corkscrew(xa, xb, y0):
    import base64
    import struct
    data = CORKSCREW_PNG.read_bytes()
    w_px, h_px = struct.unpack(">II", data[16:24])
    w = xb - xa
    h = w * h_px / w_px
    href = "data:image/png;base64," + base64.b64encode(data).decode()
    return (f'<image x="{xa:.3f}" y="{y0 - h / 2:.3f}" width="{w:.3f}" height="{h:.3f}" '
            f'preserveAspectRatio="none" xlink:href="{href}"/>')

# ---- assemble -------------------------------------------------------------
def build():
    stages, beams, optics, over, labels = [], [], [], [], []

    # stages: grating stage (orange) carries G2, its RR and the delay stage (green)
    org_x0 = X_FOC - 9.255
    stages.append(f'<rect x="{org_x0:.3f}" y="-244.000" width="{X_G2 + 14 - org_x0:.3f}" height="200.000"'
                  f' fill="{ORANGE_FILL}" stroke="{ORANGE_EDGE}" stroke-width="0.8"/>')
    grn_x0, grn_x1 = X_FOC - 2.0, X_G2 + 8.0
    stages.append(f'<rect x="{grn_x0:.3f}" y="-238.000" width="{grn_x1 - grn_x0:.3f}" height="48.000"'
                  f' fill="{GREEN_FILL}" stroke="{GREEN_EDGE}" stroke-width="0.8"/>')
    stages.append(dimension(grn_x0 + 3, grn_x1 - 3, -250.0, GREEN_EDGE, 6.0, 2.6, 0.7))
    labels.append(text((grn_x0 + grn_x1) / 2, -256.0, f"Δ{it('t')}", GREEN_EDGE))

    # prism stage (grey) and its travel
    stages.append(f'<rect x="{X_FP - 7:.1f}" y="{PRISM_TOP - 1:.3f}" width="23.0" height="23.0" fill="#eeeeee" stroke="#808080" stroke-width="0.8"/>')
    stages.append(line([(X_FP - 12, PRISM_TOP + 17), (X_FP - 12, PRISM_TOP + 4)], "#202020", 0.6)
                  + head((X_FP - 12, PRISM_TOP + 1), (0, -1), "#202020", 3.6, 1.62)
                  + head((X_FP - 12, PRISM_TOP + 20), (0, 1), "#202020", 3.6, 1.62))

    # Fourier plane (dashed) and L = 0 reference (dotted)
    stages.append(line([(X_FP, -109.0), (X_FP, -11.0)], DIM, 1.1, "4,2.5"))
    stages.append(line([(X_FOC, -72.0), (X_FOC, -11.0)], DIM, 1.1, "1.2,2"))

    # f f f f L dimension row
    xs = [X_G1, X_L1, X_FP, X_L2, X_FOC, X_G2]
    for x in xs:
        stages.append(line([(x, -17.0), (x, -5.0)], DIM, 0.9))
    for xa, xb in zip(xs, xs[1:]):
        stages.append(dimension(xa, xb, -11.0, DIM))
    for xa, xb in zip(xs[:4], xs[1:5]):
        labels.append(text((xa + xb) / 2, 11.0, it("f"), DIM))
    labels.append(text((X_FOC + X_G2) / 2 + 0.0, 11.0, it("L"), DIM))

    stages.append(f'<polygon points="{pts([(X_FP, PRISM_TOP), (X_FP, PRISM_TOP + PRISM_H), (X_FP + PRISM_W, PRISM_TOP + PRISM_H)])}"'
                  ' fill="url(#glass)" fill-opacity="0.6" stroke="#000000" stroke-width="0.75" stroke-linejoin="round"/>')

    # beams
    beams.append(line([(X_START, Y_IN), (X_PBS1, Y_IN)], PURPLE))
    beams.append(head((X_START + 26, Y_IN), (1, 0), PURPLE))
    beams.append(line([(X_PBS2, Y_OUT), (X_CORK1 - 1.5, Y_OUT)], PURPLE))
    over.append(corkscrew(X_CORK0, X_CORK1, Y_OUT))
    beams.append(line([(X_PBS1, Y_IN), (X_PBS1, Y_DA), (X_RR_DA, Y_DA), (X_RR_DA, Y_DA_RET),
                       (X_PBS2, Y_DA_RET), (X_PBS2, Y_OUT)], BLUE))
    beams.append(line([(X_PBS1, Y_IN), (X_FOLD, Y_IN), (X_G1, Y4F)], RED))
    beams.append(line([(X_PICK, Y_IN), (X_PICK, Y_OUT), (X_PBS2, Y_OUT)], RED))
    beams.extend(ray_fan())
    # the input shaper arm passes over the pick-off mirror (the return is offset vertically)
    over.append(line([(X_PICK - 16.0, Y_IN), (X_PICK, Y_IN)], RED))

    # optics
    optics += [
        mirror((X_PBS1, Y_DA), (0, -1), (1, 0)),
        mirror((X_RR_DA, Y_DA), (1, 0), (0, 1)),
        mirror((X_RR_DA, Y_DA_RET), (0, 1), (-1, 0)),
        mirror((X_PBS2, Y_DA_RET), (-1, 0), (0, 1)),
        mirror((X_FOLD, Y_IN), (1, 0), (X_G1 - X_FOLD, Y4F - Y_IN)),
        mirror((X_PICK, Y_IN), (-1, 0), (0, -1)),
        mirror((X_PICK, Y_OUT), (0, -1), (-1, 0)),
        _roof_mirror(RR_F1),
        _roof_mirror(RR_F2),
        symbol("grat", X_G1, Y4F, -165, GRAT),
        symbol("grat", X_G2, Y4F, -15, GRAT),
        symbol("lens", X_L1, Y4F, 0, LENS),
        symbol("lens", X_L2, Y4F, 0, LENS),
        symbol("dump", BD_POS[0], BD_POS[1], 32, DUMP),
        symbol("bsp", X_PBS1, Y_IN, -90, BSP),
        symbol("bsp", X_PBS2, Y_OUT, -90, BSP),
        plate(X_HWP, Y_IN),
        plate(X_QWP, Y_OUT),
    ]

    # labels
    labels += [
        text(X_PBS1, -81.0, sub("PBS", 1)),
        text(X_PBS2 + 12, -122.0, sub("PBS", 2)),
        text(X_HWP, -81.0, f"{it('λ')}/2"),
        text(X_QWP, -170.0, f"{it('λ')}/4"),
        text(X_G1 - 10, -56.0, sub("G", 1), anchor="end"),
        text(X_G2 + 4, -24.0, sub("G", 2)),
        text(X_L1, -77.0, sub("L", 1)),
        text(X_L2, -77.0, sub("L", 2)),
        text(X_FP - 17, PRISM_TOP + 21, "P", anchor="end"),
        text(X_FP, -114.0, "FP", DIM),
        text(BD_POS[0] + 24, BD_POS[1] + 12, "BD", anchor="start"),
        text(RR_C[0] - 19.0, -80.0, "RR", anchor="end"),
        text((grn_x0 + grn_x1) / 2, -168.0, "RR"),
        text(270.0, Y_DA - 6, "DA", BLUE),
        text(X_FOLD - 30, Y_IN - 5, "TA", RED),
        text(X_START + 2, Y_IN - 5, "in", PURPLE, "start"),
        text(X_START + 2, Y_OUT - 20, "out", PURPLE, "start"),
    ]

    def layer(i, name, items):
        return f'<g inkscape:groupmode="layer" inkscape:label="{name}" id="layer{i}">\n' + "\n".join(items) + "\n</g>"

    w, h = X1 - X0, Y1 - Y0
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Generated by Office_Apps/manuscript_plotting_scripts/shaped_usCFG_paper/fig_setup_schematic.py.\n"
        "     Optics from Alexander Franzen's ComponentLibrary (gwoptics.org), CC BY-NC 3.0. -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{WIDTH_IN}in" '
        f'height="{WIDTH_IN * h / w:.4f}in" viewBox="{X0} {Y0} {w} {h}">',
        '<defs><linearGradient id="glass" gradientUnits="objectBoundingBox" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#89A4B6"/></linearGradient></defs>',
        layer(0, "Stages and dimensions", stages),
        layer(1, "Beams", beams),
        layer(2, "Optics", optics),
        layer(3, "Beams over optics", over),
        layer(4, "Labels", labels),
        "</svg>\n",
    ])


if __name__ == "__main__":
    SVG.write_text(build(), encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        conf = Path(tmp) / "fonts.conf"
        conf.write_text(FONTS_CONF, encoding="utf-8")
        subprocess.run([str(INKSCAPE), str(SVG), "--export-type=pdf", f"--export-filename={PDF}"],
                       check=True, env=dict(os.environ, FONTCONFIG_FILE=str(conf)))
    print(f"wrote {SVG}\nwrote {PDF}")
