import math
from pathlib import Path

C, SN = math.cos(math.radians(30)), math.sin(math.radians(30))
SP = 62            # lattice half-spacing (px)
R = 15             # atom radius (both ions)
OUT = 2.5          # thin outline / center-line weight
LW = 7             # outer bond weight

def iso(p, ox, oy):
    x, y, z = p
    return ox + (x - z) * C * SP, oy + (x + z) * SN * SP - y * SP

def cube(ox, oy):
    pts = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)
           if i == 2 or j == 2 or k == 2]
    faces = lambda p: {ax for ax in range(3) if p[ax] == 2}
    out = []
    for p in pts:
        for ax in range(3):
            q = list(p); q[ax] += 1; q = tuple(q)
            if q in pts and (faces(p) & faces(q)) - {ax}:
                (x1, y1), (x2, y2) = iso(p, ox, oy), iso(q, ox, oy)
                L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L
                g = R + 4
                outer = len(faces(p) & faces(q)) == 2
                out.append(f'<line x1="{x1+ux*g:.1f}" y1="{y1+uy*g:.1f}" x2="{x2-ux*g:.1f}" y2="{y2-uy*g:.1f}" '
                           f'stroke-width="{LW if outer else LW*0.65:.1f}"/>')
    for p in pts:  # alchemical salt symbol: circle bisected by a horizontal line
        x, y = iso(p, ox, oy)
        cl = sum(p) % 2 == 0
        body, mid = ('#fff', '#000') if cl else ('#000', '#fff')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{R}" fill="{body}" stroke="#000" stroke-width="{OUT}"/>')
        out.append(f'<line x1="{x-R+OUT/2:.1f}" y1="{y:.1f}" x2="{x+R-OUT/2:.1f}" y2="{y:.1f}" '
                   f'stroke="{mid}" stroke-width="{OUT}" stroke-linecap="butt"/>')
    return out

def wordmark(x0, ybase, h=112, sw=15, gap=28):
    t = ybase - h; w = 0.62 * h
    S = (f'M{x0+w*0.92:.1f},{t+h*0.17:.1f} C{x0+w*0.78:.1f},{t-h*0.02:.1f} {x0+w*0.08:.1f},{t-h*0.02:.1f} {x0+w*0.08:.1f},{t+h*0.26:.1f} '
         f'C{x0+w*0.08:.1f},{t+h*0.52:.1f} {x0+w*0.95:.1f},{t+h*0.45:.1f} {x0+w*0.95:.1f},{t+h*0.74:.1f} '
         f'C{x0+w*0.95:.1f},{t+h*1.03:.1f} {x0+w*0.2:.1f},{t+h*1.03:.1f} {x0+w*0.04:.1f},{t+h*0.84:.1f}')
    x1 = x0 + w + gap; wa = 0.78 * h
    A = (f'M{x1:.1f},{ybase:.1f} L{x1+wa/2:.1f},{t:.1f} L{x1+wa:.1f},{ybase:.1f} '
         f'M{x1+wa*0.2:.1f},{t+h*0.66:.1f} L{x1+wa*0.8:.1f},{t+h*0.66:.1f}')
    x2 = x1 + wa + gap; wl = 0.5 * h
    L = f'M{x2:.1f},{t:.1f} L{x2:.1f},{ybase:.1f} L{x2+wl:.1f},{ybase:.1f}'
    x3 = x2 + wl + gap * 0.8; wt = 0.66 * h
    T = f'M{x3:.1f},{t:.1f} L{x3+wt:.1f},{t:.1f} M{x3+wt/2:.1f},{t:.1f} L{x3+wt/2:.1f},{ybase:.1f}'
    return [f'<path d="{d}" fill="none" stroke-width="{sw}"/>' for d in (S, A, L, T)], x3 + wt

SKETCH_DEFS = ('<defs><filter id="sk" x="-5%" y="-5%" width="110%" height="110%">'
               '<feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="4"/>'
               '<feDisplacementMap in="SourceGraphic" scale="3"/></filter></defs>')

def svg(w, h, body, sketch=False, bg=False):
    g = ' filter="url(#sk)"' if sketch else ''
    return '\n'.join(x for x in [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">',
        SKETCH_DEFS if sketch else '',
        '<rect width="100%" height="100%" fill="#fff"/>' if bg else '',
        f'<g stroke="#000" stroke-linecap="round" stroke-linejoin="round"{g}>', *body, '</g>', '</svg>'] if x)

pad = 24
cw, ch = 4 * C * SP, 4 * SN * SP + 2 * SP
items = cube(pad + cw / 2 + R, pad + R + 2 * SP)
mark_w, mark_h = cw + 2 * pad + 2 * R, ch + 2 * pad + 2 * R
wm, wend = wordmark(mark_w + 12, mark_h / 2 + 56)
lock_w = wend + pad + 10

output = Path(__file__).resolve().parents[1] / 'salt-logo-white.svg'
output.write_text(svg(lock_w, mark_h, items + wm, bg=True), encoding='utf-8')
print(round(mark_w), round(mark_h), round(lock_w))
