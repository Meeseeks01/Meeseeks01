#!/usr/bin/env python3
"""
contribution_chaos.py
Render a GitHub contribution graph dissolving into chaos via chaotic advection.

Usage:
    pip install numpy matplotlib imageio pillow
    python contribution_chaos.py <username> [-o chaos.gif]

Seeds particles on every contribution cell, then advects them through a
time-dependent, divergence-free vortex flow (integrated with RK4). The flow is
a chaotic Hamiltonian, so the recognizable graph stretches and folds into
chaotic filaments -- the signature of sensitive dependence on initial conditions.
"""
import argparse, re, datetime, os, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PAL = {0:"#2d333b",1:"#0e4429",2:"#006d32",3:"#26a641",4:"#39d353"}
BG  = "#0d1117"
HD, GRID_TOP = 14.0, 10.0
W_PX, H_PX, DPI = 840, 222, 100
DT, NF, HOLD, FINAL_HOLD, KTRAIL = 0.06, 165, 12, 14, 6
MODES = [(8,2,0.95),(5,3,1.27),(11,1,1.71),(3,2,0.61)]
AMP = 5.0


def fetch_cells(user):
    url = f"https://github.com/users/{user}/contributions"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    pairs = re.findall(r'data-date="(\d{4}-\d{2}-\d{2})"[^>]*?data-level="(\d)"', html)
    days = sorted((datetime.date.fromisoformat(d), int(l)) for d, l in pairs)
    start = days[0][0] - datetime.timedelta(days=(days[0][0].weekday()+1) % 7)
    return [((d-start).days//7, (d.weekday()+1) % 7, lvl) for d, lvl in days]


def hex2rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


def render(cells, user, out):
    ncols = max(c for c, _, _ in cells)+1
    WD = float(ncols)
    P, RGB, A = [], [], []
    for col, row, lvl in cells:
        cx, cy = col+0.5, GRID_TOP-row
        g = np.linspace(-0.42, 0.42, 4) if lvl >= 1 else np.linspace(-0.4, 0.4, 3)
        n = 16 if lvl >= 1 else 6
        pts = [(cx+dx, cy+dy) for dx in g for dy in g][:n]
        base, a = hex2rgb(PAL[lvl]), (0.9 if lvl >= 1 else 0.32)
        for p in pts:
            P.append(p); RGB.append(base); A.append(a)
    P = np.array(P, float); RGB = np.array(RGB); A = np.array(A)

    def vel(Q, t):
        x, y = Q[:, 0], Q[:, 1]; u = np.zeros(len(Q)); v = np.zeros(len(Q))
        for i, (n, m, w) in enumerate(MODES):
            kx, ky = n*np.pi/WD, m*np.pi/HD
            ai = AMP*np.sin(w*t + i*1.3)
            u += ai*np.sin(kx*x)*ky*np.cos(ky*y)
            v += -ai*kx*np.cos(kx*x)*np.sin(ky*y)
        return np.stack([u, v], 1)

    def rk4(Q, t, dt):
        k1 = vel(Q, t); k2 = vel(Q+0.5*dt*k1, t+0.5*dt)
        k3 = vel(Q+0.5*dt*k2, t+0.5*dt); k4 = vel(Q+dt*k3, t+dt)
        return Q+(dt/6)*(k1+2*k2+2*k3+k4)

    def frame(hist, t):
        fig = plt.figure(figsize=(W_PX/DPI, H_PX/DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-0.5, WD+0.5); ax.set_ylim(-0.5, HD+0.5)
        ax.set_aspect("equal"); ax.axis("off")
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        K = len(hist)
        for j, Pj in enumerate(hist):
            rgba = np.concatenate([RGB, (A*((j+1)/K)**1.6)[:, None]], 1)
            ax.scatter(Pj[:, 0], Pj[:, 1], s=6.0, c=rgba, linewidths=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig); return buf

    frames = [frame([P], 0.0) for _ in range(HOLD)]
    hist, cur = [P.copy()], P.copy()
    for f in range(NF):
        cur = rk4(cur, f*DT, DT); hist.append(cur.copy()); hist = hist[-KTRAIL:]
        frames.append(frame(hist, f*DT))
    frames += [frames[-1]]*FINAL_HOLD

    pil = [Image.fromarray(a) for a in frames]
    if out.endswith(".webp"):
        pil[0].save(out, save_all=True, append_images=pil[1:], duration=int(1000/30),
                    loop=0, quality=72, method=6)
    else:
        master = pil[len(pil)//2].convert("RGB").quantize(colors=48, method=Image.FASTOCTREE)
        pf = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in pil]
        pf[0].save(out, save_all=True, append_images=pf[1:], duration=1000/30,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("username"); ap.add_argument("-o", "--output", default="chaos.gif")
    a = ap.parse_args()
    render(fetch_cells(a.username), a.username, a.output)