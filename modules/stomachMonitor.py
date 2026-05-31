#!/usr/bin/env python3
# Copyright (c) 2026 Nima Abaeian
# License: GNU GPL v3
#
# stomachMonitor.py  (RFModule, anatomical + cute, Pillow-rendered)
#
# A visual YARP RFModule monitor for the orexigenic (hunger) drive of
# executiveControl.py. It changes NOTHING in the controller:
#   * polls the module RPC `/executiveControl` (command `status`) for the live
#     stomach level / hunger state, and
#   * injects meals into the real QR inlet `/alwayson/executiveControl/qr:i`
#     (SMALL_MEAL / MEDIUM_MEAL / LARGE_MEAL) so meals still arrive ONLY via QR.
#
# Buttons map 1:1 onto the RPC verbs (hunger_mode, hunger <hsN>, reset, quit).
#
#   alwayson_stomachMonitor --server /executiveControl
#   alwayson_stomachMonitor --sim
#
# Requires Pillow (pip install pillow). Falls back to a basic vector view if
# Pillow is missing, so it never hard-fails on the robot.
#
from __future__ import annotations

import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time
import traceback
import tkinter as tk
from collections import deque
from tkinter import font as tkfont
from tkinter import messagebox
from typing import Any, Dict, List, Optional

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageTk
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

try:
    import yarp
    HAVE_YARP = True
except Exception:
    yarp = None
    HAVE_YARP = False

# ── constants mirrored from executiveControl.py (display only) ────────────────
MEALS = {"SMALL_MEAL": 10.0, "MEDIUM_MEAL": 25.0, "LARGE_MEAL": 45.0}
HUNGRY_THRESHOLD = 60.0
STARVING_THRESHOLD = 25.0
DRAIN_HOURS = 4.0
QR_COOLDOWN_SEC = 3.0

STATE_TEXT = {
    "HS0": "Not available",
    "HS1": "Full & Happy",
    "HS2": "Hungry",
    "HS3": "Starving",
}

# liquid, liquid-dark, empty/air, outline
PALETTE = {
    "HS1": ("#5BCB97", "#3E9E73", "#EAF8F1", "#2C8A60"),
    "HS2": ("#F4BB52", "#D69426", "#FCF2DC", "#B9821C"),
    "HS3": ("#ED7059", "#CF4A33", "#FCE7E1", "#B23A26"),
    "HS0": ("#AEB4BA", "#838B92", "#EDEFF1", "#6E767D"),
}
ACCENT = {"HS0": "#6E767D", "HS1": "#2C8A60", "HS2": "#B9821C", "HS3": "#B23A26"}

BG = "#F6F8FB"
CARD = "#FFFFFF"
PANEL_BG = "#FFFFFF"
SUBTLE = "#F1F5F9"
BORDER = "#E3EAF2"
INK = "#243040"
MUTED = "#7F8A99"
RENDER_SS = 1
MAX_RENDER_VIEW = 720

DECOR_SYMBOLS = ["♪", "♫", "✦", "✧", "♡", "•"]
DECOR_COLORS = ["#D8C7FF", "#FFD6E0", "#BFEBD8", "#FFE0A8", "#CDE7FF"]

# anatomical control points (normalized, clockwise from cardia)
SAC = [
    (0.50, 0.085), (0.34, 0.030), (0.150, 0.090), (0.055, 0.250),
    (0.070, 0.470), (0.150, 0.660), (0.310, 0.790), (0.510, 0.800),
    (0.680, 0.720), (0.820, 0.595), (0.930, 0.500), (0.880, 0.405),
    (0.700, 0.430), (0.560, 0.380), (0.520, 0.250), (0.520, 0.140),
]

VIEW = 540
FPS = 30


def hx(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - pow(1.0 - t, 3)


def catmull_rom_closed(P, n=26):
    pts = []
    m = len(P)
    for i in range(m):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for j in range(n):
            t = j / n
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            pts.append((x, y))
    return pts


# ══════════════════════════════════════════════════════════════════════════════
# Pillow renderer — anatomical, cute, anti-aliased
# ══════════════════════════════════════════════════════════════════════════════
class StomachRenderer:
    def __init__(self, view=VIEW, ss=2):
        self.ss = ss
        self.W = view
        self.w = view * ss
        s = self.w
        self.bw, self.bh = int(s * 0.74), int(s * 0.74)
        self.ox, self.oy = int(s * 0.10), int(s * 0.14)
        self.poly = [(self.ox + x * self.bw, self.oy + y * self.bh)
                     for (x, y) in catmull_rom_closed(SAC)]
        self.mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(self.mask).polygon(self.poly, fill=255)
        xs = [p[0] for p in self.poly]; ys = [p[1] for p in self.poly]
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(20 * ss))
        except Exception:
            self.font = ImageFont.load_default()
        self.shadow = self._make_shadow()

        # Pre-compute face geometry (constant per renderer instance)
        self.fcx = self.ox + 0.34 * self.bw
        self.fcy = self.oy + 0.40 * self.bh
        self.edx = 0.115 * self.bw
        self.er = 0.052 * self.bw
        self.eyl = (self.fcx - self.edx, self.fcy)
        self.eyr = (self.fcx + self.edx, self.fcy)

        # Reusable scratch buffers (avoid per-frame allocations)
        self._scratch_l = Image.new("L", (s, s), 0)
        self._scratch_rgba = Image.new("RGBA", (s, s), (0, 0, 0, 0))

        # Surface wave x-coordinates (constant, only y varies per frame)
        self._surf_x0 = self.minx - 24 * ss
        self._surf_x1 = self.maxx + 24 * ss
        self._surf_n = 80  # higher than old 56 for smoother waves
        self._surf_xs = [self._surf_x0 + (self._surf_x1 - self._surf_x0) * (k / self._surf_n)
                         for k in range(self._surf_n + 1)]

        # Cached liquid mask state (avoid recomputation when level is stable)
        self._cached_lm_key = None
        self._cached_liquid_mask = None

        # Resolve resampler once
        self._bilinear = getattr(Image, "Resampling", Image).BILINEAR
        self._lanczos = getattr(Image, "Resampling", Image).LANCZOS

        self.cache = {st: self._build_base(st) for st in PALETTE}

    def _vgrad(self, top, bot):
        s = self.w
        g = Image.new("RGB", (1, s))
        for y in range(s):
            f = y / (s - 1)
            g.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * f) for i in range(3)))
        return g.resize((s, s)).convert("RGBA")

    def _make_shadow(self):
        s, ss = self.w, self.ss
        sh = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse(
            [self.minx + 24 * ss, self.maxy - 20 * ss,
             self.maxx - 14 * ss, self.maxy + 16 * ss], fill=(40, 50, 60, 60))
        return sh.filter(ImageFilter.GaussianBlur(9 * ss))

    def _build_base(self, state):
        s, ss = self.w, self.ss
        liquid, ldark, empty, outline = [hx(c) for c in PALETTE[state]]
        edark = tuple(int(c * 0.92) for c in empty)
        base = Image.new("RGBA", (s, s), (0, 0, 0, 0))

        tube = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        td = ImageDraw.Draw(tube)
        ex = self.ox + 0.50 * self.bw
        p0 = (ex + 6 * ss, self.oy - 14 * ss)
        p1 = (ex - 2 * ss, self.oy + 0.10 * self.bh)
        for wdt, col in ((30 * ss, outline), (22 * ss, empty)):
            td.line([p0, p1], fill=col + (255,), width=int(wdt), joint="curve")
            r = wdt / 2
            for (x, y) in (p0, p1):
                td.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))
        base.alpha_composite(tube)

        base.paste(self._vgrad(empty, edark), (0, 0), self.mask)

        gloss = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        gx = self.minx + 0.30 * (self.maxx - self.minx)
        gy = self.miny + 0.20 * (self.maxy - self.miny)
        ImageDraw.Draw(gloss).ellipse(
            [gx - 34 * ss, gy - 20 * ss, gx + 28 * ss, gy + 40 * ss],
            fill=(255, 255, 255, 90))
        gloss = gloss.filter(ImageFilter.GaussianBlur(7 * ss))
        gm = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        gm.paste(gloss, (0, 0), self.mask)
        base.alpha_composite(gm)

        ImageDraw.Draw(base).line(self.poly + [self.poly[0]],
                                  fill=outline + (255,), width=int(5 * ss), joint="curve")

        # pre-rendered cheek blush (static per state — saves a blur every frame)
        blush = None
        if state != "HS0":
            col = {"HS1": (255, 150, 150, 120), "HS2": (245, 160, 120, 85),
                   "HS3": (240, 120, 110, 110)}[state]
            fcx, fcy = self.fcx, self.fcy
            edx, er = self.edx, self.er
            blush = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            cd = ImageDraw.Draw(blush)
            for sgn in (-1, 1):
                cx = fcx + sgn * (edx + er * 1.05)
                cd.ellipse([cx - er * 0.95, fcy + er * 0.55,
                            cx + er * 0.95, fcy + er * 1.75], fill=col)
            blush = blush.filter(ImageFilter.GaussianBlur(2 * ss))

        # Pre-compute per-state colors used every frame
        bub_col = tuple(min(255, c + 45) for c in liquid)
        meniscus_col = tuple(min(255, c + 42) for c in liquid) + (235,)

        return {"base": base, "liq": self._vgrad(liquid, ldark),
                "liquid": liquid, "ldark": ldark, "outline": outline,
                "blush": blush,
                "bub_col": bub_col, "meniscus_col": meniscus_col,
                "ink": tuple(int(c * 0.55) for c in outline)}

    @staticmethod
    def _clip(layer, mask):
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
        return layer

    def water_y_disp(self, level):
        frac = max(0.0, min(1.0, level / 100.0))
        return (self.maxy - frac * (self.maxy - self.miny)) / self.ss

    def _surface_points(self, water_y, phase, amp):
        ss = self.ss
        xs = self._surf_xs
        c1, c2 = phase * 1.4, phase * 2.3
        k1, k2 = 0.012 / ss, 0.03 / ss
        amp4 = amp * 0.4
        return [(x, water_y + amp * math.sin(c1 + x * k1)
                 + amp4 * math.sin(c2 + x * k2))
                for x in xs]

    def compose(self, state, level, phase, blink, bubbles, particles):
        s, ss = self.w, self.ss
        cc = self.cache[state]
        spr = cc["base"].copy()
        liquid, ink = cc["liquid"], cc["ink"]
        bub_col = cc["bub_col"]
        meniscus_col = cc["meniscus_col"]
        off = (state == "HS0")

        frac = max(0.0, min(1.0, level / 100.0))
        water_y = self.maxy - frac * (self.maxy - self.miny)
        amp = (8 if state == "HS3" else 5) * ss if not off else 0
        surf = self._surface_points(water_y, phase, amp)

        # Quantize phase to reduce liquid mask recomputation
        lm_key = (state, round(water_y, 1), round(phase * 10) / 10)
        if lm_key == self._cached_lm_key and self._cached_liquid_mask is not None:
            liquid_mask = self._cached_liquid_mask
        else:
            below = self._scratch_l
            below.paste(0, (0, 0, s, s))  # clear reusable buffer
            ImageDraw.Draw(below).polygon(
                surf + [(self.maxx + 48 * ss, self.maxy + 80 * ss),
                        (self.minx - 48 * ss, self.maxy + 80 * ss)], fill=255)
            liquid_mask = ImageChops.multiply(below, self.mask)
            self._cached_lm_key = lm_key
            self._cached_liquid_mask = liquid_mask
        spr.paste(cc["liq"], (0, 0), liquid_mask)

        if not off and bubbles and level > 8:
            bl = self._scratch_rgba
            bl.paste((0, 0, 0, 0), (0, 0, s, s))  # clear reusable buffer
            bd = ImageDraw.Draw(bl)
            for b in bubbles:
                by = water_y + (self.maxy - water_y) * b["y"]
                bx = self.ox + (0.18 + b["x"] * 0.5) * self.bw
                r = b["r"] * ss
                a = int(150 * min(1.0, b["y"] * 3))
                bd.ellipse([bx - r, by - r, bx + r, by + r],
                           fill=bub_col + (a,))
            spr.alpha_composite(self._clip(bl, liquid_mask))

        if not off and 1 < level < 99:
            ml = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            ImageDraw.Draw(ml).line(
                surf, fill=meniscus_col,
                width=3 * ss, joint="curve")
            spr.alpha_composite(self._clip(ml, self.mask))

        self._draw_face(spr, state, blink, ink, off)
        self._draw_particles(spr, particles, ink)
        return spr

    def _draw_face(self, spr, state, blink, ink, off):
        ss = self.ss
        d = ImageDraw.Draw(spr)
        fcx, fcy = self.fcx, self.fcy
        edx, er = self.edx, self.er
        eyl, eyr = self.eyl, self.eyr

        def arc(cx, cy, rw, rh, a0, a1, wd):
            d.arc([cx - rw, cy - rh, cx + rw, cy + rh], a0, a1, fill=ink, width=int(wd))

        if not off:
            blush = self.cache[state]["blush"]
            if blush is not None:
                spr.alpha_composite(blush)
            d = ImageDraw.Draw(spr)

        def round_eye(cx, cy):
            eh = er * (0.12 + 0.88 * (1 - blink))
            d.ellipse([cx - er, cy - eh, cx + er, cy + eh], fill=ink)
            if blink < 0.5:
                hr = er * 0.34
                d.ellipse([cx - er * 0.3 - hr, cy - eh * 0.4 - hr,
                           cx - er * 0.3 + hr, cy - eh * 0.4 + hr],
                          fill=(255, 255, 255, 235))

        if off:
            for (cx, cy) in (eyl, eyr):
                arc(cx, cy + er * 0.3, er, er * 0.8, 200, 340, 4 * ss)
            arc(fcx, fcy + 0.14 * self.bh, er * 0.7, er * 0.5, 200, 340, 4 * ss)
            for j in range(3):
                d.text((self.maxx - 36 * ss + j * 16 * ss,
                        self.miny + 8 * ss - j * 20 * ss),
                       "z", fill=(150, 156, 162), font=self.font)
        elif state == "HS1":
            if blink < 0.5:
                for (cx, cy) in (eyl, eyr):
                    arc(cx, cy, er * 1.1, er * 1.2, 200, 340, 5 * ss)
            else:
                for (cx, cy) in (eyl, eyr):
                    d.line([cx - er, cy, cx + er, cy], fill=ink, width=int(5 * ss))
            arc(fcx, fcy + 0.10 * self.bh, er * 1.3, er * 1.1, 20, 160, 5 * ss)
        elif state == "HS2":
            for (cx, cy) in (eyl, eyr):
                round_eye(cx, cy)
            arc(fcx, fcy + 0.13 * self.bh, er * 0.8, er * 0.5, 20, 160, 4 * ss)
            sx, sy = fcx + edx + er * 1.6, fcy - er
            d.ellipse([sx - er * 0.4, sy - er * 0.6, sx + er * 0.4, sy + er * 0.7],
                      fill=(130, 200, 230, 230))
        else:  # HS3
            for (cx, cy) in (eyl, eyr):
                d.line([cx - er, cy - er, cx + er, cy + er], fill=ink, width=int(4 * ss))
                d.line([cx - er, cy + er, cx + er, cy - er], fill=ink, width=int(4 * ss))
            d.ellipse([fcx - er * 0.9, fcy + 0.09 * self.bh,
                       fcx + er * 0.9, fcy + 0.18 * self.bh], outline=ink, width=int(4 * ss))
            tx, ty = fcx - edx - er * 0.4, fcy + er * 0.8
            d.ellipse([tx - er * 0.4, ty - er * 0.5, tx + er * 0.4, ty + er * 0.8],
                      fill=(120, 195, 230, 230))

    def _draw_particles(self, spr, particles, ink):
        ss = self.ss
        d = ImageDraw.Draw(spr)
        for p in particles:
            if p["kind"] == "pellet":
                x, y, r = p["x"] * ss, p["y"] * ss, p["r"] * ss
                col = hx(p["col"])
                d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))
                d.ellipse([x - r * 0.4, y - r * 0.5, x, y], fill=(255, 255, 255, 120))
            elif p["kind"] == "ring":
                x, y, r = p["x"] * ss, p["y"] * ss, p["r"] * ss
                a = int(180 * p["life"])
                d.ellipse([x - r, y - r * 0.5, x + r, y + r * 0.5],
                          outline=(255, 255, 255, a), width=int(2 * ss))
            elif p["kind"] == "text":
                a = int(255 * min(1.0, p["life"]))
                d.text((p["x"] * ss, p["y"] * ss), p["txt"], font=self.font,
                       fill=ink + (a,), anchor="mm")
            elif p["kind"] == "spark":
                a = int(220 * p["life"])
                d.text((p["x"] * ss, p["y"] * ss), "✦", font=self.font,
                       fill=(255, 210, 122, a), anchor="mm")
            elif p["kind"] == "heart":
                a = int(230 * p["life"])
                d.text((p["x"] * ss, p["y"] * ss), "♡", font=self.font,
                       fill=(255, 150, 170, a), anchor="mm")

    def frame(self, sprite_pil, dx, dy, scale=1.0):
        s = self.w
        f = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        f.paste(self.shadow, (0, 0), self.shadow)
        if abs(scale - 1.0) > 0.001:
            nw = max(1, int(sprite_pil.width * scale))
            nh = max(1, int(sprite_pil.height * scale))
            sprite_pil = sprite_pil.resize((nw, nh), self._bilinear)
            px = int((s - nw) / 2 + dx * self.ss)
            py = int((s - nh) / 2 + dy * self.ss)
        else:
            px = int(dx * self.ss)
            py = int(dy * self.ss)
        f.paste(sprite_pil, (px, py), sprite_pil)
        # At SS=1 the output size equals internal size; skip downscale
        if self.W == s:
            return f
        return f.resize((self.W, self.W), self._bilinear)


# ══════════════════════════════════════════════════════════════════════════════
# Backends (unchanged contract: snapshot / cmd_rpc / cmd_meal / start / stop)
# ══════════════════════════════════════════════════════════════════════════════
class SimBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self.level = 100.0
        self.enabled = True
        self.last = time.time()
        self._last_feed = 0.0

    def _drain(self):
        now = time.time()
        rate = 100.0 / (DRAIN_HOURS * 3600.0)
        self.level = max(0.0, min(100.0, self.level - (now - self.last) * rate))
        self.last = now

    def _state(self):
        if not self.enabled:
            return "HS0"
        if self.level >= HUNGRY_THRESHOLD:
            return "HS1"
        if self.level >= STARVING_THRESHOLD:
            return "HS2"
        return "HS3"

    def snapshot(self):
        with self._lock:
            self._drain()
            return {"connected": True, "enabled": self.enabled,
                    "level": self.level, "state": self._state(),
                    "busy": False, "backend": "SIM"}

    def cmd_rpc(self, words):
        with self._lock:
            self._drain()
            c = words[0]
            if c == "hunger_mode":
                self.enabled = (len(words) > 1 and words[1] == "on")
                self.level = 100.0
            elif c == "hunger":
                arg = words[1] if len(words) > 1 else ""
                if arg == "hs0":
                    self.enabled = False; self.level = 100.0
                elif arg == "hs1":
                    self.enabled = True; self.level = 100.0
                elif arg == "hs2":
                    self.enabled = True; self.level = 59.0
                elif arg == "hs3":
                    self.enabled = True; self.level = 24.0
            elif c == "quit":
                self.enabled = False
            return True

    def cmd_meal(self, payload):
        with self._lock:
            self._drain()
            now = time.time()
            if not self.enabled or now - self._last_feed < QR_COOLDOWN_SEC:
                return False
            self._last_feed = now
            self.level = min(100.0, self.level + MEALS.get(payload, 0.0))
            return True

    def start(self): pass
    def stop(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════
class StomachApp:
    def __init__(self, root, backend, server_name):
        self.root = root
        self.backend = backend
        self.server_name = server_name
        self.view = VIEW
        self.renderer = StomachRenderer(view=self.view, ss=RENDER_SS) if HAVE_PIL else None

        self.state = "HS0"
        self.prev_state = "HS0"
        self.trans = 1.0
        self.enabled = False
        self.connected = False
        self._last_connected = None
        self._last_error_seen = ""
        self.busy = False
        self.disp_level = 100.0
        self.target_level = 100.0
        self.last_level = 100.0
        self.phase = 0.0
        self.blink = 0.0
        self._next_blink = time.monotonic() + random.uniform(2, 5)
        self.cooldown_until = 0.0
        self.bubbles = [self._new_bubble() for _ in range(7)]
        self.particles = []
        self.events = deque(maxlen=6)
        self.decor_items = []
        self._photo = None
        self._last_frame_time = time.perf_counter()
        self._last_bg = None
        self._last_badge = None

        root.title("iCub · Stomach Monitor")
        root.configure(bg=BG)
        root.minsize(880, 680)
        root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.bind("<F11>", lambda _e: root.attributes(
            "-fullscreen", not bool(root.attributes("-fullscreen"))))

        self.f_title = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self.f_big = tkfont.Font(family="Helvetica", size=26, weight="bold")
        self.f_lbl = tkfont.Font(family="Helvetica", size=8, weight="bold")
        self.f_small = tkfont.Font(family="Helvetica", size=8)
        self.f_btn = tkfont.Font(family="Helvetica", size=9, weight="bold")

        self.wrap = tk.Frame(root, bg=BG)
        self.wrap.pack(fill="both", expand=True, padx=10, pady=10)

        self.panel = tk.Frame(self.wrap, bg=PANEL_BG, height=170,
                              highlightthickness=1, highlightbackground=BORDER)
        self.panel.pack(side="bottom", fill="x")
        self.panel.pack_propagate(False)

        self.canvas = tk.Canvas(self.wrap, width=self.view, height=self.view, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_item = self.canvas.create_image(self.view // 2, self.view // 2, anchor="center")
        self.badge_bg = self.canvas.create_rectangle(0, 0, 1, 1, outline="", fill="")
        self.badge_item = self.canvas.create_text(0, 0, text="", font=self.f_title, fill=INK)
        self.title_item = self.canvas.create_text(
            self.view // 2, self.view // 2, text="", font=self.f_title, fill=INK)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._create_decor()
        self._build_panel()

        if not HAVE_PIL:
            self._vector_setup()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.backend.start()
        self._add_event("Monitor started")
        self.root.after(180, self._poll_backend)
        self.root.after(int(1000 / FPS), self._tick)

    # ── panel ───────────────────────────────────────────────────────────────────
    def _card(self, title, expand=False):
        outer = tk.Frame(self.panel, bg=PANEL_BG)
        outer.pack(side="left", fill="both", expand=expand, padx=(0, 2))
        inner = tk.Frame(outer, bg=PANEL_BG)
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        if title:
            tk.Label(inner, text=title.upper(), font=self.f_lbl, bg=PANEL_BG,
                     fg=MUTED).pack(anchor="w", pady=(0, 6))
        return inner

    def _mkbtn(self, parent, text, cmd):
        btn = tk.Button(parent, text=text, font=self.f_btn, command=lambda: None,
                        relief="flat", bd=0, cursor="hand2", padx=14, pady=10,
                        bg=SUBTLE, fg=INK, activebackground="#E9EEF3",
                        highlightthickness=0)
        self._add_hover(btn, SUBTLE, "#E9EEF3")
        def _press(_event, b=btn):
            if str(b.cget("state")) != "disabled":
                b.config(relief="sunken")

        def _release(_event, b=btn, fn=cmd):
            if str(b.cget("state")) != "disabled":
                b.config(relief="flat")
                fn()
            return "break"

        btn.bind("<ButtonPress-1>", _press)
        btn.bind("<ButtonRelease-1>", _release)
        return btn

    def _add_hover(self, btn, normal, hover):
        btn._normal_bg = normal
        btn._hover_bg = hover
        btn.bind("<Enter>", lambda _e, b=btn: b.config(bg=getattr(b, "_hover_bg", hover))
                 if str(b.cget("state")) != "disabled" else None)
        btn.bind("<Leave>", lambda _e, b=btn: b.config(bg=getattr(b, "_normal_bg", normal))
                 if str(b.cget("state")) != "disabled" else None)

    def _set_button_colors(self, btn, bg, fg=INK, hover=None, state="normal"):
        btn._normal_bg = bg
        btn._hover_bg = hover or bg
        btn.config(bg=bg, fg=fg, activebackground=btn._hover_bg, state=state)

    def _separator(self):
        sep = tk.Frame(self.panel, bg=BORDER, width=1)
        sep.pack(side="left", fill="y", pady=18, padx=8)

    def _add_event(self, text):
        self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")
        self._render_events()

    def _render_events(self):
        if hasattr(self, "lbl_events"):
            self.lbl_events.config(text="\n".join(self.events))

    def _on_canvas_resize(self, event):
        usable = max(320, min(event.width - 40, event.height - 72))
        target = int(usable)
        self._position_canvas_items(event.width, event.height)
        # Debounce: delay expensive renderer rebuild until resizing stops
        self._pending_view = target
        if hasattr(self, '_resize_timer') and self._resize_timer is not None:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(150, self._apply_resize)

    def _apply_resize(self):
        self._resize_timer = None
        target = getattr(self, '_pending_view', self.view)
        self._set_view_size(target)

    def _set_view_size(self, new_view):
        new_view = max(320, min(MAX_RENDER_VIEW, new_view))
        if abs(new_view - self.view) < 24:
            return
        old_view = self.view
        self.view = new_view
        if HAVE_PIL:
            self.renderer = StomachRenderer(view=self.view, ss=RENDER_SS)
        scale = self.view / max(1, old_view)
        for p in self.particles:
            if "x" in p:
                p["x"] *= scale
            if "y" in p:
                p["y"] *= scale
            if "r" in p:
                p["r"] *= scale

    def _position_canvas_items(self, width=None, height=None):
        width = width or self.canvas.winfo_width()
        height = height or self.canvas.winfo_height()
        cx, cy = width // 2, height // 2
        self.canvas.coords(self.img_item, cx, cy)
        self.canvas.coords(self.title_item, cx, max(36, cy - self.view // 2 + 24))
        badge_w, badge_h = 260, 34
        self.canvas.coords(
            self.badge_bg,
            cx - badge_w // 2, cy + self.view // 2 - 38,
            cx + badge_w // 2, cy + self.view // 2 - 4)
        self.canvas.coords(self.badge_item, cx, cy + self.view // 2 - 21)

    def _create_decor(self):
        for _ in range(10):
            item = self.canvas.create_text(
                random.randint(40, 900),
                random.randint(40, 520),
                text=random.choice(DECOR_SYMBOLS),
                font=tkfont.Font(family="Helvetica", size=random.randint(9, 16), weight="bold"),
                fill=random.choice(DECOR_COLORS),
            )
            self.canvas.tag_lower(item, self.img_item)
            self.decor_items.append({
                "id": item,
                "x": random.uniform(0.04, 0.96),
                "y": random.uniform(0.04, 0.88),
                "speed": random.uniform(0.15, 0.55),
                "phase": random.uniform(0, math.tau),
            })

    def _animate_decor(self, dt):
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        hidden = self.state == "HS3"
        for d in self.decor_items:
            d["phase"] += d["speed"] * dt
            x = d["x"] * w + math.sin(d["phase"]) * 18
            y = d["y"] * h + math.cos(d["phase"] * 0.7) * 10
            self.canvas.coords(d["id"], x, y)
            self.canvas.itemconfig(d["id"], state=("hidden" if hidden else "normal"))

    def _build_panel(self):
        s = self._card("STATUS")
        self.lbl_state = tk.Label(s, text="—", font=self.f_title, bg=PANEL_BG, fg=INK)
        self.lbl_state.pack(anchor="w")
        self.lbl_level = tk.Label(s, text="—", font=self.f_big, bg=PANEL_BG, fg=INK)
        self.lbl_level.pack(anchor="w")
        self.level_bar = tk.Canvas(s, width=150, height=6, bg=PANEL_BG, highlightthickness=0)
        self.level_bar.pack(anchor="w", pady=(0, 7))
        self.level_track = self.level_bar.create_rectangle(0, 0, 150, 6, outline="", fill="#E7EBEF")
        self.level_fill = self.level_bar.create_rectangle(0, 0, 1, 6, outline="", fill=ACCENT["HS0"])
        self.lbl_busy = tk.Label(s, text="", font=self.f_small, bg=PANEL_BG, fg=MUTED)
        self.lbl_busy.pack(anchor="w")
        self.lbl_conn = tk.Label(s, text="", font=self.f_small, bg=PANEL_BG, fg=MUTED)
        self.lbl_conn.pack(anchor="w")

        self._separator()
        d = self._card("Drive")
        rr = tk.Frame(d, bg=PANEL_BG); rr.pack(fill="x")
        self.btn_on = self._mkbtn(rr, "On",
                                  lambda: self._rpc(["hunger_mode", "on"], self.btn_on))
        self.btn_off = self._mkbtn(rr, "Off",
                                   lambda: self._rpc(["hunger_mode", "off"], self.btn_off))
        self.btn_on.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_off.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self._separator()
        st = self._card("State")
        g = tk.Frame(st, bg=PANEL_BG); g.pack(fill="x")
        self.btn_hs = {}
        for col, (key, txt) in enumerate(
                [("HS0", "Off"), ("HS1", "Full"),
                 ("HS2", "Hungry"), ("HS3", "Starving")]):
            b = self._mkbtn(g, txt, lambda k=key: self._rpc(["hunger", k.lower()], self.btn_hs[k]))
            b.grid(row=0, column=col, sticky="ew", padx=2)
            g.columnconfigure(col, weight=1)
            self.btn_hs[key] = b

        self._separator()
        rs = self._card("Reset")
        self.btn_reset = self._mkbtn(rs, "Refill tummy",
                                     lambda: self._rpc(["hunger_mode", "on"], self.btn_reset))
        self.btn_reset.pack(fill="x")

        self._separator()
        fd = self._card("Feed")
        fgr = tk.Frame(fd, bg=PANEL_BG); fgr.pack(fill="x")
        self.meal_btns = {}
        for col, (key, txt) in enumerate(
                [("SMALL_MEAL", "Mini snack  +10"), ("MEDIUM_MEAL", "Meal  +25"),
                 ("LARGE_MEAL", "Feast  +45")]):
            b = self._mkbtn(fgr, txt, lambda k=key: self._feed(k))
            b.grid(row=0, column=col, sticky="ew", padx=2)
            fgr.columnconfigure(col, weight=1)
            self.meal_btns[key] = b

        self._separator()
        ev = self._card("Events", expand=True)
        self.lbl_events = tk.Label(ev, text="", font=self.f_small, bg=PANEL_BG, fg=INK,
                                   justify="left", anchor="nw")
        self.lbl_events.pack(fill="both", expand=True, anchor="w")

        self._separator()
        end = self._card("")
        self.btn_shutdown = self._mkbtn(end, "Shutdown", self._shutdown)
        self.btn_shutdown.config(fg="#c0392b")
        self.btn_shutdown.pack(fill="x")

    # ── command helpers ──────────────────────────────────────────────────────────
    def _flash(self, btn):
        if btn:
            btn.config(bg="#cfe9dc")
            self.root.after(140, self._refresh_buttons)

    def _rpc(self, words, btn=None):
        self.backend.cmd_rpc(words)
        if words:
            label = " ".join(str(w) for w in words)
            self._add_event(f"Command sent: {label}")
        self._flash(btn)

    def _feed(self, payload):
        now = time.monotonic()
        if not self.enabled or now < self.cooldown_until:
            self._add_event("Feed ignored: drive unavailable" if not self.enabled else "Feed ignored: cooldown")
            return
        if self.backend.cmd_meal(payload) is False:
            self._add_event("Feed rejected by backend")
            return
        self.cooldown_until = now + QR_COOLDOWN_SEC
        self._spawn_pellet(payload)
        self._add_event(f"{payload.replace('_', ' ').title()} received +{int(MEALS[payload])}")
        self._refresh_buttons()

    def _shutdown(self):
        if messagebox.askyesno("Shutdown module",
                               "Send 'quit' to the controller?\nThis stops the "
                               "running executiveControl module on the robot."):
            self.backend.cmd_rpc(["quit"])

    # ── particles / bubbles ───────────────────────────────────────────────────────
    def _new_bubble(self):
        return {"x": random.uniform(0, 1), "y": random.uniform(0.05, 0.95),
                "r": random.uniform(2.2, 5.0), "sp": random.uniform(0.004, 0.011)}

    def _spawn_pellet(self, payload):
        col = {"SMALL_MEAL": "#9fcf7a", "MEDIUM_MEAL": "#e8a06a",
               "LARGE_MEAL": "#e87a5c"}.get(payload, "#cccccc")
        delta = MEALS.get(payload, 0)
        count = {"SMALL_MEAL": 4, "MEDIUM_MEAL": 7, "LARGE_MEAL": 11}.get(payload, 5)
        for _ in range(count):
            self.particles.append({
                "kind": "pellet",
                "x": self.view * random.uniform(0.44, 0.56),
                "y": self.view * random.uniform(0.02, 0.09),
                "vy": random.uniform(-130.0, 45.0),
                "vx": random.uniform(-90.0, 90.0),
                "r": random.uniform(3.5, 7.5),
                "col": col,
            })
        self.particles.append({"kind": "text", "x": self.view * 0.34, "y": self.view * 0.30,
                               "txt": f"+{int(delta)}", "life": 1.0})

    # ── backend poll (slow) ───────────────────────────────────────────────────────
    def _poll_backend(self):
        snap = self.backend.snapshot()
        old_connected = self.connected
        self.connected = snap["connected"]
        self.enabled = snap["enabled"]
        self.busy = snap.get("busy", False)
        err = str(snap.get("last_error") or "")
        if err and err != self._last_error_seen:
            self._last_error_seen = err
            self._add_event(err)
        if self._last_connected is None:
            self._last_connected = self.connected
            if self.connected:
                self._add_event("Controller connected")
        elif self.connected != old_connected:
            self._last_connected = self.connected
            self._add_event("Controller connected" if self.connected else "Controller disconnected")
        new_state = snap["state"]
        if new_state != self.state:
            self.prev_state = self.state
            self.state = new_state
            self.trans = 0.0
            self._add_event(f"State changed to {STATE_TEXT.get(new_state, new_state)}")
        self.target_level = snap["level"]
        if self.enabled and self.target_level - self.last_level > 1.5:
            delta = self.target_level - self.last_level
            self.particles.append({"kind": "pellet", "x": self.view * 0.5, "y": self.view * 0.06,
                                   "vy": 0.0, "r": 8, "col": "#dddddd"})
            if time.monotonic() > self.cooldown_until:
                self._add_event(f"Meal received +{int(round(delta))}")
        self.last_level = self.target_level
        self._refresh_labels()
        self._refresh_buttons()
        self.root.after(180, self._poll_backend)

    def _refresh_labels(self):
        acc = ACCENT.get(self.state, MUTED)
        state_label = STATE_TEXT.get(self.state, "—")
        if self.state == "HS2":
            state_label = f"Warning: {state_label}"
        elif self.state == "HS3":
            state_label = f"Critical: {state_label}"
        elif self.state == "HS0":
            state_label = "Drive unavailable"
        self.lbl_state.config(text=state_label, fg=acc)
        self.lbl_level.config(text=("OFF" if not self.enabled else f"{self.disp_level:0.0f}%"),
                              fg=(MUTED if not self.enabled else acc))
        fill_w = 0 if not self.enabled else max(0, min(150, int(150 * self.disp_level / 100.0)))
        self.level_bar.coords(self.level_fill, 0, 0, fill_w, 6)
        self.level_bar.itemconfig(self.level_fill, fill=(MUTED if not self.enabled else acc))
        self.lbl_busy.config(text=("● interacting…" if self.busy else "○ idle"))
        if self.connected:
            self.lbl_conn.config(text=f"● {self.server_name}", fg="#2e9e6e")
        else:
            self.lbl_conn.config(text=f"○ connecting… {self.server_name}", fg="#c0392b")
        self.canvas.itemconfig(self.title_item, text=STATE_TEXT.get(self.state, ""), fill=acc)
        self._update_canvas_chrome()

    def _refresh_buttons(self):
        on = self.enabled
        self._set_button_colors(
            self.btn_on, "#2C8A60" if on else SUBTLE,
            "#fff" if on else INK, "#3E9E73" if on else "#E9EEF3")
        self._set_button_colors(
            self.btn_off, "#6E767D" if not on else SUBTLE,
            "#fff" if not on else INK, "#838B92" if not on else "#E9EEF3")
        for k, b in self.btn_hs.items():
            cur = (k == self.state)
            pal = PALETTE.get(k, PALETTE["HS0"])
            self._set_button_colors(
                b, pal[3] if cur else SUBTLE,
                "#fff" if cur else INK,
                pal[1] if cur else "#E9EEF3")
        remain = self.cooldown_until - time.monotonic()
        for k, b in self.meal_btns.items():
            label = {
                "SMALL_MEAL": "Mini snack",
                "MEDIUM_MEAL": "Meal",
                "LARGE_MEAL": "Feast",
            }[k]
            delta = int(MEALS[k])
            if not self.enabled:
                self._set_button_colors(b, "#f0f1f3", "#c2c7cc", "#f0f1f3", state="disabled")
                b.config(text=f"{label}  +{delta}")
            elif remain > 0:
                self._set_button_colors(b, "#f0f1f3", "#b6bbc1", "#f0f1f3", state="disabled")
                b.config(text=f"{label}  {remain:0.1f}s")
            else:
                self._set_button_colors(b, SUBTLE, INK, "#E9EEF3")
                b.config(text=f"{label}  +{delta}")
        if remain > 0:
            self.root.after(100, self._refresh_buttons)

    def _update_canvas_chrome(self):
        acc = ACCENT.get(self.state, MUTED)
        bg = {"HS0": "#ECEFF1", "HS1": "#F3F8F5", "HS2": "#FFF7E8", "HS3": "#2F2526"}.get(self.state, BG)
        if bg != self._last_bg:
            self.root.configure(bg=bg)
            self.wrap.configure(bg=bg)
            self.canvas.configure(bg=bg)
            self._last_bg = bg
        width = self.canvas.winfo_width() or self.view
        height = self.canvas.winfo_height() or self.view
        badge = ""
        if self.state == "HS0":
            badge = "Drive unavailable"
        elif self.state == "HS2":
            badge = "Hunger rising"
        elif self.state == "HS3":
            badge = "Critical"
        if badge != self._last_badge:
            self.canvas.itemconfig(self.badge_item, text=badge, fill=("#fff" if self.state == "HS3" else acc))
            badge_state = "normal" if badge else "hidden"
            self.canvas.itemconfig(self.badge_item, state=badge_state)
            self.canvas.itemconfig(self.badge_bg, state=badge_state,
                                   fill=("#5B2020" if self.state == "HS3" else "#FFFFFF"))
            self._last_badge = badge
        self._position_canvas_items()

    # ── animation tick (fast) ─────────────────────────────────────────────────────
    _PHASE_SPEED = {"HS0": 1.6, "HS1": 2.4, "HS2": 4.2, "HS3": 6.4}

    def _tick(self):
        start = time.perf_counter()
        now = time.perf_counter()
        dt = min(1.0 / FPS, now - self._last_frame_time)
        self._last_frame_time = now
        try:
            self._step(dt)
            if HAVE_PIL:
                self._render_pil()
            else:
                self._render_vector()
        except Exception:
            pass
        finally:
            elapsed = time.perf_counter() - start
            target = 1.0 / max(1, FPS)
            delay = max(1, int((target - elapsed) * 1000))
            self.root.after(delay, self._tick)

    @staticmethod
    def _damp(current, target, speed, dt):
        return current + (target - current) * (1 - math.exp(-speed * dt))

    def _step(self, dt):
        self.disp_level = self._damp(self.disp_level, self.target_level, 9.0, dt)
        self.phase += self._PHASE_SPEED.get(self.state, 3.0) * dt
        if self.trans < 1.0:
            self.trans = min(1.0, self.trans + 3.2 * dt)
        self._animate_decor(dt)

        now = time.monotonic()
        if now >= self._next_blink:
            self.blink = 1.0
            if now >= self._next_blink + 0.12:
                self.blink = 0.0
                self._next_blink = now + random.uniform(2.5, 6.0)
        else:
            self.blink = 0.0

        for i, b in enumerate(self.bubbles):
            b["y"] -= b["sp"]
            if b["y"] <= 0.02:
                self.bubbles[i] = self._new_bubble()

        if self.enabled:
            # Throttle label updates — only when displayed text changes
            new_text = f"{self.disp_level:0.0f}%"
            if getattr(self, '_last_level_text', '') != new_text:
                self._last_level_text = new_text
                self.lbl_level.config(text=new_text)

        if self.renderer:
            wy_disp = self.renderer.water_y_disp(self.disp_level)
        else:
            frac = max(0.0, min(1.0, self.disp_level / 100.0))
            wy_disp = self.view * (0.86 - 0.6 * frac)

        alive = []
        for p in self.particles:
            if p["kind"] == "pellet":
                p["vy"] += 900.0 * dt
                p["y"] += p["vy"] * dt
                p["x"] += p.get("vx", 0.0) * dt
                p["vx"] = p.get("vx", 0.0) * math.exp(-3.2 * dt)
                if p["y"] >= wy_disp:
                    self.particles.append({"kind": "ring", "x": p["x"], "y": wy_disp,
                                           "r": 5, "life": 1.0})
                    continue
                alive.append(p)
            elif p["kind"] == "ring":
                p["r"] += 72.0 * dt; p["life"] -= 2.8 * dt
                if p["life"] > 0:
                    alive.append(p)
            elif p["kind"] == "text":
                p["y"] -= 42.0 * dt; p["life"] -= 0.72 * dt
                if p["life"] > 0:
                    alive.append(p)
            elif p["kind"] in ("spark", "heart"):
                p["y"] -= 18.0 * dt
                p["life"] -= 1.2 * dt
                if p["life"] > 0:
                    alive.append(p)
        self.particles = alive
        # Scale spawn probability by dt so it's FPS-independent
        # (Original was tuned for 12 FPS, so multiply by dt * 12)
        fps_scale = dt * 12.0
        if self.state == "HS1" and self.trans >= 1.0 and random.random() < 0.06 * fps_scale:
            self.particles.append({"kind": random.choice(["spark", "heart"]), "life": 1.0,
                                   "x": self.view * random.uniform(0.18, 0.52),
                                   "y": self.view * random.uniform(0.16, 0.36)})
        if self.state == "HS2" and random.random() < 0.015 * fps_scale:
            self.particles.append({"kind": "text", "life": 0.9,
                                   "x": self.view * random.uniform(0.50, 0.72),
                                   "y": self.view * random.uniform(0.34, 0.48),
                                   "txt": "grr"})

    def _bob_shake(self):
        t = self.phase
        if self.state == "HS0":
            return 0.0, 0.0
        if self.state == "HS3":
            dx = math.sin(t * 5.7) * 5.5 + math.sin(t * 13.1) * 2.0
            dy = math.sin(t * 4.2) * 2.8
            return dx, dy
        if self.state == "HS2":
            dx = math.sin(t * 2.6) * 2.4
            dy = math.sin(t * 1.8) * 1.8
            return dx, dy
        dx = math.sin(t * 0.8) * 0.8
        dy = math.sin(t * 1.2) * 2.4
        return dx, dy

    def _render_pil(self):
        r = self.renderer
        bub = self.bubbles if self.enabled else []
        if self.trans >= 1.0:
            spr = r.compose(self.state, self.disp_level, self.phase, self.blink,
                            bub, self.particles)
        else:
            # Optimized transition: compose ONCE with the target state, then
            # alpha-blend a lightweight old-state base on top. This is ~40%
            # cheaper than rendering two full sprites.
            t = ease_out_cubic(self.trans)
            spr = r.compose(self.state, self.disp_level, self.phase, self.blink,
                            bub, self.particles)
            if t < 0.95:  # skip blend when nearly done
                old = r.compose(self.prev_state, self.disp_level, self.phase, self.blink,
                                [], [])  # old state without particles/bubbles = cheaper
                spr = Image.blend(old, spr, t)
        dx, dy = self._bob_shake()
        scale = 1.0
        if self.state == "HS1":
            scale = 1.0 + 0.010 * math.sin(self.phase * 1.1)
        elif self.state == "HS2":
            scale = 1.0 + 0.016 * math.sin(self.phase * 1.9)
        elif self.state == "HS3":
            scale = 1.0 + 0.022 * math.sin(self.phase * 3.3) + 0.006 * math.sin(self.phase * 8.0)
        frame = r.frame(spr, dx, dy, scale)
        self._photo = ImageTk.PhotoImage(frame)
        self.canvas.itemconfig(self.img_item, image=self._photo)

    # ── vector fallback (no Pillow) ──────────────────────────────────────────────
    def _vector_setup(self):
        self._poly_pts = []
        for (x, y) in catmull_rom_closed(SAC, 14):
            self._poly_pts += [self.view / 2 + (x - 0.5) * self.view * 0.7,
                               24 + y * self.view * 0.74]
        self._fill = self.canvas.create_polygon(self._poly_pts, fill="#cccccc",
                                                outline="", smooth=True)
        self._out = self.canvas.create_polygon(self._poly_pts, fill="", outline="#888",
                                               width=4, smooth=True)

    def _render_vector(self):
        pal = PALETTE.get(self.state, PALETTE["HS0"])
        self.canvas.itemconfig(self._fill, fill=pal[0])
        self.canvas.itemconfig(self._out, outline=pal[3])

    def _on_close(self):
        try:
            self.backend.stop()
        except Exception:
            pass
        self.root.destroy()


_RFModuleBase = yarp.RFModule if HAVE_YARP else object


class StomachMonitorModule(_RFModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.module_name = "alwayson_stomachMonitor"
        self.server = "/executiveControl"
        self.qr_target = "/alwayson/executiveControl/qr:i"
        self.period = 1.0 / FPS
        self.poll_period = 0.25
        self.sim = False
        self._running = True
        self._closed = False

        self._rpc_port: Optional[yarp.Port] = None
        self._controller_rpc: Optional[yarp.RpcClient] = None
        self._qr_port: Optional[yarp.BufferedPortBottle] = None
        self._cmds: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._snap: Dict[str, Any] = {
            "connected": False,
            "enabled": False,
            "level": 100.0,
            "state": "HS0",
            "busy": False,
            "backend": "YARP",
            "last_error": "",
        }
        self._next_poll = 0.0
        self._sim_backend: Optional[SimBackend] = None
        self._root: Optional[tk.Tk] = None
        self._app: Optional[StomachApp] = None

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if not HAVE_YARP:
                print("[ERROR] YARP Python bindings unavailable; use --sim for offline preview.")
                return False
            if rf.check("name"):
                self.module_name = rf.find("name").asString().lstrip("/")
            self.setName(self.module_name)

            if rf.check("server"):
                self.server = rf.find("server").asString()
            if rf.check("qr"):
                self.qr_target = rf.find("qr").asString()
            else:
                name = self.server.strip("/").split("/")[-1] or "executiveControl"
                self.qr_target = f"/alwayson/{name}/qr:i"
            if rf.check("period"):
                self.poll_period = max(0.05, rf.find("period").asFloat64())
            self.sim = rf.check("sim")

            self._rpc_port = yarp.Port()
            if not self._rpc_port.open(f"/{self.module_name}/rpc"):
                print("[ERROR] Cannot open stomachMonitor RPC port")
                return False
            self.attach(self._rpc_port)

            if self.sim:
                self._sim_backend = SimBackend()
                label = "SIM"
            else:
                base = f"/alwayson/{self.module_name}"
                self._controller_rpc = yarp.RpcClient()
                self._qr_port = yarp.BufferedPortBottle()
                if not self._controller_rpc.open(f"{base}/rpc:o"):
                    print("[ERROR] Cannot open stomachMonitor controller RPC client")
                    return False
                if not self._qr_port.open(f"{base}/qr:o"):
                    print("[ERROR] Cannot open stomachMonitor QR output port")
                    return False
                label = self.server

            if not HAVE_PIL:
                print("[WARN] Pillow not installed — using a basic vector view. "
                      "For the full visuals run:  pip install pillow")

            self._root = tk.Tk()
            self._app = StomachApp(self._root, self, label)
            print(f"[INFO] StomachMonitorModule ready; RPC /{self.module_name}/rpc")
            return True
        except Exception as e:
            print(f"[ERROR] stomachMonitor configure failed: {e}")
            traceback.print_exc()
            return False

    # backend contract used by StomachApp
    def snapshot(self):
        if self._sim_backend is not None:
            return self._sim_backend.snapshot()
        with self._lock:
            return dict(self._snap)

    def cmd_rpc(self, words):
        if self._sim_backend is not None:
            return self._sim_backend.cmd_rpc(words)
        self._ensure_links()
        return self._send_rpc(words)

    def cmd_meal(self, payload):
        if self._sim_backend is not None:
            return self._sim_backend.cmd_meal(payload)
        self._ensure_links()
        return self._send_meal(payload)

    def start(self):
        pass

    def stop(self):
        self._running = False
        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        if self._sim_backend is None:
            self._ensure_links()
            self._drain_cmds()
            now = time.monotonic()
            if now >= self._next_poll:
                self._poll_status()
                self._next_poll = now + self.poll_period
        self._service_tk()
        if self._sim_backend is None:
            self._drain_cmds()
        return self._running

    def run_tk_mainloop(self) -> bool:
        if self._root is None:
            return False
        self._root.after(20, self._tk_backend_tick)
        try:
            self._root.mainloop()
        finally:
            self._running = False
        return True

    def _tk_backend_tick(self) -> None:
        if not self._running or self._root is None:
            return
        try:
            if self._sim_backend is None:
                self._ensure_links()
                self._drain_cmds()
                now = time.monotonic()
                if now >= self._next_poll:
                    self._poll_status()
                    self._next_poll = now + self.poll_period
        finally:
            if self._running and self._root is not None:
                self._root.after(20, self._tk_backend_tick)

    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        reply.clear()
        try:
            if cmd.size() < 1:
                return self._rpc_error(reply, "Empty command")
            action = cmd.get(0).asString().strip().lower()
            if action in ("status", "ping"):
                return self._rpc_ok(reply, self.snapshot())
            if action == "help":
                reply.addString(
                    "status | help | quit | meal SMALL_MEAL|MEDIUM_MEAL|LARGE_MEAL | "
                    "controller <executiveControl RPC words...>"
                )
                return True
            if action == "quit":
                self._running = False
                return self._rpc_ok(reply, {"success": True, "message": "stomachMonitor shutting down"})
            if action == "meal":
                if cmd.size() < 2:
                    return self._rpc_error(reply, "Usage: meal SMALL_MEAL|MEDIUM_MEAL|LARGE_MEAL")
                meal = cmd.get(1).asString().strip().upper()
                if meal not in MEALS:
                    return self._rpc_error(reply, f"Unknown meal: {meal}")
                ok = self.cmd_meal(meal)
                return self._rpc_ok(reply, {"success": bool(ok), "meal": meal})
            if action == "controller":
                words = [cmd.get(i).asString() for i in range(1, cmd.size())]
                if not words:
                    return self._rpc_error(reply, "Usage: controller <executiveControl RPC words...>")
                self.cmd_rpc(words)
                return self._rpc_ok(reply, {"success": True, "forwarded": words})
            return self._rpc_error(reply, f"Unknown command: {action}")
        except Exception as e:
            return self._rpc_error(reply, str(e))

    def interruptModule(self):
        self._running = False
        for port in (self._rpc_port, self._controller_rpc, self._qr_port):
            try:
                if port is not None:
                    port.interrupt()
            except Exception:
                pass
        try:
            if self._root is not None:
                self._root.quit()
        except Exception:
            pass
        return True

    def close(self):
        if self._closed:
            return True
        self._closed = True
        for port in (self._rpc_port, self._controller_rpc, self._qr_port):
            try:
                if port is not None:
                    port.close()
            except Exception:
                pass
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass
        return True

    def _service_tk(self) -> None:
        if self._root is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._running = False

    def _ensure_links(self):
        if self._controller_rpc is None or self._qr_port is None:
            return
        try:
            if self._controller_rpc.getOutputCount() < 1:
                yarp.Network.connect(self._controller_rpc.getName(), self.server, "tcp")
            if self._qr_port.getOutputCount() < 1:
                yarp.Network.connect(self._qr_port.getName(), self.qr_target, "tcp")
        except Exception:
            pass

    def _poll_status(self):
        if self._controller_rpc is None:
            return
        cmd, reply = yarp.Bottle(), yarp.Bottle()
        cmd.clear(); cmd.addString("status")
        if not self._controller_rpc.write(cmd, reply):
            with self._lock:
                self._snap["connected"] = False
            return
        try:
            data = json.loads(reply.get(1).asString())
            with self._lock:
                self._snap.update(
                    connected=True,
                    enabled=bool(data.get("hunger_enabled", False)),
                    level=float(data.get("hunger_level", 100.0)),
                    state=str(data.get("hunger_state", "HS0")),
                    busy=bool(data.get("busy", False)))
        except Exception:
            with self._lock:
                self._snap["connected"] = self._controller_rpc.getOutputCount() > 0

    def _drain_cmds(self):
        while True:
            try:
                kind, arg = self._cmds.get_nowait()
            except queue.Empty:
                return
            if kind == "rpc":
                self._send_rpc(arg)
            elif kind == "meal":
                self._send_meal(arg)

    def _send_rpc(self, words):
        if self._controller_rpc is None:
            return False
        try:
            c, r = yarp.Bottle(), yarp.Bottle()
            c.clear()
            for wd in words:
                c.addString(str(wd))
            ok = bool(self._controller_rpc.write(c, r))
            if not ok:
                with self._lock:
                    self._snap["last_error"] = f"RPC write failed: {' '.join(map(str, words))}"
            return ok
        except Exception as e:
            with self._lock:
                self._snap["last_error"] = str(e)
            return False

    def _send_meal(self, payload):
        if self._qr_port is None:
            return False
        try:
            b = self._qr_port.prepare()
            b.clear()
            b.addString(str(payload))
            self._qr_port.write()
            return True
        except Exception as e:
            with self._lock:
                self._snap["last_error"] = str(e)
            return False

    def _rpc_ok(self, reply: yarp.Bottle, data: Dict[str, Any]) -> bool:
        reply.addString("ok")
        reply.addString(json.dumps(data, ensure_ascii=False))
        return True

    def _rpc_error(self, reply: yarp.Bottle, error: str) -> bool:
        reply.addString("error")
        reply.addString(json.dumps({"success": False, "error": error}, ensure_ascii=False))
        return True


def _run_with_signal_handling(module: StomachMonitorModule, rf: yarp.ResourceFinder) -> bool:
    stop_requested = threading.Event()

    def _on_signal(signum, _frame):
        if stop_requested.is_set():
            return
        stop_requested.set()
        try:
            os.write(2, f"\n[INFO] Signal {signum} received, shutting down.\n".encode("utf-8", "replace"))
        except Exception:
            pass
        module.interruptModule()

    prev_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    try:
        if not module.configure(rf):
            return False
        return bool(module.run_tk_mainloop())
    finally:
        for sig, handler in prev_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        module.close()


def _wants_sim(argv: List[str]) -> bool:
    return any(arg == "--sim" for arg in argv[1:])


def _run_sim_preview() -> None:
    if not HAVE_PIL:
        print("[WARN] Pillow not installed — using a basic vector view. "
              "For the full visuals run:  pip install pillow")
    root = tk.Tk()
    StomachApp(root, SimBackend(), "SIM")
    root.mainloop()


if __name__ == "__main__":
    if not HAVE_YARP:
        if _wants_sim(sys.argv):
            _run_sim_preview()
            raise SystemExit(0)
        print("[ERROR] YARP Python bindings unavailable; run with --sim for offline preview.")
        raise SystemExit(1)

    yarp.Network.init()
    module: Optional[StomachMonitorModule] = None
    try:
        if not yarp.Network.checkNetwork():
            if _wants_sim(sys.argv):
                _run_sim_preview()
                raise SystemExit(0)
            print("[ERROR] YARP network unavailable — start yarpserver first.")
            sys.exit(1)

        module = StomachMonitorModule()
        rf = yarp.ResourceFinder()
        rf.setVerbose(False)
        rf.configure(sys.argv)
        _run_with_signal_handling(module, rf)
    finally:
        if module is not None:
            try:
                module.close()
            except Exception:
                pass
        yarp.Network.fini()
