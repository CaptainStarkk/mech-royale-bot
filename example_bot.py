#!/usr/bin/env python3
"""Quiet survivor bot for mechfight/1.

Strategy: placement is all that counts, so stay alive.
  1. Move early toward the NEXT zone circle and wait there quietly (standing still is silent).
  2. Keep distance from visible mechs and NPC drones.
  3. Face whoever shoots us (rear hits do 1.5x), shoot back only when close or attacked.
  4. Get unstuck from walls with a sidestep + jump/jet.
Standard library only. Logs go to stderr.
"""
import json
import math
import os
import random
import select
import sys

NAME = "quiet-survivor"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(msg):
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def norm(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-6 else (0.0, 0.0)


class Reader:
    """Reads stdin lines; drains everything queued so we can answer only the newest obs."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.buf = b""
        self.eof = False

    def _read_some(self, block):
        if self.eof:
            return
        if not block:
            r, _, _ = select.select([self.fd], [], [], 0)
            if not r:
                return
        try:
            chunk = os.read(self.fd, 1 << 20)
        except BlockingIOError:
            return
        if not chunk:
            self.eof = True
        else:
            self.buf += chunk

    def lines(self):
        while b"\n" not in self.buf and not self.eof:
            self._read_some(True)
        while True:
            before = len(self.buf)
            self._read_some(False)
            if len(self.buf) == before:
                break
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if line:
                out.append(line)
        if self.eof and self.buf.strip():
            out.append(self.buf.strip())
            self.buf = b""
        return out


class Bot:
    def __init__(self):
        self.scan_dir = 1
        self.stuck_ticks = 0
        self.unstick_until = -1.0
        self.unstick_dir = (0.0, 0.0)
        self.threat_yaw = None
        self.threat_until = -1.0
        self.rng = random.Random(0)

    def hello(self, m):
        self.rng.seed(m.get("seed", 0))

    def act(self, o):
        tick = o["tick"]
        t = o.get("t", 0.0)
        me = o.get("self") or {}
        px, py, pz = (me.get("pos") or [0, 0, 0])[:3]
        vx, vy = (me.get("vel") or [0, 0, 0])[:2]
        energy = me.get("energy", 0)
        grounded = me.get("grounded", True)
        cur_yaw = (me.get("aim") or [0, 0])[0]
        in_zone = me.get("in_zone", True)
        act = {"tick": tick}

        # ---- 1. Zone: head for the NEXT circle early ----
        z = o.get("zone") or {}
        cx, cy = (z.get("center") or [px, py])[:2]
        nx, ny = (z.get("next_center") or [cx, cy])[:2]
        nr = z.get("next_radius", z.get("radius", 1e9))
        dzone = math.hypot(nx - px, ny - py)
        margin = max(nr * 0.55, min(nr, 15.0))
        zx = zy = 0.0
        zone_urgent = False
        if not in_zone:
            zx, zy = norm(cx - px, cy - py)
            zone_urgent = True
        elif dzone > margin:
            zx, zy = norm(nx - px, ny - py)
            if dzone > nr * 0.9 or (t > z.get("shrink_start", 0) - 15 and dzone > nr * 0.75):
                zone_urgent = True

        # ---- 2. Threats: keep away from mechs and drones ----
        mechs = o.get("mechs") or []
        enemies = [m for m in mechs if m.get("kind") == "mech"]
        npcs = [m for m in mechs if m.get("kind") == "npc"]
        rx = ry = 0.0
        nearest, nd = None, 1e9
        for e in enemies:
            ex, ey = e["pos"][:2]
            d = math.hypot(ex - px, ey - py)
            if d < nd:
                nd, nearest = d, e
            if d < 140:
                w = (140 - d) / 140
                ux, uy = norm(px - ex, py - ey)
                rx += ux * w * 1.5
                ry += uy * w * 1.5
        for n in npcs:
            ex, ey = n["pos"][:2]
            d = math.hypot(ex - px, ey - py)
            if d < 70:
                w = (70 - d) / 70
                ux, uy = norm(px - ex, py - ey)
                rx += ux * w
                ry += uy * w

        for ev in o.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("bearing") is not None and ev.get("weapon") != "zone":
                self.threat_yaw = ev["bearing"]
                self.threat_until = t + 4.0
        recently_hit = self.threat_yaw is not None and t < self.threat_until
        if recently_hit and not enemies:
            rx -= math.cos(self.threat_yaw) * 0.8
            ry -= math.sin(self.threat_yaw) * 0.8

        # ---- Combine movement ----
        zw = 2.5 if zone_urgent else 1.0
        mx, my = zx * zw + rx, zy * zw + ry
        threatened = (nearest is not None and nd < 140) or recently_hit
        mag = math.hypot(mx, my)
        if mag < 1e-3:
            move = (0.0, 0.0)  # standing still is silent
        else:
            speed = 1.0 if (zone_urgent or threatened) else 0.4  # 0.4 stays under the step-sound speed
            move = (mx / mag * speed, my / mag * speed)

        # ---- 4. Unstick from walls ----
        jump = boost = False
        commanded = math.hypot(*move) > 0.3
        if commanded and math.hypot(vx, vy) < 1.0:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 1)
        if self.stuck_ticks > 8:
            ang = self.rng.choice([math.pi / 2, -math.pi / 2])
            c, s = math.cos(ang), math.sin(ang)
            self.unstick_dir = (move[0] * c - move[1] * s, move[0] * s + move[1] * c)
            self.unstick_until = t + 1.5
            self.stuck_ticks = 0
            jump = grounded
        if t < self.unstick_until:
            move = norm(*self.unstick_dir)
            if not grounded and energy > 20:
                boost = True
        if zone_urgent and commanded and grounded and energy > 35:
            boost = True
        if not in_zone and energy > 10:
            boost = True

        # ---- 3. Aim and fire ----
        weapons = me.get("weapons") or []
        gun = weapons[0] if weapons and weapons[0] and weapons[0].get("kind") == "gun" else None
        gstats = (gun or {}).get("stats") or {}
        muzzle = pz + (me.get("stats") or {}).get("muzzle_height", 7)
        aim_pitch = 0.0
        fire0 = False
        if nearest is not None:
            ex, ey, ez = nearest["pos"][:3]
            evx, evy, evz = (nearest.get("vel") or [0, 0, 0])[:3]
            d = math.hypot(ex - px, ey - py)
            tt = d / gstats.get("speed", 252)
            lx, ly, lz = ex + evx * tt, ey + evy * tt, ez + 4.5 + evz * tt
            aim_yaw = math.atan2(ly - py, lx - px)
            aim_pitch = math.atan2(lz - muzzle, max(1.0, math.hypot(lx - px, ly - py)))
            in_range = d < gstats.get("range", 230) * 0.85
            should_shoot = d < 90 or recently_hit
            if gun and in_range and should_shoot and abs(wrap(aim_yaw - cur_yaw)) < 0.08:
                if gun.get("ammo", 1) > 0 and gun.get("reload", 0) == 0:
                    fire0 = True
        elif recently_hit:
            aim_yaw = self.threat_yaw  # face the attacker: front armor
        else:
            aim_yaw = wrap(cur_yaw + 0.35 * self.scan_dir)  # slowly look around

        if gun and gun.get("ammo", 1) == 0 and gun.get("reload", 0) == 0:
            act["reload"] = True

        # ---- Upgrades: prefer survivability ----
        offer = o.get("offer")
        if offer:
            choices = offer.get("choices") or []
            idx = 0
            for pref in ("armor", "hp", "regen", "sensors"):
                hit = [i for i, c in enumerate(choices) if pref in str(c.get("id", ""))]
                if hit:
                    idx = hit[0]
                    break
            act["pick"] = idx

        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = boost
        act["jump"] = jump
        act["aim"] = [round(aim_yaw, 4), round(aim_pitch, 4)]
        act["fire"] = [fire0, False]
        return act


def main():
    rd = Reader()
    bot = Bot()
    while True:
        lines = rd.lines()
        if not lines and rd.eof:
            return
        latest = None
        ended = False
        for ln in lines:
            try:
                m = json.loads(ln)
            except Exception:
                continue
            ty = m.get("type")
            if ty == "hello":
                bot.hello(m)
                send({"type": "ready", "name": NAME})
            elif ty == "end":
                ended = True
            elif ty == "obs":
                latest = m
        if ended:
            return
        if latest is not None:
            try:
                a = bot.act(latest)
            except Exception as e:  # never crash: crashing = idle for the rest of the match
                log("act error:", repr(e))
                a = {"tick": latest["tick"]}
            send(a)


if __name__ == "__main__":
    main()
