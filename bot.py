#!/usr/bin/env python3
"""Cover-seeking survivor bot for mechfight/1.

Placement is the score, so the bot plays for survival:
  * map-aware navigation (A* on a 2 m grid built from v2/world.json + terrain)
  * reach the next zone circle early, hold a spot next to a wall
  * when an aware enemy has line of sight: fight if clearly winning, else break LOS behind a
    building and regenerate (shooting back while retreating, torso toward the threat for front armor)
  * opportunistic loot: lightning / laser pickups, cubes, close crates, lone NPC drones early
Logs go to stderr only.
"""
import json
import math
import os
import random
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav import Nav, RES  # noqa: E402

NAME = "cover-survivor"
P_SMIN = float(os.environ.get("P_SMIN", "0.5"))
P_SMAX = float(os.environ.get("P_SMAX", "1.3"))
P_TOL = float(os.environ.get("P_TOL", "1.6"))
P_HOLD = float(os.environ.get("P_HOLD", "0.9"))
HERE = os.path.dirname(os.path.abspath(__file__))
PI = math.pi
TAU = 2 * math.pi


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(msg):
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def wrap(a):
    return (a + PI) % TAU - PI


def norm(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-6 else (0.0, 0.0)


WEAPON_VALUE = {"lightning": 3.2, "laser": 3.0, "gun": 2.0, "mortar": 0.5}
UPGRADE_PREF = ["plating", "armor", "nanites", "laser_damage", "laser_charge", "lightning_efficiency",
                "gun_damage", "gun_rate", "lightning_range", "gyros", "servos", "gun_mag",
                "capacitor", "reactor", "gun_velocity", "jets", "lightning_chain", "sensors"]


class Reader:
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
        self.rng = random.Random(0)
        self.nav = None
        self.cfg = {}
        self.enemies = {}      # id -> info (latest)
        self.npcs = {}
        self.charge_since = {}  # enemy id -> t charging started
        self.last_dmg_t = -99.0
        self.last_dmg_bearing = None
        self.attackers = {}    # id -> last t they damaged us
        self.path = []
        self.path_goal = None
        self.path_t = -99.0
        self.goal_kind = ""
        self.hold = None
        self.hold_t = -99.0
        self.cover = None
        self.cover_t = -99.0
        self.stuck = 0
        self.trail = []
        self.unstick_n = 0
        self.unstick_until = -1.0
        self.unstick_dir = (0.0, 0.0)
        self.strafe_dir = 1
        self.strafe_until = 0.0
        self.scan_yaw = None
        self.last_pos = None
        self.ignore_pickups = {}
        self.take_tries = {}
        self.last_enemy_seen_t = -99.0
        self.sound_yaw = None
        self.sound_t = -99.0
        self.deadline = 0.0
        self.prev_aim = (0.0, 0.0)
        self.laser_holding = False

    # ------------------------------------------------------------------ setup
    def hello(self, m):
        self.rng.seed(m.get("seed", 0))
        self.cfg = m.get("config") or {}
        mdir = (m.get("map") or {}).get("dir") or "/arena"
        for d in (mdir, "/arena", os.path.join(HERE, "arena")):
            try:
                if os.path.exists(os.path.join(d, "v2", "world.json")):
                    self.nav = Nav(d, self.cfg.get("mech"))
                    log("nav built in %.2fs from %s" % (self.nav.build_s, d))
                    break
            except Exception as e:
                log("nav build failed", d, repr(e))
        self.wcfg = self.cfg.get("weapons") or {}

    # ------------------------------------------------------------------ helpers
    def time_left(self):
        return self.deadline - time.time()

    def zone_state(self, o, t):
        z = o.get("zone") or {}
        cx, cy = (z.get("center") or [0, 0])[:2]
        r = z.get("radius", 1e9)
        nx, ny = (z.get("next_center") or [cx, cy])[:2]
        nr = z.get("next_radius", r)
        ss, se = z.get("shrink_start", 1e9), z.get("shrink_end", 1e9)
        return cx, cy, r, nx, ny, nr, ss, se, z.get("dps", 0)

    def zone_at(self, zs, tt):
        cx, cy, r, nx, ny, nr, ss, se, _ = zs
        if tt <= ss:
            return cx, cy, r
        if tt >= se:
            return nx, ny, nr
        f = (tt - ss) / max(1e-6, se - ss)
        return cx + (nx - cx) * f, cy + (ny - cy) * f, r + (nr - r) * f

    def in_zone_at(self, zs, x, y, tt, margin=0.0):
        cx, cy, r = self.zone_at(zs, tt)
        return math.hypot(x - cx, y - cy) <= r - margin

    def follow(self, px, py, goal_xy, t, replan=1.5, tol=2.5):
        """Move vector along an A* path to goal_xy."""
        nav = self.nav
        gx, gy = goal_xy
        if math.hypot(gx - px, gy - py) < tol:
            return (0.0, 0.0)
        if nav is None:
            return norm(gx - px, gy - py)
        if nav.straight_ok((px, py), (gx, gy)) and math.hypot(gx - px, gy - py) < 30:
            self.path = []
            return norm(gx - px, gy - py)
        need = (not self.path or self.path_goal is None or
                math.hypot(self.path_goal[0] - gx, self.path_goal[1] - gy) > 6 or
                t - self.path_t > replan)
        if need and self.time_left() > 0.02:
            s = nav.nearest_walkable(px, py)
            g = nav.nearest_walkable(gx, gy)
            dl = time.time() + min(0.022, self.time_left() - 0.012)
            cells = nav.astar(s, g, deadline=dl)
            self.path = [nav.center(c) for c in cells]
            self.path_goal = (gx, gy)
            self.path_t = t
        if not self.path:
            return norm(gx - px, gy - py)
        # drop passed points
        best_k, best_d = 0, 1e9
        for k in range(min(len(self.path), 40)):
            d = math.hypot(self.path[k][0] - px, self.path[k][1] - py)
            if d < best_d:
                best_d, best_k = d, k
        self.path = self.path[best_k:]
        # pure pursuit: farthest point within 14 m with a straight walk
        tgt = self.path[min(1, len(self.path) - 1)]
        for k in range(min(len(self.path) - 1, 8), 0, -1):
            q = self.path[k]
            if math.hypot(q[0] - px, q[1] - py) < 16 and nav.straight_ok((px, py), q):
                tgt = q
                break
        if len(self.path) <= 2:
            tgt = (gx, gy)
        return norm(tgt[0] - px, tgt[1] - py)

    def lead(self, me_pos, muzzle_z, tgt, speed, center_h):
        px, py = me_pos
        ex, ey, ez = tgt["pos"][:3]
        vx, vy, vz = (tgt.get("vel") or [0, 0, 0])[:3]
        if not tgt.get("grounded_guess", True):
            pass
        lx, ly, lz = ex, ey, ez + center_h
        for _ in range(3):
            d = math.sqrt((lx - px) ** 2 + (ly - py) ** 2 + (lz - muzzle_z) ** 2)
            tt = d / max(1.0, speed) + 0.05
            lx, ly = ex + vx * tt, ey + vy * tt
            lz = ez + center_h + (vz * tt if abs(vz) > 0.5 else 0.0)
        yaw = math.atan2(ly - py, lx - px)
        pitch = math.atan2(lz - muzzle_z, max(1.0, math.hypot(lx - px, ly - py)))
        return yaw, pitch, (lx, ly, lz)

    def find_cover(self, me, threats, zs, t, radius=72.0):
        nav = self.nav
        if nav is None or not threats:
            return None
        px, py, pz = me
        s = nav.nearest_walkable(px, py)
        dist = nav.dijkstra_local(s, radius, max_nodes=6000)
        cands = []
        for c, d in dist.items():
            x, y = nav.center(c)
            if not self.in_zone_at(zs, x, y, t + d / 11.0 + 6, 3.0):
                continue
            pen = 0.0
            for th in threats:
                ex, ey = th[0], th[1]
                dn = math.hypot(px - ex, py - ey)
                dc = math.hypot(x - ex, y - ey)
                if dc < dn:
                    pen += (dn - dc) * 1.2
                if dc < 30:
                    pen += (30 - dc) * 2
            pen -= 3.0 if nav.coverish[c] else 0.0
            cands.append((d + pen, c, x, y))
        cands.sort()
        checks = 0
        for score, c, x, y in cands:
            if checks > 110 or self.time_left() < 0.008:
                break
            fz = nav.floor[c]
            hidden = True
            for th in threats:
                checks += 1
                eye = (th[0], th[1], th[2] + 8.0)
                if nav.los(eye, (x, y, fz + 8.8), 2.0) or nav.los(eye, (x, y, fz + 4.5), 2.0):
                    hidden = False
                    break
            if hidden:
                return (x, y)
        return None

    def exposed(self, me_top, me_mid, e):
        if self.nav is None:
            return True
        eye = (e["pos"][0], e["pos"][1], e["pos"][2] + 8.0)
        return self.nav.los(eye, me_top) or self.nav.los(eye, me_mid)

    # ------------------------------------------------------------------ main decision
    def act(self, o, t_recv):
        self.deadline = t_recv + 0.040
        nav = self.nav
        tick = o["tick"]
        t = o.get("t", 0.0)
        me = o.get("self") or {}
        px, py, pz = (me.get("pos") or [0, 0, 0])[:3]
        vx, vy = (me.get("vel") or [0, 0, 0])[:2]
        hp, mhp = me.get("hp", 1), max(1, me.get("max_hp", 1400))
        hpf = hp / mhp
        energy = me.get("energy", 0)
        grounded = me.get("grounded", True)
        cur_yaw, cur_pitch = (me.get("aim") or [0, 0])[:2]
        st = me.get("stats") or {}
        scale = 1 + 0.05 * me.get("cubes", 0)
        muzzle = pz + st.get("muzzle_height", 7)
        me_top = (px, py, pz + st.get("eye_height", 8) + 0.5)
        me_mid = (px, py, pz + 4.5 * scale)
        weapons = me.get("weapons") or [None, None]
        kinds = [w.get("kind") if w else None for w in weapons]
        zs = self.zone_state(o, t)
        act = {"tick": tick}

        # ---- tracking ----
        mechs = o.get("mechs") or []
        enemies = [m for m in mechs if m.get("kind") == "mech"]
        npcs = [m for m in mechs if m.get("kind") == "npc"]
        if enemies:
            self.last_enemy_seen_t = t
        for e in enemies:
            self.enemies[e["id"]] = (e["pos"][0], e["pos"][1], e["pos"][2], t, e.get("hp", 1400))
        for k in [k for k, v in self.enemies.items() if t - v[3] > 12]:
            del self.enemies[k]
        for e in enemies:
            if e.get("charging"):
                self.charge_since.setdefault(e["id"], t)
            else:
                self.charge_since.pop(e["id"], None)
        for ev in o.get("events") or []:
            ty = ev.get("type")
            if ty == "damaged" and ev.get("weapon") != "zone":
                self.last_dmg_t = t
                if ev.get("bearing") is not None:
                    self.last_dmg_bearing = ev["bearing"]
                if ev.get("by") is not None:
                    self.attackers[ev["by"]] = t
            elif ty == "take_failed":
                self.ignore_pickups[ev.get("id")] = t + 20
            elif ty == "took":
                self.ignore_pickups[ev.get("id")] = t + 1e9
        for s in o.get("sounds") or []:
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "boost", "step", "jet") \
                    and s.get("dist") in ("near", "mid"):
                self.sound_yaw = s.get("bearing")
                self.sound_t = t

        recently_hit = t - self.last_dmg_t < 2.5

        # ---- threat assessment ----
        threats = []      # enemies that can see & shoot us
        visible = []
        for e in enemies:
            ex, ey, ez = e["pos"][:3]
            d = math.hypot(ex - px, ey - py)
            e["_d"] = d
            exp = self.exposed(me_top, me_mid, e) if d < 300 else False
            e["_exp"] = exp
            # do they look at us?
            to_me = math.atan2(py - ey, px - ex)
            facing = abs(wrap((e.get("aim") or [0, 0])[0] - to_me))
            aware = d < 50 or facing < 1.15 or (t - self.attackers.get(e["id"], -99) < 4)
            e["_aware"] = aware
            wk = e.get("weapons") or []
            rng_ = 45 if wk == ["lightning"] else 285
            if exp:
                visible.append(e)
                if aware and d < rng_:
                    threats.append(e)
        # unseen attacker -> virtual threat on the damage bearing
        virtual = []
        if recently_hit and not threats and self.last_dmg_bearing is not None:
            b = self.last_dmg_bearing
            virtual.append((px + math.cos(b) * 90, py + math.sin(b) * 90, pz))

        def fight_ok(e):
            ehp = e.get("hp", 1400)
            if e["_d"] < 32:
                return True
            if ehp < 300 and hpf > 0.25:
                return True
            if "lightning" in kinds and e["_d"] < 42 and hpf > 0.3:
                return True
            return hp > ehp * 1.8 and hpf > 0.5

        mode = "normal"
        target = None
        self.backoff = False
        if threats:
            threats.sort(key=lambda e: e["_d"])
            target = threats[0]
            if len(threats) <= 1 and fight_ok(target):
                mode = "fight"
            elif len(threats) >= 2 and all(e["_d"] < 32 for e in threats[:1]) and hpf > 0.5:
                mode = "fight"
            else:
                mode = "retreat"
        elif virtual:
            mode = "retreat"
        elif hpf < 0.75 and any(t - v[3] < 9 and math.hypot(v[0] - px, v[1] - py) < 170
                                for v in self.enemies.values()):
            # wounded with enemies around: stay out of their sight and regenerate
            mode = "retreat"
            threats = []
            virtual = [(v[0], v[1], v[2]) for v in self.enemies.values()
                       if t - v[3] < 9 and math.hypot(v[0] - px, v[1] - py) < 170]
        else:
            # opportunistic shot at a weak enemy that has not noticed us
            weak = [e for e in visible if e.get("hp", 1400) < 420 and e["_d"] < 160 and hpf > 0.55]
            if weak:
                target = min(weak, key=lambda e: e["_d"])
                mode = "fight"
            elif visible and hpf < 0.9:
                # someone could see us soon: slip out of their sight
                mode = "retreat"
                threats = visible

        # ---- zone urgency ----
        cx, cy, r, nxc, nyc, nr, ss, se, dps = zs
        d_next = math.hypot(px - nxc, py - nyc)
        margin_n = min(nr * 0.25, 25.0) + 4
        travel = max(0.0, d_next - (nr - margin_n)) * 1.35 / st.get("walk_speed", 11.7)
        in_zone = me.get("in_zone", True)
        # time until the shrinking boundary reaches us
        if d_next <= nr:
            t_hit = 1e9
        else:
            t_hit = se
            steps = 12
            for k in range(steps + 1):
                tt = ss + (se - ss) * k / steps
                if not self.in_zone_at(zs, px, py, tt, 2.0):
                    t_hit = tt
                    break
        zone_urgent = (not in_zone) or (t_hit - t < travel + 8)
        zone_soon = t_hit - t < travel + 25

        # ---- goal selection ----
        move = (0.0, 0.0)
        speed = 1.0
        boost = False
        label = mode
        want_aim = None
        fire = [False, False]

        pickups = o.get("pickups") or []
        take_id = None
        drop_slot = None

        def pickup_choice():
            best = None
            for p in pickups:
                pid = p.get("id")
                if self.ignore_pickups.get(pid, -1) > t:
                    continue
                k = p.get("kind")
                if k in kinds:
                    continue
                x, y, z = p["pos"][:3]
                if nav is not None and z - nav.floor[nav.cell(x, y)] > 2.5:
                    continue  # roof
                d = math.hypot(x - px, y - py)
                if d > 75:
                    continue
                gain = 0.0
                dslot = None
                if kinds[1] is None:
                    gain = WEAPON_VALUE.get(k, 0) - 0.3
                    if k == "mortar":
                        gain = 0.0
                else:
                    # swap out the weakest slot
                    vals = [WEAPON_VALUE.get(kk, 0) for kk in kinds]
                    ws = 0 if vals[0] <= vals[1] else 1
                    if w_busy(ws):
                        continue
                    gain = WEAPON_VALUE.get(k, 0) - vals[ws] - 0.6
                    dslot = ws
                if gain <= 0:
                    continue
                score = d - gain * 15
                if best is None or score < best[0]:
                    best = (score, p, dslot, d)
            return best

        def w_busy(slot):
            w = weapons[slot] if slot < len(weapons) else None
            return bool(w and w.get("kind") == "laser" and w.get("charge", 0) > 0)

        # take anything in reach right now
        pc = pickup_choice()
        if pc is not None:
            _, p, dslot, d = pc
            if d < 3.6 and abs(p["pos"][2] - pz) < 3.8:
                take_id = p["id"]
                drop_slot = dslot
                self.take_tries[take_id] = self.take_tries.get(take_id, 0) + 1
                if self.take_tries[take_id] > 6:
                    self.ignore_pickups[take_id] = t + 30

        goal = None
        if mode == "retreat":
            th_pts = [(e["pos"][0], e["pos"][1], e["pos"][2]) for e in threats] + virtual
            need_new = (self.cover is None or t - self.cover_t > 1.2)
            if self.cover is not None and not need_new:
                # still hidden?
                fz = nav.floor[nav.cell(*self.cover)] if nav else pz
                for th in th_pts:
                    if nav and nav.los((th[0], th[1], th[2] + 8), (self.cover[0], self.cover[1], fz + 8.8), 2.0):
                        need_new = True
                        break
            if need_new and self.time_left() > 0.02:
                self.cover = self.find_cover((px, py, pz), th_pts, zs, t)
                self.cover_t = t
            if self.cover is not None:
                goal = self.cover
                label = "cover"
            else:
                # run directly away from threats along an open direction
                ax = sum(px - q[0] for q in th_pts)
                ay = sum(py - q[1] for q in th_pts)
                base = math.atan2(ay, ax)
                best = None
                for k in range(12):
                    a = base + (k // 2 + 0.0) * (0.45 if k % 2 else -0.45)
                    q = (px + math.cos(a) * 14, py + math.sin(a) * 14)
                    if nav is None or nav.straight_ok((px, py), q):
                        if self.in_zone_at(zs, q[0], q[1], t + 3, 0):
                            best = q
                            break
                goal = best
                label = "flee"
                if target is not None and (goal is None or target["_d"] < 160):
                    mode = "fight"
                    goal = None
                    self.backoff = True
            if zone_urgent:
                goal = None  # zone first
            boost = energy > 25
        if mode == "fight" and target is not None:
            # strafe around the target at a weapon-appropriate distance
            ex, ey = target["pos"][:2]
            d = target["_d"]
            if t > self.strafe_until:
                self.strafe_dir = self.rng.choice([-1, 1])
                self.strafe_until = t + self.rng.uniform(P_SMIN, P_SMAX)
            ux, uy = norm(ex - px, ey - py)
            sx, sy = -uy * self.strafe_dir, ux * self.strafe_dir
            want_d = 26 if "lightning" in kinds and target.get("hp", 1400) < hp else 90
            if self.backoff:
                want_d = 400
            radial = 0.0
            if d > want_d + 15:
                radial = 0.7
            elif d < want_d - 10:
                radial = -0.6
            mx, my = sx + ux * radial, sy + uy * radial
            q = (px + mx * 6, py + my * 6)
            if nav is not None and not nav.straight_ok((px, py), q):
                self.strafe_dir *= -1
                self.strafe_until = t + 0.8
                sx, sy = -sx, -sy
                mx, my = sx + ux * radial, sy + uy * radial
            move = norm(mx, my)
            boost = energy > 45 and d > 40
            label = "fight %d" % target["id"]
            if zone_urgent:
                goal = None

        # dodge a laser that is aimed at us
        laser_warn = False
        for e in enemies:
            if e.get("charging") and e.get("_exp"):
                to_me = math.atan2(py - e["pos"][1], px - e["pos"][0])
                if abs(wrap((e.get("aim") or [0, 0])[0] - to_me)) < 0.12:
                    laser_warn = True

        crates = o.get("crates") or []
        cubes = o.get("cubes") or []
        quiet = t - self.last_enemy_seen_t > 5 and not recently_hit
        crate_target = None
        npc_target = None
        if mode == "normal":
            if zone_urgent:
                goal = None
            elif pc is not None and pc[3] < 75:
                goal = (pc[1]["pos"][0], pc[1]["pos"][1])
                label = "loot " + pc[1].get("kind", "")
            else:
                # cubes
                bc = None
                for c in cubes:
                    x, y, z = c["pos"][:3]
                    d = math.hypot(x - px, y - py)
                    if d < 60 and abs(z - pz) < 3.5 and (bc is None or d < bc[0]):
                        if nav is None or z - nav.floor[nav.cell(x, y)] < 2.5:
                            bc = (d, (x, y))
                if bc is not None and (quiet or bc[0] < 20):
                    goal = bc[1]
                    label = "cube"
                elif quiet and not zone_soon:
                    # crates close by
                    for c in crates:
                        x, y, z = c["pos"][:3]
                        d = math.hypot(x - px, y - py)
                        if d < 45 and (nav is None or z - nav.floor[nav.cell(x, y)] < 2.5):
                            if crate_target is None or d < crate_target[0]:
                                crate_target = (d, c)
                    # a lone NPC drone, early, healthy
                    if crate_target is None and hpf > 0.75 and t < 230:
                        for n in npcs:
                            d = math.hypot(n["pos"][0] - px, n["pos"][1] - py)
                            if d < 80 and (npc_target is None or d < npc_target[0]):
                                npc_target = (d, n)
            if crate_target is not None:
                label = "crate"
                cd, c = crate_target
                if cd > 30:
                    goal = (c["pos"][0], c["pos"][1])
            elif npc_target is not None:
                label = "npc"
                nd_, n = npc_target
                if nd_ > 55:
                    goal = (n["pos"][0], n["pos"][1])
            elif t - self.last_dmg_t > 5 and goal is None:
                pass

        # zone / hold spot
        if goal is None and (mode == "normal" or zone_urgent) and not (mode == "fight" and not zone_urgent):
            if t - self.hold_t > 4 or self.hold is None or zone_urgent and \
                    math.hypot(self.hold[0] - nxc, self.hold[1] - nyc) > nr - margin_n:
                self.hold = self.pick_hold(px, py, zs)
                self.hold_t = t
            if self.hold is not None:
                goal = self.hold
                if label == "normal":
                    label = "hold" if math.hypot(goal[0] - px, goal[1] - py) < 4 else "zone"

        if goal is not None and mode != "fight":
            move = self.follow(px, py, goal, t)
        elif goal is not None and mode == "fight" and zone_urgent:
            zm = self.follow(px, py, goal, t)
            move = norm(move[0] * 0.4 + zm[0], move[1] * 0.4 + zm[1])

        # move speed: quiet walk when idle and nothing nearby
        if mode == "normal" and not zone_urgent and label in ("hold", "zone") and not recently_hit:
            dgo = math.hypot(goal[0] - px, goal[1] - py) if goal else 0
            if dgo > 40 or zone_soon:
                speed = 1.0
            else:
                speed = 1.0
        if zone_urgent and energy > 30:
            boost = True
        if not in_zone and energy > 8:
            boost = True

        # mortar shells: move away from predicted impacts
        g = self.wcfg.get("mortar", {}).get("gravity", 22)
        for pr in o.get("projectiles") or []:
            if pr.get("kind") != "mortar":
                continue
            x, y, z = pr["pos"][:3]
            ux, uy, uz = pr["vel"][:3]
            # time to fall to our height
            a, b, c = -0.5 * g, uz, z - (pz + 1)
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tt = (-b - math.sqrt(disc)) / (2 * a)
            if tt < 0 or tt > 6:
                continue
            ix, iy = x + ux * tt, y + uy * tt
            if math.hypot(ix - px, iy - py) < 11:
                ax_, ay_ = norm(px - ix, py - iy)
                if ax_ == 0 and ay_ == 0:
                    ax_, ay_ = 1, 0
                move = norm(move[0] * 0.3 + ax_, move[1] * 0.3 + ay_)
                boost = energy > 10
                label = "dodge shell"

        if laser_warn and mode != "fight":
            # sidestep perpendicular to keep moving target
            if self.cover is None:
                boost = energy > 10

        # ---- unstick ----
        jump = False
        mlen = math.hypot(*move)
        spd = math.hypot(vx, vy)
        # progress-based stuck detection: commanded to move but displaced < 3 m in 2 s
        self.trail.append((t, px, py, mlen > 0.3 and mode != "fight"))
        while self.trail and t - self.trail[0][0] > 2.0:
            self.trail.pop(0)
        if t >= self.unstick_until and len(self.trail) >= 18 and all(q[3] for q in self.trail):
            q0 = self.trail[0]
            if math.hypot(px - q0[1], py - q0[2]) < 3.0:
                self.unstick_n += 1
                if self.unstick_n % 2 == 1:
                    self.unstick_dir = move            # jet straight over the obstacle
                else:
                    ang = self.rng.choice([PI / 2, -PI / 2, PI * 0.75, -PI * 0.75])
                    c_, s_ = math.cos(ang), math.sin(ang)
                    self.unstick_dir = (move[0] * c_ - move[1] * s_, move[0] * s_ + move[1] * c_)
                self.unstick_until = t + 1.6
                self.trail = []
                self.path = []
        if t < self.unstick_until:
            move = norm(*self.unstick_dir)
            if grounded and t < self.unstick_until - 1.2:
                jump = True
            if not grounded and energy > 5:
                boost = True

        # ---- weapons / aim ----
        aim_target = target
        if aim_target is None and mode != "fight":
            # look at threats even when not fighting
            if visible:
                aim_target = min(visible, key=lambda e: e["_d"])
        fire_ok_target = aim_target is not None and aim_target.get("_exp")
        gslot = kinds.index("gun") if "gun" in kinds else None
        lslot = kinds.index("laser") if "laser" in kinds else None
        zslot = kinds.index("lightning") if "lightning" in kinds else None

        def ready(slot):
            w = weapons[slot]
            return w is not None and w.get("cooldown", 0) <= 0.02

        shoot_obj = None   # (pos, vel, center_h, radius, is_mech)
        if aim_target is not None:
            shoot_obj = (aim_target, 4.5 * (1 + 0.05 * aim_target.get("cubes", 0)), 2.5)
        elif crate_target is not None:
            c = crate_target[1]
            shoot_obj = ({"pos": c["pos"], "vel": [0, 0, 0]}, 1.2, 1.4)
        elif npc_target is not None:
            n = npc_target[1]
            shoot_obj = (n, 1.75, 1.8)
        elif mode == "normal" and npcs and recently_hit:
            n = min(npcs, key=lambda q: math.hypot(q["pos"][0] - px, q["pos"][1] - py))
            if math.hypot(n["pos"][0] - px, n["pos"][1] - py) < 60:
                shoot_obj = (n, 1.75, 1.8)

        laser_state = weapons[lslot] if lslot is not None else None
        charging_now = bool(laser_state and laser_state.get("charge", 0) > 0)
        if shoot_obj is not None:
            obj, ch, rad = shoot_obj
            ox, oy, oz = obj["pos"][:3]
            d3 = math.sqrt((ox - px) ** 2 + (oy - py) ** 2 + (oz + ch - muzzle) ** 2)
            los_ok = nav is None or nav.los((px, py, muzzle), (ox, oy, oz + ch), 1.5)
            tol = max(0.012, math.atan2(rad * 0.75, max(1.0, d3)))
            use_laser = lslot is not None and d3 < 275 and (ready(lslot) or charging_now)
            if use_laser:
                lvx, lvy = (obj.get("vel") or [0, 0, 0])[:2]
                lx, ly = ox + lvx * 0.1, oy + lvy * 0.1
                yaw = math.atan2(ly - py, lx - px)
                pitch = math.atan2(oz + ch - muzzle, max(1.0, math.hypot(lx - px, ly - py)))
            else:
                spd_ = (weapons[gslot]["stats"].get("speed", 252) if gslot is not None else 252)
                yaw, pitch, _ = self.lead((px, py), muzzle, obj, spd_, ch)
            want_aim = (yaw, pitch)
            err_y = abs(wrap(yaw - cur_yaw))
            err_p = abs(pitch - cur_pitch)
            aligned = err_y < tol and err_p < tol * 1.5
            # gun
            if gslot is not None and los_ok:
                gw = weapons[gslot]
                grange = gw.get("stats", {}).get("range", 230)
                if d3 < grange * 0.95 and gw.get("ammo", 0) > 0 and gw.get("reload", 0) == 0:
                    g_yaw, g_pitch, _ = self.lead((px, py), muzzle, obj, gw["stats"].get("speed", 252), ch)
                    if abs(wrap(g_yaw - cur_yaw)) < tol * P_TOL and abs(g_pitch - cur_pitch) < tol * 1.5 * P_TOL:
                        fire[gslot] = True
            # lightning
            if zslot is not None and ready(zslot):
                zr = weapons[zslot].get("stats", {}).get("range", 45)
                cone = math.radians(weapons[zslot].get("stats", {}).get("cone_deg", 30)) / 2
                dh = math.hypot(ox - px, oy - py)
                if dh < zr * 0.95 and abs(wrap(math.atan2(oy - py, ox - px) - cur_yaw)) < cone * 0.8 \
                        and energy > 6 and los_ok:
                    fire[zslot] = True
            # laser: hold to charge, release when full and aligned
            if lslot is not None:
                lw = weapons[lslot]
                full = lw.get("stats", {}).get("charge", 0.75)
                chg = lw.get("charge", 0)
                if charging_now:
                    release = chg >= full * 0.97 and aligned and los_ok
                    if chg > full + 1.7:
                        release = True
                    if not release and (los_ok or chg < full + 1.0):
                        fire[lslot] = True
                elif ready(lslot) and los_ok and d3 < 275 and err_y < 0.5:
                    if mode == "fight" or (aim_target is not None and recently_hit) or shoot_obj[2] != 2.5:
                        fire[lslot] = True
        else:
            # keep any charge from auto firing randomly: release it
            if charging_now and laser_state.get("charge", 0) > 1.5:
                fire[lslot] = False
            elif charging_now:
                fire[lslot] = True

        if want_aim is None:
            if recently_hit and self.last_dmg_bearing is not None:
                want_aim = (self.last_dmg_bearing, 0.0)
            elif threats:
                e = threats[0]
                want_aim = (math.atan2(e["pos"][1] - py, e["pos"][0] - px), 0.0)
            elif t - self.sound_t < 2.0 and self.sound_yaw is not None:
                want_aim = (self.sound_yaw, 0.0)
            elif mlen > 0.3 and label in ("zone", "loot", "cube") or label.startswith("loot"):
                # look roughly where we are going, with a slow sway
                sway = math.sin(t * 1.3) * 0.9
                want_aim = (math.atan2(move[1], move[0]) + sway, 0.0)
            else:
                if self.scan_yaw is None:
                    self.scan_yaw = cur_yaw
                self.scan_yaw = wrap(self.scan_yaw + 0.28)
                want_aim = (self.scan_yaw, 0.0)

        # gun reload when empty, or topping up when quiet
        if gslot is not None:
            gw = weapons[gslot]
            mag = gw.get("stats", {}).get("mag", 12)
            if gw.get("ammo", 0) == 0 and gw.get("reload", 0) == 0:
                act["reload"] = True
            elif gw.get("ammo", 0) < mag * 0.5 and quiet and gw.get("reload", 0) == 0 and not fire[gslot]:
                act["reload"] = True

        # ---- upgrades ----
        offer = o.get("offer")
        if offer:
            choices = offer.get("choices") or []
            idx, br = 0, 1e9
            for i, c in enumerate(choices):
                cid = str(c.get("id", ""))
                rank = UPGRADE_PREF.index(cid) if cid in UPGRADE_PREF else 50
                if rank < br:
                    br, idx = rank, i
            act["pick"] = idx

        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot

        sp = speed
        act["move"] = [round(move[0] * sp, 3), round(move[1] * sp, 3)]
        act["boost"] = bool(boost and mlen > 0.2) or (bool(boost) and not grounded)
        act["jump"] = jump
        pmin = st.get("min_pitch", -0.7)
        pmax = st.get("max_pitch", 1.3)
        act["aim"] = [round(wrap(want_aim[0]), 4), round(max(pmin, min(pmax, want_aim[1])), 4)]
        act["fire"] = fire
        act["debug"] = {"label": label[:60]}
        return act

    def pick_hold(self, px, py, zs):
        """A spot inside the next circle, near us, preferably hugging a wall."""
        cx, cy, r, nxc, nyc, nr, ss, se, _ = zs
        nav = self.nav
        if nr < 1:
            tx, ty = nxc, nyc
        else:
            dn = math.hypot(px - nxc, py - nyc)
            want = nr * (P_HOLD if nr > 100 else 0.35)
            if dn <= want:
                tx, ty = px, py
            else:
                ux, uy = norm(px - nxc, py - nyc)
                tx, ty = nxc + ux * want, nyc + uy * want
        if nav is None:
            return (tx, ty)
        # search nearby coverish cells around (tx, ty)
        c0 = nav.cell(tx, ty)
        ci, cj = c0 % nav.NX, c0 // nav.NX
        best, bs = None, 1e9
        R = 10
        for dj in range(-R, R + 1, 2):
            for di in range(-R, R + 1, 2):
                i, j = ci + di, cj + dj
                if not (0 <= i < nav.NX and 0 <= j < nav.NY):
                    continue
                c = j * nav.NX + i
                if nav.hard[c]:
                    continue
                x, y = nav.center(c)
                if nr >= 1 and math.hypot(x - nxc, y - nyc) > nr * 0.8:
                    continue
                s = math.hypot(di, dj) * RES * 0.5 + (0 if nav.coverish[c] else 12)
                if s < bs:
                    bs, best = s, (x, y)
        if best is None:
            c = nav.nearest_walkable(tx, ty, 20)
            best = nav.center(c)
        return best


def main():
    rd = Reader()
    bot = Bot()
    while True:
        lines = rd.lines()
        t_recv = time.time()
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
                try:
                    bot.hello(m)
                except Exception as e:
                    log("hello error:", repr(e))
                send({"type": "ready", "name": NAME})
            elif ty == "end":
                ended = True
            elif ty == "obs":
                latest = m
        if ended:
            return
        if latest is not None:
            try:
                a = bot.act(latest, t_recv)
            except Exception as e:
                import traceback
                log("act error:", repr(e), traceback.format_exc())
                a = {"tick": latest["tick"]}
            send(a)


if __name__ == "__main__":
    main()
