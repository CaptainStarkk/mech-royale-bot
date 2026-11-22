"""Static map model: walkability grid, line-of-sight, A* paths, cover search.

Grid resolution RES (2 m). Cell index c = j * NX + i, center at
x = X0 + (i + 0.5) * RES, y = Y0 + (j + 0.5) * RES.
"""
import array
import heapq
import json
import math
import os
import time

RES = 2.0
INF = float("inf")
SQ2 = math.sqrt(2.0)


class Nav:
    def __init__(self, map_dir, mech_cfg=None):
        t0 = time.time()
        mech_cfg = mech_cfg or {}
        self.step_up = mech_cfg.get("step_up", 1.2)
        self.height = mech_cfg.get("height", 9.0)
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        tr = w["terrain"]
        self.tnx, self.tny = tr["nx"], tr["ny"]
        self.tx0, self.ty0, self.tstep = tr["minX"], tr["minY"], tr["step"]
        scale = tr.get("scale", 0.01)
        a = array.array("h")
        with open(os.path.join(map_dir, "v2", tr.get("file", "terrain.i16")), "rb") as f:
            a.frombytes(f.read())
        if array.array("h", [1]).tobytes() != b"\x01\x00":
            a.byteswap()
        self.terr = [v * scale for v in a]

        b = w["bounds"]
        self.bx0, self.bx1, self.by0, self.by1 = b["minX"], b["maxX"], b["minY"], b["maxY"]
        self.X0, self.Y0 = self.tx0, self.ty0
        self.NX = int(math.ceil((self.tnx - 1) * self.tstep / RES))
        self.NY = int(math.ceil((self.tny - 1) * self.tstep / RES))
        NX, NY = self.NX, self.NY
        N = NX * NY

        # terrain height per cell center
        ter = [0.0] * N
        for j in range(NY):
            y = self.Y0 + (j + 0.5) * RES
            for i in range(NX):
                ter[j * NX + i] = self.ground(self.X0 + (i + 0.5) * RES, y)
        self.ter = ter

        # prisms per cell (scanline fill, even-odd incl. holes)
        cell_prisms = {}
        for pi, p in enumerate(w["prisms"]):
            rings = [p["ring"]] + list(p.get("holes") or [])
            edges = []
            ymin, ymax = INF, -INF
            for r in rings:
                n = len(r) // 2
                for k in range(n):
                    x1, y1 = r[2 * k], r[2 * k + 1]
                    x2, y2 = r[2 * ((k + 1) % n)], r[2 * ((k + 1) % n) + 1]
                    if y1 != y2:
                        edges.append((x1, y1, x2, y2))
                    ymin = min(ymin, y1)
                    ymax = max(ymax, y1)
            j0 = max(0, int((ymin - self.Y0) / RES) - 1)
            j1 = min(NY - 1, int((ymax - self.Y0) / RES) + 1)
            for j in range(j0, j1 + 1):
                y = self.Y0 + (j + 0.5) * RES
                xs = []
                for (x1, y1, x2, y2) in edges:
                    if (y1 <= y < y2) or (y2 <= y < y1):
                        xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                xs.sort()
                for k in range(0, len(xs) - 1, 2):
                    # cells whose center lies in [xa, xb]; widen by half a cell so thin walls register
                    ia = max(0, int(math.ceil((xs[k] - self.X0) / RES - 0.5 - 0.35)))
                    ib = min(NX - 1, int(math.floor((xs[k + 1] - self.X0) / RES - 0.5 + 0.35)))
                    for i in range(ia, ib + 1):
                        cell_prisms.setdefault(j * NX + i, []).append(pi)

        prisms = w["prisms"]
        floor = list(ter)
        blocked = bytearray(N)
        slo = [INF] * N  # solid interval for LOS
        shi = [-INF] * N
        for c, lst in cell_prisms.items():
            lst.sort(key=lambda q: prisms[q]["base"])
            fl = ter[c]
            for q in lst:
                p = prisms[q]
                base, top = p["base"], p["top"]
                if base < slo[c]:
                    slo[c] = base
                if top > shi[c]:
                    shi[c] = top
                if top <= fl + self.step_up:
                    fl = max(fl, top)
                elif base >= fl + self.height + 0.3:
                    pass  # overhead
                elif p["kind"] == "bridge":
                    fl = top
                else:
                    blocked[c] = 1
            floor[c] = fl
        self.floor = floor
        self.slo, self.shi = slo, shi

        # map bounds as walls
        for j in range(NY):
            y = self.Y0 + (j + 0.5) * RES
            for i in range(NX):
                x = self.X0 + (i + 0.5) * RES
                if x < self.bx0 + 3 or x > self.bx1 - 3 or y < self.by0 + 3 or y > self.by1 - 3:
                    blocked[j * NX + i] = 1
        # steep cells (terrain gradient) count as walls for clearance purposes
        for j in range(1, NY - 1):
            for i in range(1, NX - 1):
                c = j * NX + i
                if blocked[c]:
                    continue
                f = floor[c]
                if (abs(floor[c + 1] - f) > 2.4 and abs(floor[c - 1] - f) > 2.4) or \
                   (abs(floor[c + NX] - f) > 2.4 and abs(floor[c - NX] - f) > 2.4):
                    blocked[c] = 1
        self.blocked = blocked

        # clearance: BFS distance (in cells, chamfer) to nearest blocked cell
        clr = [99.0] * N
        pq = []
        for c in range(N):
            if blocked[c]:
                clr[c] = 0.0
                pq.append((0.0, c))
        heapq.heapify(pq)
        lim = 6.0
        while pq:
            d, c = heapq.heappop(pq)
            if d > clr[c] or d >= lim:
                continue
            i, j = c % NX, c // NX
            for di, dj, w_ in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                               (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)):
                ii, jj = i + di, j + dj
                if 0 <= ii < NX and 0 <= jj < NY:
                    n = jj * NX + ii
                    nd = d + w_
                    if nd < clr[n]:
                        clr[n] = nd
                        heapq.heappush(pq, (nd, n))
        self.clr = clr
        # hard: center closer than ~3 m to a wall center -> unwalkable for planning
        self.hard = bytearray(1 if clr[c] < 1.45 else 0 for c in range(N))
        # extra cost near walls
        self.pen = [0.0 if clr[c] >= 4 else (4 - clr[c]) * 0.6 for c in range(N)]
        # cover candidates: walkable cells hugging a wall (good hiding spots)
        self.coverish = bytearray(1 if (not self.hard[c] and clr[c] < 2.6) else 0 for c in range(N))

        self.neigh = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            self.neigh.append((di, dj, dj * NX + di, SQ2 if di and dj else 1.0))
        self.build_s = time.time() - t0

    # ---------- geometry ----------
    def ground(self, x, y):
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        i = int(fx)
        j = int(fy)
        if i < 0:
            i, fx = 0, 0.0
        if j < 0:
            j, fy = 0, 0.0
        if i >= self.tnx - 1:
            i, fx = self.tnx - 2, float(self.tnx - 1)
        if j >= self.tny - 1:
            j, fy = self.tny - 2, float(self.tny - 1)
        u, v = fx - i, fy - j
        t = self.terr
        k = j * self.tnx + i
        a = t[k] * (1 - u) + t[k + 1] * u
        b = t[k + self.tnx] * (1 - u) + t[k + self.tnx + 1] * u
        return a * (1 - v) + b * v

    def cell(self, x, y):
        i = int((x - self.X0) / RES)
        j = int((y - self.Y0) / RES)
        i = 0 if i < 0 else (self.NX - 1 if i >= self.NX else i)
        j = 0 if j < 0 else (self.NY - 1 if j >= self.NY else j)
        return j * self.NX + i

    def center(self, c):
        return (self.X0 + (c % self.NX + 0.5) * RES, self.Y0 + (c // self.NX + 0.5) * RES)

    def walkable(self, c):
        return not self.hard[c]

    def los(self, a, b, step=1.5):
        """True if segment a->b (3D) is not blocked by prisms or terrain."""
        ax, ay, az = a
        bx, by, bz = b
        dx, dy, dz = bx - ax, by - ay, bz - az
        L = math.sqrt(dx * dx + dy * dy)
        n = int(L / step)
        if n < 1:
            return True
        X0, Y0, NX, NY = self.X0, self.Y0, self.NX, self.NY
        slo, shi, ter = self.slo, self.shi, self.ter
        inv = 1.0 / n
        for k in range(1, n):
            f = k * inv
            i = int((ax + dx * f - X0) / RES)
            j = int((ay + dy * f - Y0) / RES)
            if i < 0 or j < 0 or i >= NX or j >= NY:
                return False
            c = j * NX + i
            z = az + dz * f
            if z < ter[c] - 0.3:
                return False
            if slo[c] <= z <= shi[c]:
                return False
        return True

    def straight_ok(self, a, b):
        """Can we walk straight from a to b (2D) through non-hard cells without big climbs?"""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        n = int(L / 1.0) + 1
        prev = self.cell(ax, ay)
        hard, floor = self.hard, self.floor
        for k in range(1, n + 1):
            f = k / n
            c = self.cell(ax + dx * f, ay + dy * f)
            if c == prev:
                continue
            if hard[c] or abs(floor[c] - floor[prev]) > 1.6:
                return False
            prev = c
        return True

    def nearest_walkable(self, x, y, maxr=12):
        c = self.cell(x, y)
        if not self.hard[c]:
            return c
        ci, cj = c % self.NX, c // self.NX
        best, bd = None, INF
        for r in range(1, maxr + 1):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = ci + di, cj + dj
                    if 0 <= i < self.NX and 0 <= j < self.NY:
                        n = j * self.NX + i
                        if not self.hard[n]:
                            d = di * di + dj * dj
                            if d < bd:
                                bd, best = d, n
            if best is not None:
                return best
        return c

    # ---------- planning ----------
    def astar(self, start, goal, extra_cost=None, max_nodes=25000, deadline=None):
        """Returns list of cells from start to goal (or toward the closest reached cell)."""
        NX, NY = self.NX, self.NY
        hard, floor, pen = self.hard, self.floor, self.pen
        su = 1.6
        gx, gy = goal % NX, goal // NX
        g = {start: 0.0}
        par = {start: -1}
        def h(c):
            dx = abs(c % NX - gx)
            dy = abs(c // NX - gy)
            return (dx + dy) + (SQ2 - 2) * min(dx, dy)
        op = [(h(start), 0.0, start)]
        best, bh = start, h(start)
        cnt = 0
        closed = set()
        while op:
            f, gc, c = heapq.heappop(op)
            if c in closed:
                continue
            closed.add(c)
            if c == goal:
                best = c
                break
            hc = f - gc
            if hc < bh:
                bh, best = hc, c
            cnt += 1
            if cnt > max_nodes or (deadline is not None and (cnt & 255) == 0 and time.time() > deadline):
                break
            ci, cj = c % NX, c // NX
            fc = floor[c]
            for di, dj, off, wgt in self.neigh:
                ii, jj = ci + di, cj + dj
                if ii < 0 or jj < 0 or ii >= NX or jj >= NY:
                    continue
                n = c + off
                if hard[n] or n in closed:
                    continue
                if abs(floor[n] - fc) > su:
                    continue
                if di and dj and (hard[c + di] or hard[c + dj * NX]):
                    continue
                ng = gc + wgt * (1.0 + pen[n])
                if extra_cost is not None:
                    ng += extra_cost(n)
                if ng < g.get(n, INF):
                    g[n] = ng
                    par[n] = c
                    heapq.heappush(op, (ng + h(n), ng, n))
        path = []
        c = best
        while c != -1:
            path.append(c)
            c = par[c]
        path.reverse()
        return path

    def dijkstra_local(self, start, radius_m, max_nodes=4000):
        """Path costs (meters) from start to cells within radius."""
        NX, NY = self.NX, self.NY
        hard, floor = self.hard, self.floor
        lim = radius_m / RES
        dist = {start: 0.0}
        op = [(0.0, start)]
        cnt = 0
        while op and cnt < max_nodes:
            d, c = heapq.heappop(op)
            if d > dist.get(c, INF):
                continue
            cnt += 1
            ci, cj = c % NX, c // NX
            fc = floor[c]
            for di, dj, off, wgt in self.neigh:
                ii, jj = ci + di, cj + dj
                if ii < 0 or jj < 0 or ii >= NX or jj >= NY:
                    continue
                n = c + off
                if hard[n] or abs(floor[n] - fc) > 1.6:
                    continue
                if di and dj and (hard[c + di] or hard[c + dj * NX]):
                    continue
                nd = d + wgt
                if nd <= lim and nd < dist.get(n, INF):
                    dist[n] = nd
                    heapq.heappush(op, (nd, n))
        return {c: d * RES for c, d in dist.items()}
