// Rock sculpt tool (Rockscape tab): raise / lower / smooth / flatten / tunnel brushes on
// a voxel density grid of the rockscape, in real time, like the terrain tools of
// park-builder games. Left-drag sculpts; holding still keeps working. The mouse wheel
// sets the brush size.
//
// Raise / Lower (D-843) work like a terrain brush seen from your eye: everything under
// the round brush that faces you moves toward you (Raise) or away from you (Lower) at a
// steady speed, most in the middle and fading to nothing at the rim. Passes add up, so
// scrubbing builds or digs gradually and holding still grows a smooth dome or hollow.
// Raise never removes rock and Lower never adds any.
//
// Tunnel works along a fixed straight line (the tank axis nearest to the view, or an
// axis the user picks and turns in 15° steps): it drills a straight round hole, deeper
// the longer you hold, and stops once it breaks through; with Ctrl it extrudes an arm of
// rock out toward you instead.
//
// Data: Float32 density per grid node, 0 = water, 255 = rock, surface = 127.5 (the
// server's reefflow.ui.sculpt module builds the starting grid and turns the result
// into the project's rockscape STL). Display: naive surface nets, re-meshed per
// 16-cell chunk, only where a brush touched. Picking: ray-marching the grid itself.
import * as THREE from 'three';
import { api, el, clear, num, status, busy, kvTable, selectRow, checkRow } from './util.js';
import { rockMaterial, disposeTree } from './viewport.js';

const ISO = 127.5;
const CH = 16;                    // cells per mesh-chunk edge
const B = 16;                     // nodes per undo-block edge
const SHIFT_SPEED = 0.08;         // m/s: Raise / Lower surface speed at the brush centre at strength 1 (D-843)
const WALL_GAP = 0.01;            // m: Raise / Extrude leave this much water at the glass and the water surface
const SMOOTH_RATE = 10.0;
const FLATTEN_RATE = 8.0;
const UNDO_BYTES = 160e6;         // undo history memory cap
const TOOLS = {
  raise: { label: 'Raise', title: 'Raise: rock under the brush comes toward you, most in the middle; passes add up, holding still grows a dome (1)', key: '1', color: 0x2fbf71 },
  lower: { label: 'Lower', title: 'Lower: rock under the brush moves away from you, most in the middle; passes add up, holding still digs a hollow (2)', key: '2', color: 0xe8504a },
  smooth: { label: 'Smooth', title: 'Smooth: round off bumps and steps (3)', key: '3', color: 0x3a8ee6 },
  flatten: { label: 'Flatten', title: 'Flatten: level everything under the brush to the height where the stroke starts; cuts rock above, fills below, so dragging past an edge builds a flat shelf (4)', key: '4', color: 0xe0a020 },
  drill: { label: 'Tunnel', title: 'Tunnel: drill a straight round hole along the brush direction, deeper the longer you hold; stops when it breaks through. Ctrl: extrude an arm of rock toward you instead (5)', key: '5', color: 0xb05ce0 },
};
const LABEL = { raise: 'Raise', lower: 'Lower', smooth: 'Smooth', flatten: 'Flatten', drill: 'Tunnel', extrude: 'Extrude' };
// Mouse pointer per tool: a crosshair with a small badge in the tool's colour and symbol, so
// the tool shows right at the pointer (the 3-D circle shows where it acts on the rock).
const GLYPH = {
  raise: 'M20 8H28M24 4V12', lower: 'M20 8H28', smooth: 'M19.5 9.5q2.25-4 4.5 0t4.5 0',
  flatten: 'M19.5 11H28.5M21.5 5.5L24 8L26.5 5.5', drill: 'M27 8A3 3 0 1 1 21 8A3 3 0 1 1 27 8', extrude: 'M24 12V4.5M21 7.5L24 4.5L27 7.5',
};
const _cursors = {};
export function toolCursor(tool) {
  if (_cursors[tool]) return _cursors[tool];
  const col = '#' + TOOLS[tool === 'extrude' ? 'drill' : tool].color.toString(16).padStart(6, '0');
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'>` +
    `<g stroke-linecap='round' fill='none'><path d='M10 16H22M16 10V22' stroke='white' stroke-width='3.5'/><path d='M10 16H22M16 10V22' stroke='black' stroke-width='1.5'/></g>` +
    `<circle cx='24' cy='8' r='7.2' fill='${col}' stroke='white' stroke-width='1.5'/>` +
    `<path d='${GLYPH[tool]}' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' fill='none'/></svg>`;
  _cursors[tool] = `url("data:image/svg+xml,${encodeURIComponent(svg)}") 16 16, crosshair`;
  return _cursors[tool];
}
const DX_CHOICES = [[0.003, '3 mm (fine, slower)'], [0.004, '4 mm (default)'], [0.005, '5 mm'], [0.006, '6 mm (fast)']];
const DRILL_SPEED = 0.1;          // m/s: how fast Tunnel drills / Extrude extrudes at strength 1
const MAX_RADIUS = 0.2;           // m
const STEP_DEG = 15;              // turn / tilt step of a fixed direction
const DIR_MODES = {
  view: { label: 'Straight in', title: 'Tunnel straight into the rock along the tank axis nearest to your view: front → back when you look through the front glass, straight down from above (V)' },
  fixed: { label: 'Fixed', title: 'Tunnel along a direction you pick; turn and tilt it in 15° steps (Q / E turn, W / S tilt)' },
};
// fixed-direction presets: [az, el] in degrees; az 0 = toward +x (right), 90 = toward the back glass
const AXIS_PRESETS = [['Front → back', 90, 0], ['Back → front', -90, 0], ['Left → right', 0, 0], ['Right → left', 180, 0], ['Down', 90, -90], ['Up', 90, 90]];

const rad = (deg) => deg * Math.PI / 180;
const deg = (r) => r * 180 / Math.PI;
const wrap180 = (a) => ((a + 180) % 360 + 360) % 360 - 180;
const round15 = (a) => Math.round(a / STEP_DEG) * STEP_DEG;

// unit vector of a direction: az in plan (0 = +x, 90 = +y), el up from horizontal
export function dirFromAngles(az, el) {
  const c = Math.cos(rad(el));
  const v = [c * Math.cos(rad(az)), c * Math.sin(rad(az)), Math.sin(rad(el))].map(x => Math.abs(x) < 1e-12 ? 0 : x);
  return new THREE.Vector3(...v).normalize();
}

// the tank axis (±x, ±y, ±z) closest to a vector
export function snapAxis(v) {
  const a = [Math.abs(v.x), Math.abs(v.y), Math.abs(v.z)];
  const m = a.indexOf(Math.max(...a));
  const out = new THREE.Vector3();
  out.setComponent(m, Math.sign(v.getComponent(m)) || 1);
  return out;
}

// plain-words name of a direction (the way the brush travels into the rock)
export function dirLabel(d) {
  if (!d) return 'follows the surface under the cursor';
  if (d.z <= -0.9999) return 'straight down';
  if (d.z >= 0.9999) return 'straight up';
  const az = deg(Math.atan2(d.y, d.x)), el = Math.round(deg(Math.asin(Math.max(-1, Math.min(1, d.z)))));
  const base = Math.round(az / 90) * 90;
  let s = { 0: 'left → right', 90: 'front → back', 180: 'right → left', '-180': 'right → left', '-90': 'back → front' }[base];
  const off = Math.round(wrap180(az - base));
  if (off) s += `, turned ${Math.abs(off)}° ${off > 0 ? 'left' : 'right'}`;
  if (el) s += `, tilted ${Math.abs(el)}° ${el < 0 ? 'down' : 'up'}`;
  return s;
}

// ------------------------------------------------------------------ density grid
export class Field {
  constructor(meta, bytes) {
    const g = meta.grid;
    [this.nx, this.ny, this.nz] = g.shape;
    this.dx = g.dx; this.o = g.origin; this.ks = g.k_sand;
    this.sandZ = g.sand_z; this.waterZ = g.water_z; this.tank = g.tank;
    const n = this.nx * this.ny * this.nz;
    if (bytes.length !== n) throw new Error(`grid size mismatch: ${bytes.length} values for ${n} nodes`);
    this.d = new Float32Array(n);
    for (let i = 0; i < n; i++) this.d[i] = bytes[i];
  }

  bytes() { const u = new Uint8Array(this.d.length); for (let i = 0; i < u.length; i++) u[i] = Math.round(this.d[i]); return u; }
  at(i, j, k) { return (i < 0 || j < 0 || k < 0 || i >= this.nx || j >= this.ny || k >= this.nz) ? 0 : this.d[i + this.nx * (j + this.ny * k)]; }

  // trilinear sample at a world point (0 outside the grid)
  sample(x, y, z) {
    const fx = (x - this.o[0]) / this.dx, fy = (y - this.o[1]) / this.dx, fz = (z - this.o[2]) / this.dx;
    const i = Math.floor(fx), j = Math.floor(fy), k = Math.floor(fz);
    const tx = fx - i, ty = fy - j, tz = fz - k;
    const c00 = this.at(i, j, k) * (1 - tx) + this.at(i + 1, j, k) * tx;
    const c10 = this.at(i, j + 1, k) * (1 - tx) + this.at(i + 1, j + 1, k) * tx;
    const c01 = this.at(i, j, k + 1) * (1 - tx) + this.at(i + 1, j, k + 1) * tx;
    const c11 = this.at(i, j + 1, k + 1) * (1 - tx) + this.at(i + 1, j + 1, k + 1) * tx;
    return (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz;
  }

  normal(p) {        // outward surface normal (density falls outward)
    const h = 0.5 * this.dx;
    const n = new THREE.Vector3(
      this.sample(p.x - h, p.y, p.z) - this.sample(p.x + h, p.y, p.z),
      this.sample(p.x, p.y - h, p.z) - this.sample(p.x, p.y + h, p.z),
      this.sample(p.x, p.y, p.z - h) - this.sample(p.x, p.y, p.z + h));
    return n.lengthSq() > 1e-12 ? n.normalize() : null;
  }

  // normal averaged over the brush size (steadier on rough rock than the point normal)
  smoothNormal(p, R) {
    const h = Math.max(this.dx, 0.5 * R);
    const n = new THREE.Vector3(
      this.sample(p.x - h, p.y, p.z) - this.sample(p.x + h, p.y, p.z),
      this.sample(p.x, p.y - h, p.z) - this.sample(p.x, p.y + h, p.z),
      this.sample(p.x, p.y, p.z - h) - this.sample(p.x, p.y, p.z + h));
    return n.lengthSq() > 1e-12 ? n.normalize() : null;
  }

  // first rock (or sand) surface along a ray -> {point, normal, sand} | null
  raycast(ro, rd) {
    const [L, W, H] = this.tank;
    const lo = [0, 0, this.sandZ], hi = [L, W, H];
    let t0 = 0, t1 = Infinity, exitAxis = -1, exitSide = 0;
    const o = [ro.x, ro.y, ro.z], d = [rd.x, rd.y, rd.z];
    for (let a = 0; a < 3; a++) {
      if (Math.abs(d[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) return null; continue; }
      let ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a], sa = 0, sb = 1;
      if (ta > tb) { [ta, tb] = [tb, ta]; [sa, sb] = [sb, sa]; }
      if (ta > t0) t0 = ta;
      if (tb < t1) { t1 = tb; exitAxis = a; exitSide = sb; }
    }
    if (t1 <= t0) return null;
    const step = 0.4 * this.dx, P = new THREE.Vector3();
    let prev = t0;
    for (let t = t0; t <= t1; t += step) {
      P.copy(rd).multiplyScalar(t).add(ro);
      if (this.sample(P.x, P.y, P.z) >= ISO) {
        let a = prev, b = t;
        for (let it = 0; it < 8; it++) {
          const m = 0.5 * (a + b);
          P.copy(rd).multiplyScalar(m).add(ro);
          if (this.sample(P.x, P.y, P.z) >= ISO) b = m; else a = m;
        }
        P.copy(rd).multiplyScalar(b).add(ro);
        return { point: P.clone(), normal: this.normal(P) || rd.clone().negate(), sand: false };
      }
      prev = t;
    }
    if (exitAxis === 2 && exitSide === 0) {        // left the box through the sand surface
      P.copy(rd).multiplyScalar(t1).add(ro);
      return { point: P.clone(), normal: new THREE.Vector3(0, 0, 1), sand: true };
    }
    return null;
  }

  rockVolume() {       // m^3 of rock nodes above the sand surface
    let c = 0;
    const start = this.ks * this.nx * this.ny;
    for (let i = start; i < this.d.length; i++) if (this.d[i] >= ISO) c++;
    return c * this.dx ** 3;
  }
}

// ------------------------------------------------------------------ surface nets mesher
const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
const E3 = (CH + 1) ** 3;
const S_POS = new Float32Array(E3 * 3), S_NOR = new Float32Array(E3 * 3), S_COL = new Float32Array(E3 * 3), S_VID = new Int32Array(E3);
// Display shading only (ambient occlusion): from each vertex, short rays in the 26 grid
// directions that face outward; the share that hits rock (or sand) within AO_MARGIN cells
// darkens the vertex. Open surfaces stay full brightness; the inside of a hole or cave,
// where most directions hit the opposite wall, goes dark, so openings read from any angle.
export const AO_MARGIN = 12;            // cells
const AO_DIRS = [];
for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
  if (!a && !b && !c) continue;
  const n = Math.hypot(a, b, c);
  AO_DIRS.push([a / n, b / n, c / n]);
}
const AO_STEPS = [1.5, 3, 4.5, 6.5, 9, 12];
const S_IDX = new Uint32Array(CH * CH * CH * 3 * 6);

// Mesh of chunk (cx, cy, cz): cells c in [cx*CH - 1, cx*CH - 1 + CH) per axis (cell c spans
// nodes c..c+1; nodes outside the grid are 0, so the surface closes at the grid edge).
export function meshChunk(F, cx, cy, cz) {
  const c0 = [cx * CH - 1, cy * CH - 1, cz * CH - 1];
  const n = [F.nx, F.ny, F.nz];
  const c1 = [0, 1, 2].map(a => Math.min(c0[a] + CH, n[a]));
  if (c1[0] <= c0[0] || c1[1] <= c0[1] || c1[2] <= c0[2]) return null;
  const e0 = [c0[0] - 1, c0[1] - 1, c0[2] - 1];
  const E = [c1[0] - e0[0], c1[1] - e0[1], c1[2] - e0[2]];
  const v = new Float32Array(8);
  const dx = F.dx, o = F.o;
  let nv = 0;
  // ---- one vertex per surface-crossing cell (extended range, for the quads below)
  for (let k = 0; k < E[2]; k++) for (let j = 0; j < E[1]; j++) for (let i = 0; i < E[0]; i++) {
    const ci = e0[0] + i, cj = e0[1] + j, ck = e0[2] + k;
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      v[c] = F.at(ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1));
      if (v[c] > ISO) mask |= 1 << c;
    }
    const li = i + E[0] * (j + E[1] * k);
    if (mask === 0 || mask === 255) { S_VID[li] = -1; continue; }
    let px = 0, py = 0, pz = 0, cnt = 0;
    for (const [a, b] of EDGES) {
      const ia = (mask >> a) & 1, ib = (mask >> b) & 1;
      if (ia === ib) continue;
      const t = (ISO - v[a]) / (v[b] - v[a]);
      px += (a & 1) + t * ((b & 1) - (a & 1));
      py += ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1));
      pz += ((a >> 2) & 1) + t * (((b >> 2) & 1) - ((a >> 2) & 1));
      cnt++;
    }
    px /= cnt; py /= cnt; pz /= cnt;
    // gradient of the trilinear interpolant at the vertex -> smooth normal (outward = -grad)
    const gx = ((v[1] - v[0]) * (1 - py) + (v[3] - v[2]) * py) * (1 - pz) + ((v[5] - v[4]) * (1 - py) + (v[7] - v[6]) * py) * pz;
    const gy = ((v[2] - v[0]) * (1 - px) + (v[3] - v[1]) * px) * (1 - pz) + ((v[6] - v[4]) * (1 - px) + (v[7] - v[5]) * px) * pz;
    const gz = ((v[4] - v[0]) * (1 - px) + (v[5] - v[1]) * px) * (1 - py) + ((v[6] - v[2]) * (1 - px) + (v[7] - v[3]) * px) * py;
    const gl = Math.hypot(gx, gy, gz) || 1;
    S_POS[3 * nv] = o[0] + (ci + px) * dx; S_POS[3 * nv + 1] = o[1] + (cj + py) * dx; S_POS[3 * nv + 2] = o[2] + (ck + pz) * dx;
    S_NOR[3 * nv] = -gx / gl; S_NOR[3 * nv + 1] = -gy / gl; S_NOR[3 * nv + 2] = -gz / gl;
    {
      const vx = ci + px, vy = cj + py, vz = ck + pz, nx_ = -gx / gl, ny_ = -gy / gl, nz_ = -gz / gl;
      let wsum = 0, hit = 0;
      for (let q = 0; q < AO_DIRS.length; q++) {
        const [ax, ay, az] = AO_DIRS[q], w = ax * nx_ + ay * ny_ + az * nz_;
        if (w < 0.25) continue;                 // only directions facing out of the surface
        wsum += w;
        for (let st = 0; st < AO_STEPS.length; st++) {
          const s = AO_STEPS[st], kk = Math.round(vz + az * s);
          if (kk < F.ks || F.at(Math.round(vx + ax * s), Math.round(vy + ay * s), kk) > ISO) { hit += w; break; }   // sand counts as solid
        }
      }
      const occ = wsum > 0 ? hit / wsum : 0;
      const sh = (1 - 0.75 * occ) ** 2.2;          // vertex colours are linear; 2.2 makes the darkening look as intended
      S_COL[3 * nv] = S_COL[3 * nv + 1] = S_COL[3 * nv + 2] = sh;
    }
    S_VID[li] = nv++;
  }
  if (!nv) return null;
  // ---- one quad per sign-changing grid edge owned by a chunk cell (its min-corner edges)
  let ni = 0;
  const vid = (i, j, k) => S_VID[(i - e0[0]) + E[0] * ((j - e0[1]) + E[1] * (k - e0[2]))];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) { const t = b; b = d; d = t; }
    S_IDX[ni++] = a; S_IDX[ni++] = b; S_IDX[ni++] = c;
    S_IDX[ni++] = a; S_IDX[ni++] = c; S_IDX[ni++] = d;
  };
  for (let k = c0[2]; k < c1[2]; k++) for (let j = c0[1]; j < c1[1]; j++) for (let i = c0[0]; i < c1[0]; i++) {
    const s = F.at(i, j, k) > ISO;
    if (s !== (F.at(i + 1, j, k) > ISO)) quad(vid(i, j - 1, k - 1), vid(i, j, k - 1), vid(i, j, k), vid(i, j - 1, k), !s);   // x-edge, normal +x when solid at i
    if (s !== (F.at(i, j + 1, k) > ISO)) quad(vid(i - 1, j, k - 1), vid(i - 1, j, k), vid(i, j, k), vid(i, j, k - 1), !s);   // y-edge
    if (s !== (F.at(i, j, k + 1) > ISO)) quad(vid(i - 1, j - 1, k), vid(i, j - 1, k), vid(i, j, k), vid(i - 1, j, k), !s);   // z-edge
  }
  if (!ni) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(S_POS.slice(0, 3 * nv), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(S_NOR.slice(0, 3 * nv), 3));
  g.setAttribute('color', new THREE.BufferAttribute(S_COL.slice(0, 3 * nv), 3));
  g.setIndex(new THREE.BufferAttribute(S_IDX.slice(0, ni), 1));
  g.computeBoundingSphere();
  return g;
}

// ------------------------------------------------------------------ the tool
export class Sculpt {
  constructor(app) {
    this.app = app;
    this.active = false;
    this.F = null; this.meta = null;
    this.tool = 'raise'; this.radius = 0.05; this.strength = 0.5;
    this.dirMode = 'view'; this.az = 90; this.el = 0;    // brush direction (see DIR_MODES)
    this.bore = false;                                     // one click bores straight through
    this._nrm = new THREE.Vector3(0, 0, 1);               // smoothed cursor normal (Smooth / Flatten cursor)
    this.dxChoice = 0.004;
    this.group = new THREE.Group();
    this.group.visible = false;
    app.view.editorRoot.add(this.group);
    this.mat = rockMaterial();
    this.mat.vertexColors = true;    // cave / hole shading from meshChunk
    this.chunks = new Map(); this.dirty = new Set();
    this.undo = []; this.redo = []; this.stroke = null;
    this.modified = false; this.lastReport = null;
    this.mouse = null; this.mods = { ctrl: false, shift: false }; this.down = false;
    this.cursor = this._makeCursor();
    this.group.add(this.cursor);
    this._makeGuides();
    this.label = null;
    if (typeof document !== 'undefined') {
      this.label = document.createElement('div');
      this.label.className = 'sculpt-label';
      this.label.hidden = true;
      document.body.append(this.label);
    }
    this._loop = (ts) => this._tick(ts);
    const dom = app.view.renderer.domElement;
    this._h = {
      down: (e) => this._onDown(e), move: (e) => this._onMove(e), up: (e) => this._onUp(e),
      wheel: (e) => this._onWheel(e), leave: () => { this.mouse = null; },
      blur: () => { this.mods = { ctrl: false, shift: false }; this._endStroke(); },
      unload: (e) => { if (this.active && this.modified) { e.preventDefault(); e.returnValue = ''; } },
    };
    this.dom = dom;
    window.addEventListener('beforeunload', this._h.unload);
  }

  // ---------------------------------------------------------------- enter / leave
  async enter(opts = {}) {
    const app = this.app;
    if (this.active && !opts.restart) return;
    if (app.designer.stale() && app.ed.designer.elements.length) {
      if (!confirm('The rock elements changed since the last build. Build the rockscape first, so you sculpt what you see?')) return;
      await app.designer.build();
    }
    if (app.dirty) await app.save();
    let r;
    try {
      r = await busy('preparing the rock for sculpting', () => api.binary(`/api/projects/${app.pid}/sculpt/start`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dx: opts.dx ?? null, fresh: !!opts.fresh }) }));
    } catch (e) { status(e.message, 'err'); return; }
    this._clearMeshes();
    this.meta = r.meta;
    this.F = new Field(r.meta, new Uint8Array(r.buf));
    this.dxChoice = this.F.dx;
    this.undo = []; this.redo = []; this.modified = false; this.lastReport = null;
    this.volume = this.F.rockVolume();
    this.radius = Math.min(Math.max(this.radius, 2 * this.F.dx), MAX_RADIUS);
    this.nc = [Math.ceil((this.F.nx + 1) / CH), Math.ceil((this.F.ny + 1) / CH), Math.ceil((this.F.nz + 1) / CH)];
    for (let cz = 0; cz < this.nc[2]; cz++) for (let cy = 0; cy < this.nc[1]; cy++) for (let cx = 0; cx < this.nc[0]; cx++) this.dirty.add(this._key(cx, cy, cz));
    await busy('meshing', async () => { while (this.dirty.size) { this._remesh(40); await new Promise(res => setTimeout(res, 0)); } });
    if (!this.active) this._attach();
    this.active = true;
    status(`sculpting: ${r.meta.source}; grid ${r.meta.grid.shape.join(' × ')} at ${num(this.F.dx * 1000, 2)} mm` + (r.meta.draft ? ' (unapplied draft restored)' : ''), 'ok');
    app.renderPanel();
    app.view.render();
  }

  _attach() {
    const v = this.app.view, dom = this.dom;
    this.app.designer.group.visible = false;
    this.group.visible = true;
    v.attach(null);
    document.getElementById('gizmoModes').hidden = true;
    this._orbitSave = { buttons: { ...v.orbit.mouseButtons }, zoom: v.orbit.enableZoom };
    v.orbit.mouseButtons = { LEFT: -1, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
    v.orbit.enableZoom = false;
    dom.addEventListener('pointerdown', this._h.down);
    dom.addEventListener('pointermove', this._h.move);
    dom.addEventListener('pointerup', this._h.up);
    dom.addEventListener('pointercancel', this._h.up);
    dom.addEventListener('pointerleave', this._h.leave);
    dom.addEventListener('wheel', this._h.wheel, { passive: false });
    window.addEventListener('blur', this._h.blur);
    this._running = true;
    requestAnimationFrame(this._loop);
  }

  async leave(force = false) {
    if (!this.active) return true;
    if (!force && this.modified) {
      const keep = confirm('Keep the sculpt as a draft before leaving? (OK = save draft, Cancel = discard the changes since the last save/apply)');
      if (keep) { try { await this.saveDraft(); } catch (e) { status(e.message, 'err'); return false; } }
    }
    const v = this.app.view, dom = this.dom;
    this._running = false;
    this.active = false; this.down = false; this.stroke = null;
    dom.removeEventListener('pointerdown', this._h.down);
    dom.removeEventListener('pointermove', this._h.move);
    dom.removeEventListener('pointerup', this._h.up);
    dom.removeEventListener('pointercancel', this._h.up);
    dom.removeEventListener('pointerleave', this._h.leave);
    dom.removeEventListener('wheel', this._h.wheel);
    window.removeEventListener('blur', this._h.blur);
    if (this._orbitSave) { v.orbit.mouseButtons = this._orbitSave.buttons; v.orbit.enableZoom = this._orbitSave.zoom; }
    if (this.label) this.label.hidden = true;
    dom.style.cursor = '';
    this._clearMeshes();
    this.F = null; this.undo = []; this.redo = [];
    this.group.visible = false;
    this.app.designer.group.visible = true;
    document.getElementById('gizmoModes').hidden = this.app.tab !== 'rocks';
    v.render();
    if (this.app.tab === 'rocks') this.app.renderPanel();
    return true;
  }

  _clearMeshes() {
    for (const m of this.chunks.values()) { this.group.remove(m); m.geometry.dispose(); }
    this.chunks.clear(); this.dirty.clear();
  }

  // ---------------------------------------------------------------- server
  base() { return `/api/projects/${this.app.pid}/sculpt`; }

  async saveDraft() {
    const F = this.F;
    const r = await busy('saving the sculpt draft', async () => {
      const res = await fetch(`${this.base()}/draft?dx=${F.dx}&base_sig=${encodeURIComponent(this.meta.base_sig || '')}`,
        { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: F.bytes() });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      return res.json();
    });
    this.modified = false;
    status('sculpt draft saved (it is restored the next time you sculpt this rockscape)', 'ok');
    this.app.renderPanel();
    return r;
  }

  async applyToProject() {
    const app = this.app, F = this.F;
    if (app.dirty) await app.save();
    let r;
    try {
      r = await busy('turning the sculpt into the project rockscape', async () => {
        const res = await fetch(`${this.base()}/apply?dx=${F.dx}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: F.bytes() });
        const j = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(j.detail || res.statusText);
        return j;
      });
    } catch (e) { status(e.message, 'err'); return; }
    this.meta.base_sig = r.base_sig;
    this.modified = false;
    this.lastReport = r.report;
    app.loadPayload(r.project, true);        // designer: one 'sculpted rock' mesh element (hidden while sculpting)
    app.designer.group.visible = false;
    status(`rockscape updated from the sculpt: ${num(r.report.volume_voxel_m3 * 1000, 3)} L of rock above the sand, ` +
      `${r.report.triangles.toLocaleString()} triangles, ${r.report.watertight ? 'watertight' : 'NOT watertight'}`, r.report.watertight ? 'ok' : 'err');
    app.renderPanel();
  }

  // ---------------------------------------------------------------- input
  _ray(e) {
    const v = this.app.view;
    v.raycaster.setFromCamera(v.ndc(e), v.camera);
    return v.raycaster.ray;
  }

  _onDown(e) {
    this.mods = { ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey };
    if (e.button !== 0) return;
    if (!this._overRock) return;        // left-drag on empty space orbits the view (see _tick)
    this.mouse = { clientX: e.clientX, clientY: e.clientY };
    this.down = true;
    try { this.dom.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
    this._startStroke();
  }

  _onMove(e) {
    this.mouse = { clientX: e.clientX, clientY: e.clientY };
    this.mods = { ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey };
  }

  _onUp(e) {
    if (e.button !== 0 && e.type !== 'pointercancel') return;
    this.down = false;
    try { this.dom.releasePointerCapture(e.pointerId); } catch (err) { /* not captured */ }
    this._endStroke();
  }

  _onWheel(e) {
    e.preventDefault();
    const dy = e.deltaY || e.deltaX;               // Shift+wheel scrolls horizontally on some systems
    if (!dy) return;
    const k = Math.exp(-Math.sign(dy) * Math.min(Math.abs(dy), 200) * 0.0015);
    if (e.ctrlKey || e.metaKey) {                  // camera zoom
      const v = this.app.view, off = v.camera.position.clone().sub(v.orbit.target);
      off.multiplyScalar(1 / k);
      v.camera.position.copy(v.orbit.target).add(off);
      v.orbit.update();
    } else if (e.shiftKey || e.altKey) {
      this.setStrength(this.strength * k);
    } else {
      this.setRadius(this.radius * k);
    }
    this.app.view.render();
  }

  setRadius(r) {
    const dx = this.F ? this.F.dx : 0.004;
    this.radius = Math.min(Math.max(r, 1.5 * dx), MAX_RADIUS);
    this._syncPanelValues();
  }

  setStrength(s) { this.strength = Math.min(Math.max(s, 0.05), 1); this._syncPanelValues(); }

  setTool(t) {
    this.tool = t;
    this.cursor.children.forEach(c => { if (!c.userData.fixedColor) c.material.color.setHex(TOOLS[t].color); });
    if (this.app.tab === 'rocks') this.app.renderPanel();
  }

  // ---------------------------------------------------------------- brush direction
  // Direction the brush travels into the rock (unit vector), or null = follow the surface.
  lockedDir() {
    if (this.dirMode === 'fixed') return dirFromAngles(this.az, this.el);
    const v = this.app.view;
    return snapAxis(v.orbit.target.clone().sub(v.camera.position));
  }

  // Per tool: Tunnel / Extrude follow the brush direction; Flatten levels (horizontal plane)
  // unless a fixed direction is set, in which case it cuts a flat face square to that
  // direction; Raise / Lower work along your line of sight and Smooth has no direction.
  toolDir(tool) {
    if (tool === 'flatten') return this.dirMode === 'fixed' ? dirFromAngles(this.az, this.el) : new THREE.Vector3(0, 0, -1);
    if (tool === 'drill' || tool === 'extrude') return this.lockedDir();
    return null;
  }

  setDirMode(m) {
    if (m === 'fixed' && this.dirMode !== 'fixed') { this._toFixed(); return; }
    this.dirMode = m;
    this._dirChanged();
  }

  _toFixed() {       // switch to a fixed direction, starting from the current one
    if (this.dirMode !== 'fixed') {
      const d = this.lockedDir();
      this.el = Math.max(-90, Math.min(90, round15(deg(Math.asin(Math.max(-1, Math.min(1, d.z)))))));
      if (Math.abs(this.el) < 90) this.az = round15(deg(Math.atan2(d.y, d.x)));
      this.dirMode = 'fixed';
    }
    this._dirChanged();
  }

  setAxis(az, el) { this.dirMode = 'fixed'; this.az = az; this.el = el; this._dirChanged(); }
  turn(dAz) { this._toFixed(); this.az = wrap180(this.az + dAz); this._dirChanged(); }
  tilt(dEl) { this._toFixed(); this.el = Math.max(-90, Math.min(90, this.el + dEl)); this._dirChanged(); }
  setBore(b) { this.bore = b; this._dirChanged(); }

  _dirChanged() {
    if (this.app.tab === 'rocks' && this.active) this.app.renderPanel();
    if (this.dirMode !== 'view') status(`brush direction: ${dirLabel(this.lockedDir())}`);
  }

  effectiveTool() {
    if (this.mods.shift) return 'smooth';
    if (this.mods.ctrl) {
      if (this.tool === 'raise') return 'lower';
      if (this.tool === 'lower') return 'raise';
      if (this.tool === 'drill') return 'extrude';
    }
    return this.tool;
  }

  onKey(e) {         // returns true when handled
    if (!this.active) return false;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z' && !e.shiftKey) { e.preventDefault(); this.doUndo(); return true; }
    if ((e.ctrlKey || e.metaKey) && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); this.doRedo(); return true; }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    for (const [t, d] of Object.entries(TOOLS)) if (k === d.key) { this.setTool(t); return true; }
    if (k === '[') { this.setRadius(this.radius / 1.15); return true; }
    if (k === ']') { this.setRadius(this.radius * 1.15); return true; }
    if (k === 'q') { this.turn(STEP_DEG); return true; }
    if (k === 'e') { this.turn(-STEP_DEG); return true; }
    if (k === 'w') { this.tilt(STEP_DEG); return true; }
    if (k === 's') { this.tilt(-STEP_DEG); return true; }
    if (k === 'v') { this.setDirMode('view'); return true; }
    if (k === 'b') { this.setBore(!this.bore); return true; }
    if (k === 'delete' || k === 'backspace' || k === 'r') return true;   // designer shortcuts off while sculpting
    return false;
  }

  // ---------------------------------------------------------------- frame loop
  _tick(ts) {
    if (!this._running) return;
    const dt = this._last ? Math.min((ts - this._last) / 1000, 0.1) : 0.016;     // slow frames still sculpt at the stated rate (up to 10 fps)
    this._last = ts;
    let changed = false;
    const v = this.app.view;
    const ray = this.mouse && this.F ? this._ray(this.mouse) : null;
    const hit = ray ? this.F.raycast(ray.origin, ray.direction) : null;
    if (!this.down) {        // left button: sculpt over rock / sand, orbit over empty space
      this._overRock = !!hit;
      const want = hit ? -1 : THREE.MOUSE.ROTATE;
      const cur = hit ? toolCursor(this.effectiveTool()) : 'grab';       // tool icon over the rock, hand over empty space
      if (this.dom.style.cursor !== cur) this.dom.style.cursor = cur;
      if (v.orbit.mouseButtons.LEFT !== want) v.orbit.mouseButtons.LEFT = want;
    }
    if (this.down && this.stroke && ray) changed = this._applyStroke(hit, dt, ray);
    if (this.down) {                 // keep the icon of the tool in use (a modifier key may switch it mid-stroke)
      const cur = toolCursor(this.effectiveTool());
      if (this.dom.style.cursor !== cur) this.dom.style.cursor = cur;
    }
    if (this._updateCursor(hit, ray)) changed = true;
    if (this.dirMode === 'view') {   // the view-snapped direction follows the camera
      const key = this.lockedDir().toArray().join();
      if (key !== this._dirKey) { this._dirKey = key; this._syncPanelValues(); }
    }
    if (this.dirty.size) { this._remesh(12); changed = true; }
    if (changed) v.render();
    requestAnimationFrame(this._loop);
  }

  // place the cursor, direction arrow, drill-path preview and the label; true = redraw
  _updateCursor(hit, ray = null) {
    const tool = this.effectiveTool(), s = this.stroke, R = this.radius, G = this.guides;
    const along = tool === 'drill' || tool === 'extrude';
    const shift = tool === 'raise' || tool === 'lower';
    const drilling = !!(s && s.anchor && s.d && along && s.tool === tool);
    const show = !!hit || drilling;
    const wasShown = this.cursor.visible;
    this.cursor.visible = show;
    G.path.visible = G.exit.visible = G.tip.visible = false;
    if (!show) { if (this.label) this.label.hidden = true; return wasShown; }
    const d = drilling ? s.d : this.toolDir(tool);
    const pos = drilling ? s.anchor : hit.point;
    let out;                         // the cursor ring faces this way
    if (d) out = d.clone().negate();
    else if (shift && ray) out = ray.direction.clone().negate();      // Raise / Lower: a disc square to your view
    else {
      const n = hit.sand ? new THREE.Vector3(0, 0, 1) : (this.F.smoothNormal(hit.point, R) || hit.normal);
      this._nrm.lerp(n, 0.35).normalize();
      out = this._nrm;
    }
    const col = TOOLS[tool === 'extrude' ? 'drill' : tool].color;
    this.cursor.position.copy(pos);
    this.cursor.scale.setScalar(R);
    this.cursor.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), out);
    this.cursor.children.forEach(c => { if (!c.userData.fixedColor) c.material.color.setHex(col); });
    const C = this.cursorParts;
    C.sph.visible = !(d && along) && !shift;
    C.arrow.visible = !!(d && along);
    C.arrow.rotation.set(tool === 'extrude' ? Math.PI : 0, 0, 0);    // tunnel: into the rock; extrude: out toward you
    let text = LABEL[tool];
    const U = this.app.units, L = (x) => U.fmt(x, 'length', 3);
    if (shift) {
      text = `${LABEL[tool]} · ${L(SHIFT_SPEED * this.strength)}/s ${tool === 'raise' ? 'toward you' : 'away from you'} at the centre`;
    } else if (d && tool === 'drill') {
      const path = drilling ? s.path : this._path(pos, d, 'push', this.F);
      const end = path.exit !== null ? path.exit : path.tmax;
      this._setLine(pos, pos.clone().addScaledVector(d, end));
      if (path.exit !== null) {
        G.exit.visible = true;
        G.exit.position.copy(pos).addScaledVector(d, path.exit);
        G.exit.quaternion.copy(this.cursor.quaternion);
        G.exit.scale.setScalar(R);
      }
      if (drilling) {
        G.tip.visible = true;
        G.tip.position.copy(pos).addScaledVector(d, s.depth);
        G.tip.scale.setScalar(Math.max(R * 0.18, 1.5 * this.F.dx));
        text = s.done ? `through ✓ (${L(path.exit)}): release` : `${L(s.depth)} deep` + (path.exit !== null ? ` · through at ${L(path.exit)}` : ` · ${path.wall} at ${L(path.tmax)}`);
      } else {
        text = `Tunnel ${dirLabel(d)}: ` + (path.exit !== null ? `through at ${L(path.exit)}` : !path.rock ? 'no rock on this line'
          : `no way through before the ${path.wall} (${L(path.tmax)})`) + (this.bore && path.exit !== null ? ' · click bores through' : '');
      }
    } else if (d && tool === 'extrude') {
      if (drilling) {
        G.tip.visible = true;
        G.tip.position.copy(pos).addScaledVector(d, -s.depth);
        G.tip.scale.setScalar(Math.max(R * 0.18, 1.5 * this.F.dx));
        this._setLine(pos, G.tip.position);
        text = `pulled out ${L(s.depth)}` + (s.depth >= s.path.stop - 1e-9 ? ` (stops at the ${s.path.wall})` : '');
      } else text = `Extrude ${dirLabel(d.clone().negate())}`;
    } else if (tool === 'flatten') {
      text = this.dirMode === 'fixed' ? `Flatten: flat face, ${dirLabel(d)}` : 'Flatten: level';
    }
    if (this.label && this.mouse) {
      this.label.hidden = false;
      this.label.textContent = text;
      this.label.style.left = `${this.mouse.clientX + 18}px`;
      this.label.style.top = `${this.mouse.clientY + 14}px`;
    }
    return true;
  }

  _setLine(a, b) {
    const g = this.guides.path, p = g.geometry.attributes.position;
    p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, b.x, b.y, b.z);
    p.needsUpdate = true;
    g.geometry.computeBoundingSphere();
    g.computeLineDistances();
    g.visible = true;
  }

  _makeCursor() {
    const g = new THREE.Group();
    const sph = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color: TOOLS.raise.color, transparent: true, opacity: 0.16, depthWrite: false }));
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045, 6, 64), new THREE.MeshBasicMaterial({ color: TOOLS.raise.color, depthTest: false }));
    ring.renderOrder = 21;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), new THREE.MeshBasicMaterial({ color: TOOLS.raise.color, depthTest: false }));
    dot.renderOrder = 21;
    const halo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.09, 6, 64), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, depthTest: false }));
    halo.renderOrder = 20;
    halo.userData.fixedColor = true;
    // arrow along local -z (into the rock), drawn on top so it shows through the rock
    const arrow = new THREE.Group();
    const amat = new THREE.MeshBasicMaterial({ color: TOOLS.raise.color, depthTest: false });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.1, 8), amat);
    shaft.rotation.x = Math.PI / 2; shaft.position.z = 0.15;
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.45, 12), amat);
    head.rotation.x = -Math.PI / 2; head.position.z = -0.55;
    shaft.renderOrder = head.renderOrder = 22;
    arrow.add(shaft, head);
    arrow.material = amat;           // so the recolour loop over the cursor's children reaches it
    g.add(sph, halo, ring, dot, arrow);
    g.visible = false;
    this.cursorParts = { sph, arrow };
    return g;
  }

  _makeGuides() {
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const path = new THREE.Line(pg, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.006, gapSize: 0.004, depthTest: false, transparent: true, opacity: 0.9 }));
    path.renderOrder = 23;
    const exit = new THREE.Mesh(new THREE.TorusGeometry(1, 0.06, 6, 48), new THREE.MeshBasicMaterial({ color: 0xffe066, depthTest: false }));
    exit.renderOrder = 23;
    const tip = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
    tip.renderOrder = 23;
    for (const o of [path, exit, tip]) { o.visible = false; o.frustumCulled = false; this.group.add(o); }
    this.guides = { path, exit, tip };
  }

  // ---------------------------------------------------------------- strokes
  _startStroke() {
    if (!this.F) return;
    this.stroke = { blocks: new Map(), last: null, plane: null, bytes: 0, tool: null, d: null, anchor: null, s: 0, depth: 0, path: null, done: false };
  }

  _endStroke() {
    const s = this.stroke;
    this.stroke = null;
    if (!s || !s.blocks.size) return;
    this.undo.push(s.blocks);
    this.redo = [];
    let mem = this.undo.reduce((a, m) => a + m.size * B * B * B * 4, 0);
    while (mem > UNDO_BYTES && this.undo.length > 1) { mem -= this.undo[0].size * B * B * B * 4; this.undo.shift(); }
    this.modified = true;
    this.volume = this.F.rockVolume();
    this._syncPanelValues();
  }

  // One frame of a stroke. hit: current surface under the cursor (may be null);
  // ray: the cursor's camera ray (used to find the pre-stroke surface for locked strokes).
  _applyStroke(hit, dt, ray = null) {
    const s = this.stroke;
    const tool = this.effectiveTool();
    if (s.tool === null) Object.assign(s, { tool, d: this.toolDir(tool) });         // first frame
    else if (s.tool !== tool) {      // a modifier key switched the tool mid-stroke: start its shape afresh
      Object.assign(s, { tool, d: this.toolDir(tool), anchor: null, s: 0, depth: 0, path: null, done: false, plane: null, last: null });
    }
    if (tool === 'drill' || tool === 'extrude') return s.d ? this._drill(tool === 'drill' ? 'push' : 'pull', hit, ray, dt) : false;
    if (!hit) return false;
    if (s.plane === null) s.plane = { p: hit.point.clone(), n: s.d ? s.d.clone().negate() : new THREE.Vector3(0, 0, 1) };
    // airbrush: the frame's amount is spread over points spaced R/3 along the path
    const pts = [];
    if (s.last && s.last.distanceTo(hit.point) > this.radius / 3) {
      const n = Math.min(Math.ceil(s.last.distanceTo(hit.point) / (this.radius / 3)), 30);
      for (let i = 1; i <= n; i++) pts.push(s.last.clone().lerp(hit.point, i / n));
    } else pts.push(hit.point);
    s.last = hit.point.clone();
    const share = 1 / pts.length;
    if (tool === 'raise' || tool === 'lower') {
      // the column reaches as deep as the stroke has moved the surface (plus the radius),
      // so a dome / hollow keeps growing as a whole while you hold still
      const amt = SHIFT_SPEED * this.strength * dt;
      s.moved = Math.min((s.moved || 0) + amt, 3 * this.radius);
      const rd = ray ? ray.direction.clone().normalize() : hit.normal.clone().negate();
      for (const p of pts) this.shift(p, rd, this.radius, amt * share, tool === 'raise', this.radius + s.moved);
      return true;
    }
    for (const p of pts) {
      const a = tool === 'smooth' ? Math.min(1, SMOOTH_RATE * this.strength * dt * share)
        : Math.min(1, FLATTEN_RATE * this.strength * dt * share);
      this.brush(p, this.radius, tool, a, s.plane);
    }
    return true;
  }

  // Raise / Lower dab (D-843): inside a round column of radius R around the line of sight
  // rd through c (|along the line| <= half), every rock surface that faces the viewer moves
  // along rd by amt * w (toward the viewer for Raise), w = (1 - r^2/R^2)^2 with r the
  // distance from the line. Semi-Lagrangian: the new density at x is the old one at
  // x + amt*w*rd (Raise) or x - amt*w*rd (Lower), taken as a max (Raise) / min (Lower) so
  // surfaces facing away never change. For Raise the sand counts as solid, so raising on
  // the sand grows rock up out of it; Raise adds nothing within WALL_GAP of the glass or
  // the water surface (the flow model needs water there).
  shift(c, rd, R, amt, raise, half) {
    const F = this.F, dx = F.dx, o = F.o, nx = F.nx, ny = F.ny, d = F.d, L = F.tank[0], W = F.tank[1];
    const ax = c.x - rd.x * half, ay = c.y - rd.y * half, az = c.z - rd.z * half;
    const bx = c.x + rd.x * half, by = c.y + rd.y * half, bz = c.z + rd.z * half;
    const i0 = Math.max(0, Math.ceil((Math.min(ax, bx) - R - o[0]) / dx)), i1 = Math.min(nx - 1, Math.floor((Math.max(ax, bx) + R - o[0]) / dx));
    const j0 = Math.max(0, Math.ceil((Math.min(ay, by) - R - o[1]) / dx)), j1 = Math.min(ny - 1, Math.floor((Math.max(ay, by) + R - o[1]) / dx));
    const k0 = Math.max(F.ks, Math.ceil((Math.min(az, bz) - R - o[2]) / dx)), k1 = Math.min(F.nz - 1, Math.floor((Math.max(az, bz) + R - o[2]) / dx));
    if (i0 > i1 || j0 > j1 || k0 > k1 || !(amt > 0)) return;
    // read from a copy (one extra cell of margin per cell of travel) so the result does not depend on loop order
    const m = Math.ceil(amt / dx) + 1;
    const ci0 = i0 - m, cj0 = j0 - m, ck0 = k0 - m, cw = i1 - i0 + 1 + 2 * m, ch = j1 - j0 + 1 + 2 * m, cd = k1 - k0 + 1 + 2 * m;
    const src = new Float32Array(cw * ch * cd);
    let q = 0;
    for (let k = ck0; k < ck0 + cd; k++) for (let j = cj0; j < cj0 + ch; j++) for (let i = ci0; i < ci0 + cw; i++) {
      src[q++] = raise && k < F.ks ? 255 : F.at(i, j, k);           // Raise: the sand is solid
    }
    const at = (i, j, k) => {
      const a = i - ci0, b = j - cj0, e = k - ck0;
      if (a < 0 || b < 0 || e < 0 || a >= cw || b >= ch || e >= cd) return raise && k < F.ks ? 255 : F.at(i, j, k);
      return src[a + cw * (b + ch * e)];
    };
    const samp = (x, y, z) => {                      // trilinear, on the copy
      const fx = (x - o[0]) / dx, fy = (y - o[1]) / dx, fz = (z - o[2]) / dx;
      const i = Math.floor(fx), j = Math.floor(fy), k = Math.floor(fz), tx = fx - i, ty = fy - j, tz = fz - k;
      const c00 = at(i, j, k) * (1 - tx) + at(i + 1, j, k) * tx, c10 = at(i, j + 1, k) * (1 - tx) + at(i + 1, j + 1, k) * tx;
      const c01 = at(i, j, k + 1) * (1 - tx) + at(i + 1, j, k + 1) * tx, c11 = at(i, j + 1, k + 1) * (1 - tx) + at(i + 1, j + 1, k + 1) * tx;
      return (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz;
    };
    const kb = k0 === F.ks ? 0 : k0;
    this._saveBlocks(i0, i1, j0, j1, kb, k1);
    const R2 = R * R, sgn = raise ? 1 : -1, zTop = F.waterZ - WALL_GAP;
    for (let k = k0; k <= k1; k++) {
      const z = o[2] + k * dx;
      for (let j = j0; j <= j1; j++) {
        const y = o[1] + j * dx;
        for (let i = i0; i <= i1; i++) {
          const x = o[0] + i * dx;
          const px = x - c.x, py = y - c.y, pz = z - c.z;
          const t = px * rd.x + py * rd.y + pz * rd.z;
          if (t > half || t < -half) continue;
          const qx = px - t * rd.x, qy = py - t * rd.y, qz = pz - t * rd.z, r2 = qx * qx + qy * qy + qz * qz;
          if (r2 >= R2) continue;
          const s1 = 1 - r2 / R2, step = sgn * amt * s1 * s1;
          const v = samp(x + step * rd.x, y + step * rd.y, z + step * rd.z);
          const idx = i + nx * (j + ny * k);
          if (raise) {
            if (v > d[idx] && x > WALL_GAP && x < L - WALL_GAP && y > WALL_GAP && y < W - WALL_GAP && z < zTop) d[idx] = v;
          } else if (v < d[idx]) d[idx] = v;
        }
      }
    }
    if (k0 === F.ks) this._bury(i0, i1, j0, j1);
    this._markDirty(i0, i1, j0, j1, kb, k1);
  }

  // Locked Push / Pull: a straight round hole (or arm of rock) along the stroke direction.
  // The anchor is where the cursor ray meets the rock as it was before this stroke, so
  // holding still drills one straight line and dragging routes a groove (or ridge) that
  // follows the original surface. Depth grows with hold time; Push stops once it breaks
  // through the far side, both stop at the glass, sand or water surface.
  _drill(tool, hit, ray, dt) {
    const s = this.stroke, R = this.radius, dx = this.F.dx;
    const u = tool === 'pull' ? s.d.clone().negate() : s.d;      // the way the tip travels
    const Fv = this._orig();
    let a = null;
    if (ray) { const h = Fv.raycast(ray.origin, ray.direction); if (h) a = h.point; }
    else if (hit) a = hit.point.clone();
    if (!a && !s.anchor) return false;
    s.s += DRILL_SPEED * this.strength * dt;
    const capsule = (p, depth) => this.carve(p.clone().addScaledVector(u, -R), p.clone().addScaledVector(u, depth - R), R, tool === 'pull');
    if (!s.anchor) {                                       // stroke start
      s.anchor = a.clone();
      s.path = this._path(a, u, tool, Fv);
      if (this.bore && tool === 'push' && s.path.exit !== null) s.s = s.path.stop;
      s.depth = Math.min(s.s, s.path.stop);
      capsule(s.anchor, s.depth);
    } else if (!a || a.distanceTo(s.anchor) <= 0.25 * dx) { // holding still: extend the hole
      s.s = Math.min(s.s, s.path.stop);
      if (s.s > s.depth) {
        this.carve(s.anchor.clone().addScaledVector(u, s.depth - R), s.anchor.clone().addScaledVector(u, s.s - R), R, tool === 'pull');
        s.depth = s.s;
      }
    } else {                                               // dragging: groove / ridge at the stroke depth
      const n = Math.min(Math.ceil(a.distanceTo(s.anchor) / (R / 3)), 30);
      for (let i = 1; i <= n; i++) {
        const p = s.anchor.clone().lerp(a, i / n);
        const path = this._path(p, u, tool, Fv), depth = Math.min(s.s, path.stop);
        capsule(p, depth);
        if (i === n) { s.path = path; s.depth = depth; }
      }
      s.anchor = a.clone();
    }
    s.done = tool === 'push' && s.path.exit !== null && s.depth >= s.path.stop - 1e-9;
    return true;
  }

  // The field as it was before the current stroke (the stroke's undo copies, else live).
  _orig() {
    const s = this.stroke, F = this.F;
    if (!s) return F;
    if (s.origView) return s.origView;
    const o = Object.create(F), nx = F.nx, ny = F.ny, nz = F.nz, blocks = s.blocks;
    o.at = (i, j, k) => {
      if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return 0;
      const bi = Math.floor(i / B), bj = Math.floor(j / B), bk = Math.floor(k / B);
      const blk = blocks.get(this._bkey(bi, bj, bk));
      if (!blk) return F.d[i + nx * (j + ny * k)];
      const w = Math.min(B, nx - bi * B), h = Math.min(B, ny - bj * B);
      return blk.data[(i - bi * B) + w * ((j - bj * B) + h * (k - bk * B))];
    };
    s.origView = o;
    return o;
  }

  // Along the line a + t·u: where a Push breaks through (start of the first water stretch
  // after rock that is at least max(R/2, 2 cells) long, or that reaches the boundary), and
  // how far the line runs before the glass / sand / water surface.
  // -> {exit: t | null, stop: max useful tip depth, tmax, wall}
  _path(a, u, tool, Fv) {
    const F = this.F, R = this.radius, [L, W, H] = F.tank;
    const lo = [0, 0, F.sandZ], hi = [L, W, H], p = [a.x, a.y, a.z], d = [u.x, u.y, u.z];
    let tmax = Infinity, wall = 'glass';
    for (let ax = 0; ax < 3; ax++) {
      if (Math.abs(d[ax]) < 1e-9) continue;
      const t = Math.max(0, ((d[ax] > 0 ? hi[ax] : lo[ax]) - p[ax]) / d[ax]);
      if (t < tmax) { tmax = t; wall = ax < 2 ? 'glass' : (d[ax] > 0 ? 'top' : 'sand'); }
    }
    // Pull stops 10 mm (about 2 Standard CFD cells) short of the glass / water surface, so
    // the flow model still sees a gap there (rock touching the glass blocks flow behind it).
    if (tool !== 'push') return { exit: null, stop: Math.max(0, tmax - 0.01), tmax, wall };
    const step = 0.5 * F.dx, gap = Math.max(0.5 * R, 2 * F.dx), P = new THREE.Vector3();
    let seen = false, run = null;
    for (let t = 0; t <= tmax; t += step) {
      P.copy(u).multiplyScalar(t).add(a);
      if (Fv.sample(P.x, P.y, P.z) >= ISO) { seen = true; run = null; }
      else if (seen) {
        if (run === null) run = t;
        if (t - run >= gap) break;
      }
    }
    if (seen && run !== null) return { exit: run, stop: Math.min(run + R, tmax), tmax, wall, rock: true };
    return { exit: null, stop: tmax, tmax, wall, rock: seen };
  }

  // Cut (or fill) a round-ended cylinder of radius R from A to Bp, with a soft edge.
  // Takes the min (cut) / max (fill) with the existing density, so going over the same
  // place again changes nothing: depth comes only from the shape, not from repetition.
  carve(A, Bp, R, fill = false) {
    const F = this.F, dx = F.dx, o = F.o, nx = F.nx, ny = F.ny, d = F.d;
    const band = Math.max(1.5 * dx, 0.2 * R), Rb = R + band, R0 = R - band, Rb2 = Rb * Rb;
    const i0 = Math.max(0, Math.ceil((Math.min(A.x, Bp.x) - Rb - o[0]) / dx)), i1 = Math.min(nx - 1, Math.floor((Math.max(A.x, Bp.x) + Rb - o[0]) / dx));
    const j0 = Math.max(0, Math.ceil((Math.min(A.y, Bp.y) - Rb - o[1]) / dx)), j1 = Math.min(ny - 1, Math.floor((Math.max(A.y, Bp.y) + Rb - o[1]) / dx));
    const k0 = Math.max(F.ks, Math.ceil((Math.min(A.z, Bp.z) - Rb - o[2]) / dx)), k1 = Math.min(F.nz - 1, Math.floor((Math.max(A.z, Bp.z) + Rb - o[2]) / dx));
    if (i0 > i1 || j0 > j1 || k0 > k1) return;
    const kb = k0 === F.ks ? 0 : k0;
    this._saveBlocks(i0, i1, j0, j1, kb, k1);
    const abx = Bp.x - A.x, aby = Bp.y - A.y, abz = Bp.z - A.z, L2 = abx * abx + aby * aby + abz * abz;
    const sy = nx, sz = nx * ny;
    for (let k = k0; k <= k1; k++) {
      const pz = o[2] + k * dx - A.z;
      for (let j = j0; j <= j1; j++) {
        const py = o[1] + j * dx - A.y;
        for (let i = i0; i <= i1; i++) {
          const px = o[0] + i * dx - A.x;
          let t = L2 > 0 ? (px * abx + py * aby + pz * abz) / L2 : 0;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const qx = px - t * abx, qy = py - t * aby, qz = pz - t * abz, q2 = qx * qx + qy * qy + qz * qz;
          if (q2 >= Rb2) continue;
          const x = (Math.sqrt(q2) - R0) / (2 * band);
          const sm = x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);   // 0 inside .. 1 outside; surface at distance R
          const idx = i + sy * j + sz * k;
          if (fill) { const v = 255 * (1 - sm); if (v > d[idx]) d[idx] = v; }
          else { const v = 255 * sm; if (v < d[idx]) d[idx] = v; }
        }
      }
    }
    if (k0 === F.ks) this._bury(i0, i1, j0, j1);
    this._markDirty(i0, i1, j0, j1, kb, k1);
  }

  _bury(i0, i1, j0, j1) {           // the layers below the sand follow the first layer above it
    const F = this.F, d = F.d, sy = F.nx, sz = F.nx * F.ny;
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const top = d[i + sy * j + sz * F.ks];
      for (let k = 0; k < F.ks; k++) d[i + sy * j + sz * k] = top;
    }
  }

  // Apply one Smooth / Flatten dab at world point c (radius R). amt: blend factor.
  // plane (flatten): {p, n} = point and outward normal of the flat face, or a number = level z.
  brush(c, R, tool, amt, plane) {
    const pl = (plane && typeof plane === 'object') ? plane : { p: new THREE.Vector3(0, 0, plane), n: new THREE.Vector3(0, 0, 1) };
    const F = this.F, dx = F.dx, o = F.o, nx = F.nx, ny = F.ny, d = F.d;
    const i0 = Math.max(0, Math.ceil((c.x - R - o[0]) / dx)), i1 = Math.min(nx - 1, Math.floor((c.x + R - o[0]) / dx));
    const j0 = Math.max(0, Math.ceil((c.y - R - o[1]) / dx)), j1 = Math.min(ny - 1, Math.floor((c.y + R - o[1]) / dx));
    const k0 = Math.max(F.ks, Math.ceil((c.z - R - o[2]) / dx)), k1 = Math.min(F.nz - 1, Math.floor((c.z + R - o[2]) / dx));
    if (i0 > i1 || j0 > j1 || k0 > k1) return;
    const kb = k0 === F.ks ? 0 : k0;           // the buried layers follow the first layer above the sand
    this._saveBlocks(i0, i1, j0, j1, kb, k1);
    const R2 = R * R, sx = 1, sy = nx, sz = nx * ny;
    let src = null;
    if (tool === 'smooth') {                    // read neighbours from a copy so the result doesn't depend on loop order
      src = new Float32Array((i1 - i0 + 3) * (j1 - j0 + 3) * (k1 - k0 + 3));
      let q = 0;
      for (let k = k0 - 1; k <= k1 + 1; k++) for (let j = j0 - 1; j <= j1 + 1; j++) for (let i = i0 - 1; i <= i1 + 1; i++) src[q++] = F.at(i, j, k);
    }
    const W = i1 - i0 + 3, Hh = j1 - j0 + 3;
    for (let k = k0; k <= k1; k++) {
      const z = o[2] + k * dx, dz2 = (z - c.z) ** 2;
      for (let j = j0; j <= j1; j++) {
        const y = o[1] + j * dx, dyz = dz2 + (y - c.y) ** 2;
        if (dyz >= R2) continue;
        for (let i = i0; i <= i1; i++) {
          const x = o[0] + i * dx, r2 = dyz + (x - c.x) ** 2;
          if (r2 >= R2) continue;
          const s = 1 - r2 / R2, w = s * s;
          const idx = i + sy * j + sz * k;
          let val = d[idx];
          if (tool === 'smooth') {
            const q = (i - i0 + 1) + W * ((j - j0 + 1) + Hh * (k - k0 + 1));
            const avg = (src[q - 1] + src[q + 1] + src[q - W] + src[q + W] + src[q - W * Hh] + src[q + W * Hh]) / 6;
            val += amt * w * (avg - val);
          } else {                              // flatten: solid behind the plane, water in front (2-cell ramp)
            const h = (x - pl.p.x) * pl.n.x + (y - pl.p.y) * pl.n.y + (z - pl.p.z) * pl.n.z;
            const t = Math.min(1, Math.max(0, 0.5 - h / (2 * dx))) * 255;
            val += amt * w * (t - val);
          }
          d[idx] = val;
        }
      }
    }
    if (k0 === F.ks) this._bury(i0, i1, j0, j1);
    this._markDirty(i0, i1, j0, j1, kb, k1);
  }

  // ---------------------------------------------------------------- undo blocks
  _bkey(bi, bj, bk) { return bi + 4096 * (bj + 4096 * bk); }

  _blockRange(b, n) { const a = b * B; return [a, Math.min(a + B, n) - 1]; }

  _readBlock(bi, bj, bk) {
    const F = this.F, [i0, i1] = this._blockRange(bi, F.nx), [j0, j1] = this._blockRange(bj, F.ny), [k0, k1] = this._blockRange(bk, F.nz);
    const out = new Float32Array((i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1));
    let q = 0;
    for (let k = k0; k <= k1; k++) for (let j = j0; j <= j1; j++) { const base = F.nx * (j + F.ny * k); for (let i = i0; i <= i1; i++) out[q++] = F.d[base + i]; }
    return out;
  }

  _writeBlock(bi, bj, bk, data) {
    const F = this.F, [i0, i1] = this._blockRange(bi, F.nx), [j0, j1] = this._blockRange(bj, F.ny), [k0, k1] = this._blockRange(bk, F.nz);
    let q = 0;
    for (let k = k0; k <= k1; k++) for (let j = j0; j <= j1; j++) { const base = F.nx * (j + F.ny * k); for (let i = i0; i <= i1; i++) F.d[base + i] = data[q++]; }
    this._markDirty(i0, i1, j0, j1, k0, k1);
  }

  _saveBlocks(i0, i1, j0, j1, k0, k1) {
    const s = this.stroke;
    if (!s) return;
    for (let bk = Math.floor(k0 / B); bk <= Math.floor(k1 / B); bk++)
      for (let bj = Math.floor(j0 / B); bj <= Math.floor(j1 / B); bj++)
        for (let bi = Math.floor(i0 / B); bi <= Math.floor(i1 / B); bi++) {
          const key = this._bkey(bi, bj, bk);
          if (!s.blocks.has(key)) s.blocks.set(key, { b: [bi, bj, bk], data: this._readBlock(bi, bj, bk) });
        }
  }

  _swap(from, to) {
    const m = from.pop();
    if (!m) return false;
    const back = new Map();
    for (const [key, { b, data }] of m) { back.set(key, { b, data: this._readBlock(...b) }); this._writeBlock(...b, data); }
    to.push(back);
    this.modified = true;
    this.volume = this.F.rockVolume();
    this._syncPanelValues();
    return true;
  }

  doUndo() { if (this.F && !this.stroke && this._swap(this.undo, this.redo)) status('sculpt: undone'); }
  doRedo() { if (this.F && !this.stroke && this._swap(this.redo, this.undo)) status('sculpt: redone'); }

  // ---------------------------------------------------------------- meshing
  _key(cx, cy, cz) { return cx + 1024 * (cy + 1024 * cz); }

  _markDirty(i0, i1, j0, j1, k0, k1) {
    // node n changes the vertices of cells n-1 and n; chunk c uses cells c*CH-2 .. c*CH+CH-2;
    // the hole shading reads up to AO_MARGIN nodes away, so chunks that close also redraw
    const m = AO_MARGIN;
    const r = (a, b, n) => [Math.max(0, Math.floor((a + 1) / CH)), Math.min(n - 1, Math.floor((b + 2) / CH))];
    const [x0, x1] = r(i0 - 1 - m, i1 + m, this.nc[0]), [y0, y1] = r(j0 - 1 - m, j1 + m, this.nc[1]), [z0, z1] = r(k0 - 1 - m, k1 + m, this.nc[2]);
    for (let cz = z0; cz <= z1; cz++) for (let cy = y0; cy <= y1; cy++) for (let cx = x0; cx <= x1; cx++) this.dirty.add(this._key(cx, cy, cz));
  }

  _remesh(budgetMs) {
    const t0 = performance.now();
    for (const key of this.dirty) {
      this.dirty.delete(key);
      const cx = key % 1024, cy = Math.floor(key / 1024) % 1024, cz = Math.floor(key / 1048576);
      const g = meshChunk(this.F, cx, cy, cz);
      const old = this.chunks.get(key);
      if (old) { this.group.remove(old); old.geometry.dispose(); this.chunks.delete(key); }
      if (g) { const m = new THREE.Mesh(g, this.mat); m.userData = { kind: 'sculpt' }; this.group.add(m); this.chunks.set(key, m); }
      if (performance.now() - t0 > budgetMs) break;
    }
  }

  // ---------------------------------------------------------------- panel
  _syncPanelValues() {
    const U = this.app.units;
    if (this._rv?.isConnected) { this._rv.textContent = U.fmt(this.radius, 'length', 3); this._rr.value = String(Math.log(this.radius)); }
    if (this._sv?.isConnected) {
      this._sv.textContent = `${Math.round(this.strength * 100)} %` + (this.tool === 'raise' || this.tool === 'lower' ? ` · ${U.fmt(SHIFT_SPEED * this.strength, 'length', 2)}/s` : '');
      this._sr.value = String(this.strength);
    }
    if (this._vol?.isConnected) {
      const dens = this.app.ed.designer.rock_density || 1600;
      this._vol.textContent = `${U.fmt(this.volume, 'volume', 3)} of rock above the sand · about ${U.fmt(this.volume * dens, 'mass', 3)} dry`;
    }
    if (this._ub?.isConnected) { this._ub.disabled = !this.undo.length; this._rb.disabled = !this.redo.length; }
    if (this._mod?.isConnected) this._mod.hidden = !this.modified;
    if (this._dirTxt?.isConnected) {
      const d = this.lockedDir();
      const t = `Tunnel drills ${dirLabel(d)}; Ctrl + Tunnel extrudes rock out the opposite way.` + (this.dirMode === 'view' ? ' Follows your view: turn the view to change it.' : '') +
          (this.dirMode === 'fixed' ? ' Flatten cuts a flat face square to this direction.' : ' Flatten levels (horizontal).');
      if (this._dirTxt.textContent !== t) this._dirTxt.textContent = t;
    }
  }

  // section shown in the designer panel when not sculpting
  entrySection(P) {
    P.append(el('h3', {}, 'Sculpt (terrain-style brushes)'));
    P.append(el('div', { class: 'hint' }, 'Shape the whole rockscape by hand, like a terrain tool: raise rock toward you, lower it away, smooth, flatten shelves, or drill tunnels. Works on whatever the rockscape is now: built elements, an imported scan or a photo reconstruction.'));
    P.append(selectRow('Sculpt resolution', DX_CHOICES.map(([v, t]) => [String(v), t]), String(this.dxChoice), (v) => { this.dxChoice = parseFloat(v); },
      { title: 'Grid spacing of the sculpt. Use at most the CFD cell size of the finest run you plan (Standard = 5 mm, Fine = 3 mm).' }));
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: () => this.enter({ dx: this.dxChoice }) }, 'Sculpt the rockscape')));
  }

  renderPanel() {
    const app = this.app, U = app.units, F = this.F;
    let P = app.panel;
    clear(P);
    P.append(el('h2', {}, 'Sculpt rock'));
    const seg = el('div', { class: 'seg tools' });
    for (const [t, d] of Object.entries(TOOLS)) {
      seg.append(el('button', { class: t === this.tool ? 'on' : '', title: d.title, onclick: () => this.setTool(t) },
        el('span', { class: 'swatch', style: `background:#${d.color.toString(16).padStart(6, '0')}` }), `${d.key} ${d.label}`));
    }
    P.append(seg);
    P.append(el('div', { class: 'hint' }, {
      raise: 'Raise: rock under the brush comes toward you, most in the middle. Scrub back and forth to build up gradually; hold still to grow a dome. Hold Ctrl to lower.',
      lower: 'Lower: rock under the brush moves away from you, most in the middle. Scrub to dig gradually; hold still to dig a hollow. For an opening all the way through, use Tunnel.',
      smooth: 'Smooth: rounds off bumps, steps and sharp edges under the brush.',
      flatten: 'Flatten: levels to the height where the stroke starts (cuts above, fills below).',
      drill: 'Tunnel: a straight round hole along the direction below, deeper the longer you hold; stops when it breaks through. Ctrl: extrude an arm of rock toward you.',
    }[this.tool]));
    // brush direction (Tunnel / Extrude, and Flatten with a fixed direction)
    const P0 = P;
    P = el('div', { hidden: !(this.tool === 'drill' || this.tool === 'flatten') });
    P0.append(P);
    P.append(el('h3', {}, 'Brush direction'));
    const dseg = el('div', { class: 'seg tools' });
    for (const [m, dm] of Object.entries(DIR_MODES)) {
      dseg.append(el('button', { class: m === this.dirMode ? 'on' : '', title: dm.title, onclick: () => this.setDirMode(m) }, dm.label));
    }
    P.append(dseg);
    this._dirTxt = el('div', { class: 'hint' });
    P.append(this._dirTxt);
    if (this.dirMode === 'fixed') {
      const grid = el('div', { class: 'axisgrid' });
      for (const [name, az, elv] of AXIS_PRESETS) {
        const on = Math.abs(this.el - elv) < 1e-6 && (Math.abs(elv) === 90 || Math.abs(wrap180(this.az - az)) < 1e-6);
        grid.append(el('button', { class: on ? 'on' : '', onclick: () => this.setAxis(az, elv) }, name));
      }
      P.append(grid);
      P.append(el('div', { class: 'btns' },
        el('button', { class: 'small', title: 'Turn 15° left (Q)', onclick: () => this.turn(STEP_DEG) }, '↺ 15°'),
        el('button', { class: 'small', title: 'Turn 15° right (E)', onclick: () => this.turn(-STEP_DEG) }, '↻ 15°'),
        el('button', { class: 'small', title: 'Tilt 15° up (W)', onclick: () => this.tilt(STEP_DEG) }, 'Tilt up 15°'),
        el('button', { class: 'small', title: 'Tilt 15° down (S)', onclick: () => this.tilt(-STEP_DEG) }, 'Tilt down 15°')));
    }
    if (this.tool === 'drill') {
      P.append(checkRow('Click bores through', this.bore, (b) => this.setBore(b),
        { title: 'Tunnel: one click drills the whole way through the rock along the direction, in one go (B)' }));
    }
    P = P0;
    // size / strength sliders
    this._rr = el('input', { type: 'range', min: String(Math.log(1.5 * F.dx)), max: String(Math.log(MAX_RADIUS)), step: '0.01', value: String(Math.log(this.radius)) });
    this._rr.addEventListener('input', () => this.setRadius(Math.exp(+this._rr.value)));
    this._rv = el('span', { class: 'val' });
    this._sr = el('input', { type: 'range', min: '0.05', max: '1', step: '0.01', value: String(this.strength) });
    this._sr.addEventListener('input', () => this.setStrength(+this._sr.value));
    this._sv = el('span', { class: 'val' });
    P.append(el('div', { class: 'row' }, el('label', {}, 'Brush radius'), this._rr, this._rv));
    P.append(el('div', { class: 'row' }, el('label', {}, 'Strength'), this._sr, this._sv));
    P.append(el('table', { class: 'keys' },
      ...[['Left drag on rock', 'sculpt; scrub to build up, hold still to keep going'],
        ['Left / right drag off the rock', 'orbit the view'], ['Middle drag', 'pan'],
        ['Mouse wheel', 'brush size  ( [ and ] too )'], ['Shift + wheel', 'strength (speed)'], ['Ctrl + wheel', 'zoom'],
        ['hold Ctrl', 'Raise ↔ Lower, Tunnel → Extrude'], ['hold Shift', 'Smooth'], ['1 2 3 4 5', 'Raise, Lower, Smooth, Flatten, Tunnel'],
        ['Q / E · W / S', 'Tunnel: turn / tilt the direction 15°'], ['V', 'Tunnel: straight in (follow the view)'], ['B', 'Tunnel: click bores through'],
        ['Ctrl+Z / Ctrl+Y', 'undo / redo strokes']]
        .map(([k, v]) => el('tr', {}, el('td', {}, el('kbd', {}, k)), el('td', {}, v)))));
    this._vol = el('div', { class: 'hint num' });
    P.append(this._vol);
    this._ub = el('button', { onclick: () => this.doUndo(), title: 'Ctrl+Z' }, 'Undo stroke');
    this._rb = el('button', { onclick: () => this.doRedo(), title: 'Ctrl+Y' }, 'Redo');
    P.append(el('div', { class: 'btns' }, this._ub, this._rb));
    this._mod = el('div', { class: 'warnbox' }, 'Unapplied changes: the simulation still uses the previous rockscape until you press "Apply to project".');
    P.append(this._mod);
    P.append(el('div', { class: 'btns' },
      el('button', { class: 'primary', onclick: () => this.applyToProject(), title: 'Write the sculpt as the project rockscape (STL) used by runs and lighting' }, 'Apply to project'),
      el('button', { onclick: () => this.saveDraft().catch(e => status(e.message, 'err')), title: 'Keep the work without changing the project rockscape' }, 'Save draft'),
      el('button', { onclick: () => this.leave() }, 'Done')));
    if (this.lastReport) {
      const r = this.lastReport;
      P.append(el('h3', {}, 'Applied'));
      P.append(kvTable([['Rock above the sand', U.fmt(r.volume_voxel_m3, 'volume', 3)], ['Mesh', `${r.triangles.toLocaleString()} triangles, ${r.watertight ? 'watertight' : 'NOT watertight'}`],
        ['Grid', `${r.grid.join(' × ')} at ${U.fmt(r.resolution_m, 'length', 2)}`]]));
      P.append(el('h3', {}, `Geometry checks (${r.issues.length})`));
      if (!r.issues.length) P.append(el('div', { class: 'okbox' }, 'No issues: rock clear of the glass, supported on the sand, below the water surface.'));
      for (const is of r.issues) P.append(el('div', { class: 'issue ' + is.severity }, (is.severity === 'warning' ? '⚠ ' : 'ℹ ') + is.message));
    }
    P.append(el('h3', {}, 'Grid'));
    P.append(el('div', { class: 'hint' }, `${this.meta.source}. ${F.nx} × ${F.ny} × ${F.nz} nodes at ${U.fmt(F.dx, 'length', 2)}. ` +
      `The flow model sees the rock at its own cell size: a cave or gap narrower than about 3 cells (${U.fmt(0.015, 'length', 2)} at Standard, ${U.fmt(0.03, 'length', 2)} at Preview) carries little or no flow there.`));
    P.append(el('div', { class: 'btns' },
      selectRow('Resolution', DX_CHOICES.map(([v, t]) => [String(v), t]), String(F.dx), async (v) => {
        if (this.modified && !confirm('Changing the resolution restarts from the project rockscape (apply first to keep your strokes). Continue?')) { this.app.renderPanel(); return; }
        this.modified = false; await this.enter({ dx: parseFloat(v), restart: true, fresh: true });
      }),
      el('button', { class: 'small', onclick: async () => {
        if (!confirm('Start over from the project rockscape? Unapplied strokes are lost.')) return;
        this.modified = false; await this.enter({ dx: F.dx, restart: true, fresh: true });
      } }, 'Start over')));
    this._syncPanelValues();
  }
}
