// Results viewer (slices from fields.npz, sand shear, coral scores on the rock, PAR hook)
// and comparison mode. Slices are cut on the server; only 2-D arrays reach the browser.
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { api, el, clear, kvTable, num, status, busy, selectRow, modal } from './util.js';
import { meshFromBuffers, rockMaterial, disposeTree, textSprite } from './viewport.js';
import { pumpObject } from './equipment.js';
import { Particles } from './particles.js';

const FIELD_CMAP = { speed: 'viridis', umag: 'viridis', sigma: 'viridis', ti: 'plasma', age: 'magma', par: 'YlOrRd' };
const AXES = { z: 'Plan (horizontal)', y: 'Front elevation', x: 'Side elevation' };
const FIGS = [['slices.png', 'Speed slices'], ['dead_zones.png', 'Dead zones'], ['age.png', 'Water age'], ['sand_shear.png', 'Sand shear'],
  ['coral_scores.png', 'Coral scores'], ['history.png', 'Run history'], ['geometry.png', 'Voxel model']];

export class Results {
  constructor(app) {
    this.app = app;
    this.group = new THREE.Group();
    app.view.resultsRoot.add(this.group);
    this.runs = [];
    this.cur = null;            // {pid, rid}
    this.sum = null;
    this.snap = null;           // editor model of the run's project snapshot
    this.cmaps = null;
    this.layers = { field: 'speed', rock: 'score', profile: 'SPS', spots: true, pumps: true, vectors: false, dead: false, sand: false,
      z: { on: true, f: 0.5 }, y: { on: false, f: 0.5 }, x: { on: false, f: 0.5 }, auto: true, lo: 0, hi: 0.3 };
    this.particles = new Particles(app, this.group.parent);
    this.slices = {};           // axis -> {mesh, dead, vec, meta, data}
    this.objs = {};
    this.seq = 0;
  }

  async init() { if (!this.cmaps) this.cmaps = await api.get('/api/colormaps'); }

  async refreshRuns() {
    if (!this.app.pid) return;
    try { this.runs = (await api.get(`/api/projects/${this.app.pid}/runs`)).runs; } catch (e) { this.runs = []; }
    this.app.updateSteps?.();
    if (this.app.tab === 'results') this.render();
  }

  async open(pid, rid) {
    await this.init();
    this.cur = { pid, rid };
    this.clearScene();
    this.particles.forget();
    this.sum = null; this.snap = null;
    try {
      [this.sum, this.snap] = await busy('loading results', () => Promise.all([api.get(`/api/runs/${pid}/${rid}`), api.get(`/api/runs/${pid}/${rid}/project`)]));
    } catch (e) { status(e.message, 'err'); this.render(); return; }
    this.snap = this.snap.editor;
    if (this.sum.profiles?.length && !this.sum.profiles.includes(this.layers.profile)) this.layers.profile = this.sum.profiles[0];
    if (this.layers.field === 'par' && !this.sum.layers.par) this.layers.field = 'speed';
    if (this.app.tab === 'results') { this.showShell(); await this.buildScene(); }
    this.render();
  }

  showShell() {
    if (this.snap) this.app.view.setTank(this.snap.tank, this.snap.overflow);
  }

  base() { return `/api/runs/${this.cur.pid}/${this.cur.rid}`; }

  clearScene() {
    for (const c of [...this.group.children]) { this.group.remove(c); disposeTree(c); }
    this.slices = {}; this.objs = {};
    this.app.view.render();
  }

  // ---------------------------------------------------------------- scene
  async buildScene() {
    if (!this.sum?.layers?.speed) { this.updateLegend(); return; }
    const seq = ++this.seq;
    await Promise.all([this.loadRock(seq), this.updateSlices(), this.updateSand(), this.updatePumps()]);
    this.updateSpots();
    this.updateLegend();
  }

  async loadRock(seq = this.seq) {
    if (this.objs.rock) { this.group.remove(this.objs.rock); disposeTree(this.objs.rock); this.objs.rock = null; }
    const L = this.layers;
    if (L.rock === 'off') { this.app.view.render(); return; }
    let mesh;
    try {
      if (L.rock === 'score' && this.sum.profiles.includes(L.profile)) {
        const geo = await new PLYLoader().loadAsync(`${this.base()}/file/rock_score_${L.profile}.ply`);
        const c = geo.getAttribute('color');
        if (c) {                                  // PLY colours are sRGB; three.js vertex colours are linear
          const col = new THREE.Color();
          for (let i = 0; i < c.count; i++) { col.setRGB(c.getX(i), c.getY(i), c.getZ(i), THREE.SRGBColorSpace); c.setXYZ(i, col.r, col.g, col.b); }
        }
        geo.computeVertexNormals();
        mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: !!c, roughness: 0.9 }));
      } else {
        const m = await api.mesh(`${this.base()}/rock_mesh`);
        if (!m.nv) return;
        mesh = meshFromBuffers(m.positions, m.index, rockMaterial());
      }
    } catch (e) { status(`rock mesh: ${e.message}`, 'err'); return; }
    if (seq !== this.seq) { disposeTree(mesh); return; }
    mesh.userData = { kind: 'rock' };
    this.objs.rock = mesh;
    this.group.add(mesh);
    this.app.view.render();
  }

  range(field, meta) {
    const L = this.layers;
    if (!L.auto && L.field === field) return [L.lo, L.hi];
    if (field === 'ti') return [0, Math.min(meta.p99 ?? 1, 2)];
    if (field === 'age') return [meta.p1 ?? meta.min ?? 0, meta.p99 || meta.max || 1];
    return [0, meta.p99 || meta.max || 1];
  }

  // display value + unit of a field's SI value
  disp(field) {
    const U = this.app.units;
    if (['speed', 'umag', 'sigma'].includes(field)) return { f: (v) => U.to(v, 'velocity'), unit: U.unit('velocity'), inv: (v) => U.from(v, 'velocity') };
    if (field === 'age') return { f: (v) => v / 60, unit: 'min', inv: (v) => v * 60 };
    if (field === 'ti') return { f: (v) => 100 * v, unit: '%', inv: (v) => v / 100 };
    if (field === 'par') return { f: (v) => v, unit: 'µmol/m²/s', inv: (v) => v };
    if (field === 'shear') return { f: (v) => v, unit: 'Pa', inv: (v) => v };
    return { f: (v) => v, unit: '', inv: (v) => v };
  }

  texture(data, w, h, lut, lo, hi, alpha = 255) {
    const rgba = new Uint8Array(w * h * 4);
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!Number.isFinite(v)) continue;
      const t = Math.min(Math.max((v - lo) / span, 0), 1);
      const k = Math.round(t * 255) * 3;
      rgba[4 * i] = lut[k]; rgba[4 * i + 1] = lut[k + 1]; rgba[4 * i + 2] = lut[k + 2]; rgba[4 * i + 3] = alpha;
    }
    const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
    tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  }

  planeGeometry(axis, meta) {
    const [u0, u1, v0, v1] = meta.extent, dx = meta.dx;
    const w = u1 - u0 + dx, h = v1 - v0 + dx, cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
    const g = new THREE.PlaneGeometry(w, h);
    if (axis === 'z') g.translate(cu, cv, meta.pos);
    if (axis === 'y') { g.rotateX(Math.PI / 2); g.translate(cu, meta.pos, cv); }
    if (axis === 'x') { g.rotateX(Math.PI / 2); g.rotateZ(Math.PI / 2); g.translate(meta.pos, cu, cv); }
    return g;
  }

  posOf(axis) {
    const g = this.sum.grid, a = 'xyz'.indexOf(axis);
    const lo = g.origin_m[a] + g.dx_m, hi = g.origin_m[a] + g.dx_m * (g.shape[a] - 2);
    return lo + (hi - lo) * this.layers[axis].f;
  }

  async updateSlices(only) {
    for (const axis of only ? [only] : ['z', 'y', 'x']) await this.updateSlice(axis);
    this.updateLegend();
  }

  async updateSlice(axis) {
    const s = this.slices[axis];
    if (s) { for (const k of ['mesh', 'dead', 'vec']) if (s[k]) { this.group.remove(s[k]); disposeTree(s[k]); } }
    delete this.slices[axis];
    const L = this.layers;
    if (!L[axis].on || !this.sum?.layers?.speed) { this.app.view.render(); return; }
    const pos = this.posOf(axis);
    const token = (this._tok = (this._tok || 0) + 1);
    let r;
    try { r = await api.binary(`${this.base()}/slice?field=${L.field}&axis=${axis}&pos=${pos}`); }
    catch (e) { status(e.message, 'err'); return; }
    const meta = r.meta, data = new Float32Array(r.buf);
    const [h, w] = meta.shape;
    const [lo, hi] = this.range(L.field, meta);
    const tex = this.texture(data, w, h, this.cmaps[FIELD_CMAP[L.field]], lo, hi);
    const mesh = new THREE.Mesh(this.planeGeometry(axis, meta), new THREE.MeshBasicMaterial({ map: tex, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide }));
    mesh.userData = { kind: 'slice', axis };
    const entry = { mesh, meta, data, lo, hi };
    if (this.slices[axis]) { const o = this.slices[axis]; for (const k of ['mesh', 'dead', 'vec']) if (o[k]) { this.group.remove(o[k]); disposeTree(o[k]); } }
    this.slices[axis] = entry;
    this.group.add(mesh);
    if (L.dead && this.sum.layers.dead) {
      try {
        const d = await api.binary(`${this.base()}/slice?field=dead&axis=${axis}&pos=${pos}`);
        const dd = new Float32Array(d.buf);
        const rgba = new Uint8Array(w * h * 4);
        for (let i = 0; i < dd.length; i++) if (dd[i] > 0.5) { rgba[4 * i] = 230; rgba[4 * i + 1] = 30; rgba[4 * i + 2] = 30; rgba[4 * i + 3] = 170; }
        const t2 = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
        t2.magFilter = t2.minFilter = THREE.NearestFilter; t2.needsUpdate = true;
        const dm = new THREE.Mesh(this.planeGeometry(axis, meta), new THREE.MeshBasicMaterial({ map: t2, transparent: true, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
        dm.renderOrder = 2;
        entry.dead = dm; this.group.add(dm);
      } catch (e) { status(e.message, 'err'); }
    }
    if (L.vectors) {
      try {
        const v = await api.binary(`${this.base()}/vectors?axis=${axis}&pos=${pos}`);
        entry.vec = this.vectorLines(axis, v.meta, new Float32Array(v.buf));
        this.group.add(entry.vec);
      } catch (e) { status(e.message, 'err'); }
    }
    this.app.view.render();
  }

  vectorLines(axis, meta, buf) {
    const n = meta.n, P = buf.subarray(0, 2 * n), V = buf.subarray(2 * n, 4 * n);
    const mags = []; for (let i = 0; i < n; i++) mags.push(Math.hypot(V[2 * i], V[2 * i + 1]));
    const sorted = [...mags].sort((a, b) => a - b);
    const vref = sorted[Math.floor(0.95 * (n - 1))] || 1;
    const scale = (meta.stride * this.sum.grid.dx_m * 0.95) / vref;
    const pts = [];
    const off = 0.0015;
    const map = (u, v) => axis === 'z' ? [u, v, meta.pos + off] : axis === 'y' ? [u, meta.pos - off, v] : [meta.pos + off, u, v];
    for (let i = 0; i < n; i++) {
      const u = P[2 * i], v = P[2 * i + 1], du = V[2 * i] * scale, dv = V[2 * i + 1] * scale;
      const len = Math.hypot(du, dv);
      if (len < 1e-5) continue;
      const k = Math.min(len, meta.stride * this.sum.grid.dx_m * 1.6) / len;
      const eu = u + du * k, ev = v + dv * k;
      pts.push(...map(u, v), ...map(eu, ev));
      const hu = -du * k * 0.3, hv = -dv * k * 0.3;
      const c = Math.cos(0.45), s = Math.sin(0.45);
      pts.push(...map(eu, ev), ...map(eu + hu * c - hv * s, ev + hu * s + hv * c));
      pts.push(...map(eu, ev), ...map(eu + hu * c + hv * s, ev - hu * s + hv * c));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x111111 }));
  }

  async updateSand() {
    if (this.objs.sand) { this.group.remove(this.objs.sand); disposeTree(this.objs.sand); this.objs.sand = null; }
    this.sandMeta = null;
    if (!this.layers.sand || !this.sum?.layers?.sand_shear) { this.app.view.render(); return; }
    try {
      const r = await api.binary(`${this.base()}/sand_shear`);
      const m = r.meta, data = new Float32Array(r.buf);
      const [h, w] = m.shape;
      const hi = Math.max(m.p99 || 0, 3 * (m.tau_crit_Pa || 0.02));
      const tex = this.texture(data, w, h, this.cmaps.cividis, 0, hi);
      const dx = this.sum.grid.dx_m;
      const [u0, u1, v0, v1] = m.extent;
      const g = new THREE.PlaneGeometry(u1 - u0 + dx, v1 - v0 + dx);
      g.translate((u0 + u1) / 2, (v0 + v1) / 2, this.snap.tank.sand_depth + 0.0008);
      const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, transparent: true, alphaTest: 0.5 }));
      mesh.userData = { kind: 'sand' };
      this.objs.sand = mesh; this.group.add(mesh);
      this.sandMeta = { ...m, lo: 0, hi, data, w, h };
    } catch (e) { status(e.message, 'err'); }
    this.updateLegend();
    this.app.view.render();
  }

  updatePumps() {
    if (this.objs.pumps) { this.group.remove(this.objs.pumps); disposeTree(this.objs.pumps); this.objs.pumps = null; }
    if (!this.layers.pumps || !this.snap) { this.app.view.render(); return; }
    const g = new THREE.Group();
    for (const p of this.snap.pumps) g.add(pumpObject(p, this.app.lib, { arrow: 0.12 }));
    this.objs.pumps = g; this.group.add(g);
    this.app.view.render();
  }

  spots() { return this.sum?.analysis?.coral?.best_spots?.[this.layers.profile] || []; }

  updateSpots() {
    if (this.objs.spots) { this.group.remove(this.objs.spots); disposeTree(this.objs.spots); this.objs.spots = null; }
    if (!this.layers.spots || this.layers.rock === 'off') { this.app.view.render(); return; }
    const g = new THREE.Group();
    this.spots().forEach((s, i) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.007, 16, 12), new THREE.MeshBasicMaterial({ color: 0xe0119d }));
      m.position.set(...s.position_m);
      m.userData = { kind: 'spot', index: i };
      g.add(m);
      g.add(textSprite(String(i + 1), s.position_m[0], s.position_m[1], s.position_m[2] + 0.022, 0.022, '#fff', 'rgba(224,17,157,0.9)'));
    });
    this.objs.spots = g; this.group.add(g);
    this.app.view.render();
  }

  // ---------------------------------------------------------------- legend + hover
  colorbar(cmap, lo, hi, label, unit, f, extra) {
    const c = el('canvas', { width: 256, height: 12 });
    const ctx = c.getContext('2d');
    const lut = this.cmaps[cmap];
    for (let i = 0; i < 256; i++) { ctx.fillStyle = `rgb(${lut[3 * i]},${lut[3 * i + 1]},${lut[3 * i + 2]})`; ctx.fillRect(i, 0, 1, 12); }
    return el('div', { class: 'bar' }, el('div', {}, el('b', {}, label), unit ? ` [${unit}]` : ''), c,
      el('div', { class: 'ticks' }, el('span', {}, num(f(lo), 3)), el('span', {}, num(f((lo + hi) / 2), 3)), el('span', {}, num(f(hi), 3) + (hi === (this._p99 ?? null) ? '+' : ''))),
      extra ? el('div', { class: 'hint', style: 'margin:0' }, extra) : null);
  }

  updateLegend() {
    const lg = document.getElementById('legend');
    if (this.app.tab !== 'results' || !this.sum?.layers?.speed) { lg.hidden = true; return; }
    clear(lg);
    const L = this.layers;
    const anySlice = ['z', 'y', 'x'].find(a => this.slices[a]);
    if (anySlice) {
      const s = this.slices[anySlice], d = this.disp(L.field);
      const lab = this.sum.fields[L.field]?.label || L.field;
      lg.append(this.colorbar(FIELD_CMAP[L.field], s.lo, s.hi, lab, d.unit, d.f, L.field === 'par' ? (this.sum.par_label || '') : null));
      if (L.dead) lg.append(el('div', { class: 'hint', style: 'margin:0 0 6px' }, el('span', { style: 'display:inline-block;width:12px;height:10px;background:rgba(230,30,30,.7);margin-right:4px' }),
        `dead zone: <|u|> < ${this.app.units.fmt(this.sum.analysis?.criteria?.dead_zone_speed, 'velocity', 3)}`));
    }
    if (this.sandMeta) {
      const m = this.sandMeta;
      lg.append(this.colorbar('cividis', m.lo, m.hi, 'Sand bed shear stress', 'Pa', (v) => v,
        `below ${num(m.tau_crit_Pa, 2)} Pa detritus settles; above ${num(m.tau_sand_move_Pa, 2)} Pa sand moves`));
    }
    if (L.rock === 'score' && this.objs.rock) lg.append(this.colorbar('viridis', 0, 1, `Coral score (${L.profile})`, '0–1', (v) => v,
      this.sum.analysis?.coral?.light_available ? 'flow × light' : 'flow only (no PAR in this run); grey = undersides, not scored'));
    lg.hidden = !lg.children.length;
  }

  hover(e) {
    const ro = document.getElementById('readout');
    if (!e || this.app.tab !== 'results') { ro.hidden = true; return; }
    const objs = Object.values(this.slices).map(s => s.mesh);
    if (this.objs.sand) objs.push(this.objs.sand);
    const hit = objs.length ? this.app.view.pick(e, objs) : null;
    if (!hit) { ro.hidden = true; return; }
    const p = hit.point, U = this.app.units, dx = this.sum.grid.dx_m;
    let txt;
    if (hit.object.userData.kind === 'sand') {
      const m = this.sandMeta, i = Math.round((p.x - m.extent[0]) / dx), j = Math.round((p.y - m.extent[2]) / dx);
      const v = m.data[j * m.w + i];
      txt = `sand shear ${Number.isFinite(v) ? num(v, 3) + ' Pa' : '–'}`;
    } else {
      const axis = hit.object.userData.axis, s = this.slices[axis], m = s.meta;
      const [u, v] = axis === 'z' ? [p.x, p.y] : axis === 'y' ? [p.x, p.z] : [p.y, p.z];
      const i = Math.round((u - m.extent[0]) / dx), j = Math.round((v - m.extent[2]) / dx);
      const val = s.data[j * m.shape[1] + i];
      const d = this.disp(this.layers.field);
      txt = `${this.sum.fields[this.layers.field].label}: ${Number.isFinite(val) ? num(d.f(val), 3) + ' ' + d.unit : 'solid'}`;
    }
    ro.textContent = `${txt}   at x ${U.fmtv(p.x, 'length', 3)}, y ${U.fmtv(p.y, 'length', 3)}, z ${U.fmtv(p.z, 'length', 3)} ${U.unit('length')}`;
    ro.hidden = false;
  }

  // ---------------------------------------------------------------- panel
  render() {
    const app = this.app, P = app.panel, U = app.units;
    clear(P);
    P.append(el('h2', {}, 'Results'));
    const opts = this.runs.map(r => [r.id, `${r.label} — ${r.status}`]);
    if (!opts.length) {
      P.append(el('div', { class: 'hint' }, 'No runs for this project yet. Start one in the Run tab.'));
      this.updateLegend();
      return;
    }
    const sel = selectRow('Run', [['', '— choose a run —'], ...opts], this.cur?.pid === app.pid ? this.cur.rid : '', (rid) => { if (rid) this.open(app.pid, rid); });
    P.append(sel);
    if (!this.cur || this.cur.pid !== app.pid || !this.sum) { this.updateLegend(); return; }
    const st = this.sum.status, an = this.sum.analysis;
    P.append(el('div', { class: 'btns' },
      this.sum.layers.speed ? null : el('span', { class: 'status-' + st.status }, `status: ${st.status}`),
      el('a', { class: 'button', href: `${this.base()}/file/report.html`, target: '_blank', hidden: !this.sum.files.includes('report.html') }, 'Open report'),
      el('button', { onclick: () => this.reanalyze(), disabled: !this.sum.layers.speed, title: 'Run the analysis again (e.g. after changing criteria)' }, 'Re-analyze'),
      el('a', { class: 'button', href: `${this.base()}/file/ui_log.txt`, target: '_blank', hidden: !this.sum.files.includes('ui_log.txt') }, 'Log'),
      el('button', { class: 'danger', onclick: () => this.deleteRun() }, 'Delete run')));
    if (st.error) P.append(el('div', { class: 'errbox' }, st.error));
    if (!this.sum.layers.speed) { P.append(el('div', { class: 'hint' }, 'This run has no flow results (yet).')); this.updateLegend(); return; }

    this._layersPanel(P);
    if (an) this._metricsPanel(P, an);
    else P.append(el('div', { class: 'warnbox' }, 'Not analysed yet: press Re-analyze for dead zones, water age, sand shear and coral scores.'));
    this._runInfo(P);
    this.updateLegend();
  }

  _layersPanel(P) {
    const L = this.layers, s = this.sum, U = this.app.units;
    P.append(el('h3', {}, 'Layers'));
    const fields = Object.entries(s.fields).filter(([k]) => k !== 'dead').map(([k, f]) => {
      const ok = s.layers[k];
      return [k, f.label + (ok ? '' : k === 'par' ? ' (run with lighting)' : ' (n/a)')];
    });
    const fsel = selectRow('Slice field', fields, L.field, (v) => {
      if (!s.layers[v]) { status(v === 'par' ? 'No PAR in this run: tick "Compute PAR" when starting a run (needs light fixtures).' : `${v} not available for this run`, 'err'); this.render(); return; }
      L.field = v; L.auto = true; this.updateSlices(); this.render();
    }, { title: s.fields[L.field]?.description });
    P.append(fsel);
    for (const axis of ['z', 'y', 'x']) {
      const cb = el('input', { type: 'checkbox' }); cb.checked = L[axis].on;
      const rng = el('input', { type: 'range', min: 0, max: 1, step: 0.005, value: L[axis].f });
      const val = el('span', { class: 'val' }, U.fmt(this.posOf(axis), 'length', 3));
      cb.addEventListener('change', () => { L[axis].on = cb.checked; this.updateSlices(axis); });
      let t = null;
      rng.addEventListener('input', () => {
        L[axis].f = parseFloat(rng.value); val.textContent = U.fmt(this.posOf(axis), 'length', 3);
        clearTimeout(t); t = setTimeout(() => { if (L[axis].on) this.updateSlices(axis); }, 120);
      });
      P.append(el('div', { class: 'layer' }, cb, el('span', {}, `${AXES[axis]} at ${axis} =`), el('span', {}, rng, ' ', val)));
    }
    const chk = (label, key, fn, avail = true, title) => {
      const cb = el('input', { type: 'checkbox', disabled: !avail }); cb.checked = L[key] && avail;
      cb.addEventListener('change', () => { L[key] = cb.checked; fn(); });
      return el('div', { class: 'layer', title }, cb, el('span', {}, label + (avail ? '' : ' (n/a)')), el('span'));
    };
    P.append(chk('Velocity vectors on slices', 'vectors', () => this.updateSlices()));
    P.append(chk('Dead-zone overlay (red)', 'dead', () => this.updateSlices(), s.layers.dead, 'cells whose time-averaged speed is below the dead-zone threshold'));
    P.append(chk('Sand bed shear stress', 'sand', () => this.updateSand(), s.layers.sand_shear, 'detritus settles where the bed shear is low'));
    P.append(chk('Pumps', 'pumps', () => this.updatePumps()));
    // colour range
    const d = this.disp(L.field);
    const sl = Object.values(this.slices)[0];
    const lo = el('input', { type: 'number', step: 'any', value: sl ? +d.f(sl.lo).toPrecision(4) : '', disabled: L.auto });
    const hi = el('input', { type: 'number', step: 'any', value: sl ? +d.f(sl.hi).toPrecision(4) : '', disabled: L.auto });
    const auto = el('input', { type: 'checkbox' }); auto.checked = L.auto;
    auto.addEventListener('change', () => { L.auto = auto.checked; lo.disabled = hi.disabled = L.auto; if (L.auto) this.updateSlices(); });
    const apply = () => { const a = parseFloat(lo.value), b = parseFloat(hi.value); if (Number.isFinite(a) && Number.isFinite(b) && b > a) { L.lo = d.inv(a); L.hi = d.inv(b); this.updateSlices(); } };
    lo.addEventListener('change', apply); hi.addEventListener('change', apply);
    P.append(el('div', { class: 'row', style: 'grid-template-columns: 1fr auto auto auto 52px' }, el('label', {}, el('span', {}, auto, ' auto colour range')), lo, el('span', {}, '–'), hi, el('span', { class: 'unit' }, d.unit)));
    // rock
    P.append(selectRow('Rock display', [['score', 'coral score'], ['plain', 'plain rock'], ['off', 'hidden']], L.rock, (v) => { L.rock = v; this.loadRock().then(() => { this.updateSpots(); this.updateLegend(); }); }));
    if (s.profiles.length) {
      P.append(selectRow('Coral profile', s.profiles, L.profile, (v) => { L.profile = v; this.loadRock().then(() => { this.updateSpots(); this.updateLegend(); this.render(); }); }));
      P.append(chk('Best-spot markers', 'spots', () => this.updateSpots()));
    }
    this._particlePanel(P);
    if (!s.layers.par) P.append(el('div', { class: 'hint' }, 'PAR layer: not in this run. Runs started with "Compute PAR" (Phase 4 lighting) add it, and the coral score then includes light.'));
  }

  // ---------------------------------------------------------------- particle animation (D-849)
  _pumpSeeds() {
    const out = [];
    for (const p of this.snap?.pumps || []) for (const n of (p.nozzles?.length ? p.nozzles : [p])) {
      const a = n.aim || [1, 0, 0], m = Math.hypot(...a) || 1;
      out.push({ p: n.position, aim: a.map(v => v / m) });
    }
    return out;
  }

  _particlePanel(P) {
    const pa = this.particles, o = pa.opts;
    const go = async () => {
      try { await busy('loading the flow field', () => pa.start(this.cur.pid, this.cur.rid, this._pumpSeeds())); }
      catch (e) { status(e.message, 'err'); pa.on = false; }
      this.render();
    };
    P.append(el('h3', {}, 'Particle animation'));
    P.append(el('div', { class: 'btns' },
      el('button', { class: pa.on ? '' : 'primary', onclick: () => { if (pa.on) { pa.stop(); this.render(); } else go(); } }, pa.on ? 'Stop particles' : 'Animate particles')));
    P.append(selectRow('Release particles', [['everywhere', 'everywhere in the water'], ['pumps', 'at the pump outlets']], o.release, (v) => { o.release = v; pa.restart(this.cur.pid, this.cur.rid, this._pumpSeeds()); }));
    P.append(selectRow('Number of particles', [[1000, '1,000'], [3000, '3,000'], [6000, '6,000'], [12000, '12,000']], o.count, (v) => { o.count = +v; pa.restart(this.cur.pid, this.cur.rid, this._pumpSeeds()); }));
    P.append(selectRow('Playback speed', [[0.25, '1/4 real time'], [0.5, '1/2 real time'], [1, 'real time'], [2, '2x'], [4, '4x']], o.speed, (v) => { o.speed = +v; }));
    P.append(selectRow('Trail length', [[8, 'short'], [16, 'medium'], [32, 'long']], o.trail, (v) => { o.trail = +v; pa.restart(this.cur.pid, this.cur.rid, this._pumpSeeds()); }));
    const vmax = pa.field ? this.app.units.fmt(pa.field.vmax, 'small_velocity', 2) : null;
    P.append(el('div', { class: 'hint' }, `Particles drift with the time-averaged flow: blue is slow, cyan medium, pale yellow fast${vmax ? ` (${vmax} and up)` : ''}. Particles that hang about or circle show slow or recirculating water. Real water also swirls on top of this average.`));
  }

  _metricsPanel(P, an) {
    const U = this.app.units, km = this.sum.key_metrics, m = an.metrics, unc = an.metrics_stat_uncertainty || {};
    const conv = (v, unit) => unit === 'cm/s' ? [U.to(v, 'velocity'), U.unit('velocity')] : unit === '%' ? [100 * v, '%'] : [v, unit];
    P.append(el('h3', {}, 'Key results'));
    const age = an.age;
    P.append(el('div', { class: 'kpi' },
      el('div', {}, el('b', {}, num(conv(m.mean_speed, 'cm/s')[0], 3)), el('span', {}, `mean speed [${U.unit('velocity')}]`)),
      el('div', {}, el('b', {}, num(100 * m.dead_fraction, 3)), el('span', {}, '% dead water')),
      el('div', {}, el('b', {}, age ? num(age.mean_age_over_tau, 3) : '–'), el('span', {}, 'mean age / turnover time'))));
    const t = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Metric'), el('th', {}, 'Value'), el('th', {}, '± (1σ stat.)')));
    for (const [k, d] of Object.entries(km)) {
      if (m[k] === undefined) continue;
      const [v, unit] = conv(m[k], d.unit);
      const [e] = unc[k] !== undefined ? conv(unc[k], d.unit) : [null];
      t.append(el('tr', {}, el('td', {}, `${d.label} [${unit}]`), el('td', { class: 'num' }, num(v, 3)), el('td', { class: 'num' }, e === null ? '' : num(e, 2))));
    }
    P.append(t);
    if (age) P.append(el('details', {}, el('summary', {}, 'Water age'), kvTable([
      ['Nominal turnover time V/Q', `${num(age.tau_nominal_s / 60, 3)} min`],
      ['Mean age', `${num(age.mean_age_s / 60, 3)} min (${num(age.mean_age_over_tau, 3)} × turnover)`],
      ['95th percentile age', `${num(age.p95_age_over_tau, 3)} × turnover`],
      ['Max age', `${num(age.max_age_over_tau, 3)} × turnover`]])));
    const dz = an.dead_zones;
    if (dz?.zones?.length) {
      const tz = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, '#'), el('th', {}, 'Volume'), el('th', {}, 'Centre x, y, z'), el('th', {}, 'Mean speed')));
      dz.zones.slice(0, 8).forEach((z, i) => tz.append(el('tr', { class: 'click', title: 'touches: ' + (z.touches || []).join(', '), onclick: () => this.focus(z.centroid_m) },
        el('td', {}, String(i + 1)), el('td', {}, `${num(z.volume_L, 3)} L (${num(100 * z.volume_fraction, 2)} %)`),
        el('td', {}, z.centroid_m.map(c => U.fmtv(c, 'length', 3)).join(', ') + ' ' + U.unit('length')), el('td', {}, U.fmt(z.mean_speed_ms, 'velocity', 2)))));
      P.append(el('details', { open: true }, el('summary', {}, `Largest dead zones (${dz.n_zones}, threshold ${U.fmt(dz.threshold_ms, 'velocity', 3)})`), tz));
    }
    const ws = an.wall_shear;
    if (ws) P.append(el('details', {}, el('summary', {}, 'Sand bed / detritus'), kvTable([
      ['Sand area where detritus settles', `${num(100 * ws.sand_area_settling_fraction, 3)} %`, `bed shear below ${ws.tau_crit_Pa} Pa`],
      ['Sand area where sand moves', `${num(100 * ws.sand_area_moving_fraction, 3)} %`, `bed shear above ${ws.tau_sand_move_Pa} Pa`],
      ['Median bed shear', `${num(ws.sand_tau_median_Pa, 3)} Pa`],
      ['Rock tops where detritus settles', `${num(100 * ws.rock_top_settling_fraction, 3)} %`]])));
    const co = an.coral;
    if (co) {
      const tc = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Profile'), el('th', {}, 'Mean score'), el('th', {}, 'Area ≥ 0.7'), el('th', {}, 'Jet-penalised')));
      for (const [k, p] of Object.entries(co.profiles)) tc.append(el('tr', {}, el('td', {}, k), el('td', {}, num(p.mean_score, 3)), el('td', {}, `${num(100 * p['fraction_score_ge_0.7'], 3)} %`), el('td', {}, `${num(100 * p.fraction_jet_penalised, 3)} %`)));
      P.append(el('h3', {}, `Coral placement ${co.light_available ? '(flow × light)' : '(flow only)'}`), tc);
      const spots = this.spots();
      if (spots.length) {
        const ts = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, `#`), el('th', {}, 'Position x, y, z'), el('th', {}, 'Score'), el('th', {}, 'Speed'), el('th', {}, 'TI')));
        spots.forEach((s, i) => ts.append(el('tr', { class: 'click', onclick: () => this.focus(s.position_m) }, el('td', {}, String(i + 1)),
          el('td', {}, s.position_m.map(c => U.fmtv(c, 'length', 3)).join(', ')), el('td', {}, num(s.score, 2)),
          el('td', {}, U.fmt(s.speed_ms, 'velocity', 2)), el('td', {}, `${num(100 * s.turbulence_intensity, 2)} %`))));
        P.append(el('details', { open: true }, el('summary', {}, `Best spots for ${this.layers.profile} (click to look)`), ts));
      }
    }
  }

  _runInfo(P) {
    const r = this.sum.run, U = this.app.units;
    if (!r) return;
    P.append(el('details', {}, el('summary', {}, 'Run details'), kvTable([
      ['Backend / precision', `${r.backend} / ${r.precision}`], ['Lattice / collision', `${r.lattice} / ${r.collision}`],
      ['Cell size', U.fmt(r.dx_m, 'length', 3)], ['Spin-up + averaging', `${num(r.spinup_s, 3)} s + ${num(r.average_s, 3)} s`],
      ['Wall time', `${num(r.wall_time_s / 60, 3)} min (${num(r.mlups, 4)} MLUPS)`], ['Mass drift', `${num(100 * r.mass_rel_drift, 3)} %`],
      ['Max speed (lattice)', `${num(r.u_max_overall_ms, 3)} m/s (${num(r.u_max_overall_lattice, 3)})`],
      ['KE first/second half of averaging', `${num(r.ke_avg_first_half, 3)} / ${num(r.ke_avg_second_half, 3)} m²/s²`, 'similar values = statistically steady']]),
    r.notes?.length ? el('ul', { style: 'font-size:12px;padding-left:18px' }, r.notes.map(n => el('li', {}, n))) : null));
    const figs = FIGS.filter(([f]) => this.sum.files.includes(f));
    if (figs.length) {
      P.append(el('details', {}, el('summary', {}, 'Figures'), ...figs.map(([f, t]) => el('div', {}, el('div', { class: 'hint' }, t),
        el('img', { class: 'fig', src: `${this.base()}/file/${f}`, alt: t, onclick: () => modal(el('img', { src: `${this.base()}/file/${f}`, style: 'max-width:100%' }), { wide: true }) })))));
    }
  }

  focus(p) {
    const v = this.app.view;
    const off = v.camera.position.clone().sub(v.orbit.target);
    v.orbit.target.set(...p);
    v.camera.position.copy(v.orbit.target).add(off.setLength(Math.min(off.length(), 0.7)));
    v.orbit.update();
    v.render();
  }

  async reanalyze() {
    try { await api.post(`${this.base()}/analyze`); status('analysis queued (see the Run tab for progress)', 'ok'); this.app.runPanel.refreshJobs(); }
    catch (e) { status(e.message, 'err'); }
  }

  async deleteRun() {
    if (!confirm(`Delete run ${this.cur.rid} and all its files?`)) return;
    try { await api.del(this.base()); this.cur = null; this.sum = null; this.clearScene(); status('run deleted', 'ok'); await this.refreshRuns(); }
    catch (e) { status(e.message, 'err'); }
  }
}

// ------------------------------------------------------------------ comparison mode
export class Compare {
  constructor(app) { this.app = app; this.res = null; this.a = ''; this.b = ''; this.runs = []; }

  async render() {
    const app = this.app, P = app.panel, U = app.units;
    clear(P);
    P.append(el('h2', {}, 'Compare two runs'));
    P.append(el('div', { class: 'hint' }, 'The main design workflow: change one thing (move or re-aim a pump, edit the rock), run again, and compare. Differences are B − A; a change is flagged when it exceeds 2σ of the statistical noise of the two runs.'));
    try { this.runs = (await api.get('/api/runs')).runs.filter(r => r.has_fields); } catch (e) { this.runs = []; }
    const opts = [['', '— choose —'], ...this.runs.map(r => [`${r.project}/${r.id}`, `${r.project} · ${r.label}${r.has_analysis ? '' : ' (not analysed)'}`])];
    P.append(selectRow('Run A (baseline)', opts, this.a, (v) => { this.a = v; }));
    P.append(selectRow('Run B (change)', opts, this.b, (v) => { this.b = v; }));
    const la = el('input', { type: 'text', value: 'A', style: 'width:120px' }), lb = el('input', { type: 'text', value: 'B', style: 'width:120px' });
    P.append(el('div', { class: 'row' }, el('label', {}, 'Labels A / B'), el('span', {}, la, ' ', lb), el('span')));
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: async () => {
      if (!this.a || !this.b || this.a === this.b) { status('choose two different runs', 'err'); return; }
      try {
        this.res = await busy('comparing (rebuilding both tank models)', () => api.post('/api/compare', { a: this.a.split('/'), b: this.b.split('/'), label_a: la.value || 'A', label_b: lb.value || 'B' }));
        status('comparison ready', 'ok');
      } catch (e) { status(e.message, 'err'); }
      this.render();
    } }, 'Compare')));
    if (!this.res) return;
    const r = this.res.result;
    const conv = (v, unit) => unit === 'cm/s' ? [v * 0.01 / U.factor(U.unit('velocity')), U.unit('velocity')] : [v, unit];
    const t = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Metric'), el('th', {}, 'A'), el('th', {}, 'B'), el('th', {}, 'B − A'), el('th', {}, '> noise?')));
    for (const row of r.rows) {
      const [a, unit] = conv(row.A, row.unit), [b] = conv(row.B, row.unit), [d] = conv(row.delta, row.unit);
      t.append(el('tr', {}, el('td', {}, `${row.label} [${unit}]`), el('td', {}, num(a, 3)), el('td', {}, num(b, 3)),
        el('td', { style: row.significant ? 'font-weight:700' : '' }, `${d >= 0 ? '+' : ''}${num(d, 3)}${row.delta_pct !== null ? ` (${row.delta_pct >= 0 ? '+' : ''}${num(row.delta_pct, 2)} %)` : ''}`),
        el('td', {}, row.significant === null ? '' : row.significant ? 'yes' : 'no')));
    }
    for (const [k, [a, b]] of Object.entries(r.extra || {})) t.append(el('tr', {}, el('td', {}, k), el('td', {}, num(a, 3)), el('td', {}, num(b, 3)), el('td', {}, `${b - a >= 0 ? '+' : ''}${num(b - a, 3)}`), el('td')));
    P.append(el('h3', {}, 'Metric changes'), t);
    P.append(kvTable([['Dead water gained in B', `${num(100 * r.dead_gained_fraction, 3)} % of volume`], ['Dead water removed in B', `${num(100 * r.dead_removed_fraction, 3)} % of volume`],
      ['Mean |speed change|', U.fmt(r.speed_change_mean_abs_ms, 'velocity', 3)]]));
    P.append(el('div', { class: 'btns' }, el('a', { class: 'button', href: this.res.html, target: '_blank' }, 'Open comparison report')));
    P.append(el('img', { class: 'fig', src: this.res.image + '?t=' + Date.now(), onclick: () => modal(el('img', { src: this.res.image, style: 'max-width:100%' }), { wide: true }) }));
  }
}
