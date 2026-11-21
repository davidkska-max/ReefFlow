// Pumps (powerheads, return nozzles), overflow and light fixtures: 3-D markers + forms.
import * as THREE from 'three';
import { el, clear, numRow, vecRow, selectRow, textRow, checkRow, kvTable, num, deg, status } from './util.js';
import { textSprite, disposeTree, overflowBox, overflowWallLength } from './viewport.js';

const PUMP_COLOR = { powerhead: 0x1f77b4, return: 0x2ca02c };
const OVERFLOW_SEL = 0xff9f1c;

// ---------------------------------------------------------------- product catalogue (D-844)
// Library entries with product: true are real models (brand, model, sources); the others are
// generic archetypes. Pickers group them: generic first, then one group per brand.
const GENERIC = 'Generic (typical values, not a product)';
const pretty = (k) => k.replace(/_/g, ' ');
export function modelName(key, a) {
  if (!a) return key;
  return a.product ? `${a.brand} ${a.model}` : `generic ${pretty(key)}`;
}
export function modelSelect(entries, value, onChange, opts = {}) {
  const s = el('select', { title: opts.title || '', class: 'model' });
  if (opts.custom) {
    const op = el('option', { value: '' }, opts.custom);
    if (!value) op.selected = true;
    s.append(op);
  }
  const groups = new Map();
  for (const [k, a] of entries) {
    const g = a.product ? a.brand : GENERIC;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push([k, a]);
  }
  const order = [...groups.keys()].sort((x, y) => (x === GENERIC ? -1 : y === GENERIC ? 1 : x.localeCompare(y)));
  const label = ([k, a]) => (a.product ? a.model : pretty(k)) + (a.status === 'discontinued' ? ' (discontinued)' : '') + (opts.suffix ? opts.suffix(k, a) : '');
  for (const g of order) {
    const og = el('optgroup', { label: g });
    for (const e of groups.get(g).sort((p, q) => label(p).localeCompare(label(q), undefined, { numeric: true }))) {
      const op = el('option', { value: e[0], title: e[1].description || e[1].notes || '' }, label(e));
      if (e[0] === value) op.selected = true;
      og.append(op);
    }
    s.append(og);
  }
  if (onChange) s.addEventListener('change', () => onChange(s.value));
  return s;
}
function host(u) { try { return new URL(u).hostname.replace(/^www\./, ''); } catch { return u; } }
// Spec card of a real product: the numbers, where each came from, and the source links.
export function productCard(a, rows) {
  if (!a?.product) return null;
  const links = [];
  for (const u of a.sources || []) { if (links.length) links.push(' · '); links.push(el('a', { href: u, target: '_blank', rel: 'noopener' }, host(u))); }
  return el('div', { class: 'spec' },
    el('table', { class: 'kv spec' }, rows.filter(Boolean).map(([k, v, b]) => el('tr', {}, el('td', {}, k), el('td', {}, v, b ? el('div', { class: 'basis' }, b) : null)))),
    el('div', { class: 'hint' }, `${a.status === 'discontinued' ? 'Discontinued. ' : ''}Specs checked ${a.checked || '–'}${links.length ? ' · ' : ''}`, ...links),
    a.notes ? el('details', { class: 'hint' }, el('summary', {}, 'Notes on the data'), a.notes) : null);
}

// Orthonormal frame of a pump (same as reefflow.geometry.frame with the default 'up'):
// local x = aim, local z = up (world z projected), local y = up x aim.
export function pumpBasis(aim) {
  const a = new THREE.Vector3(...aim).normalize();
  const z = new THREE.Vector3(0, 0, 1);
  const ref = Math.abs(a.dot(z)) < 0.9 ? z : new THREE.Vector3(0, 1, 0);
  const up = ref.clone().addScaledVector(a, -ref.dot(a)).normalize();
  const lat = new THREE.Vector3().crossVectors(up, a);
  return new THREE.Matrix4().makeBasis(a, lat, up);
}

// Nozzle list of a pump (editor model). A pump without a list has one outlet at position/aim.
export function nozzlesOf(p) {
  if (!p.nozzles || !p.nozzles.length) p.nozzles = [{ position: p.position, aim: p.aim, diameter: null, share: null, extra: {} }];
  return p.nozzles;
}

// Set nozzle k's position and/or aim; nozzle 0 is mirrored into the pump's position/aim
// (the single-outlet project-file form uses those).
function setNozzle(p, k, pos, aim) {
  const n = nozzlesOf(p)[k];
  if (pos) n.position = pos;
  if (aim) n.aim = aim;
  if (k === 0) { p.position = n.position; p.aim = n.aim; }
}

function nozzleOutlet(p, k, lib) {
  const arch = lib.pumps[p.archetype] || { outlet: { shape: 'circle', diameter: 0.03 }, kind: 'powerhead' };
  const n = nozzlesOf(p)[k];
  return n?.diameter ? { shape: 'circle', diameter: n.diameter } : arch.outlet;
}

function nozzleObject(p, k, lib, opts = {}) {
  const arch = lib.pumps[p.archetype] || { outlet: { shape: 'circle', diameter: 0.03 }, kind: 'powerhead' };
  const kind = arch.kind;
  const n = nozzlesOf(p)[k], multi = p.nozzles.length > 1;
  const g = new THREE.Group();
  const color = PUMP_COLOR[kind] || 0x777777;
  const o = nozzleOutlet(p, k, lib);
  const r = o.shape === 'circle' ? o.diameter / 2 : Math.max(o.width, o.height) / 2;
  let disc;
  if (o.shape === 'circle') disc = new THREE.CircleGeometry(o.diameter / 2, 32);
  else disc = new THREE.PlaneGeometry(o.width, o.height);
  disc.rotateY(Math.PI / 2);                                   // face +x (the aim)
  g.add(new THREE.Mesh(disc, new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide })));
  // body (display only: housings are not modelled by default)
  if (kind === 'powerhead') {
    const bodyR = o.shape === 'circle' ? Math.max(o.diameter * 0.75, 0.025) : o.height * 0.9;
    const body = o.shape === 'circle' ? new THREE.CylinderGeometry(bodyR, bodyR, 0.07, 24) : new THREE.BoxGeometry(0.07, o.width * 1.05, o.height * 1.6);
    if (o.shape === 'circle') body.rotateZ(Math.PI / 2);
    body.translate(-0.036, 0, 0);
    g.add(new THREE.Mesh(body, new THREE.MeshStandardMaterial({ color: 0x9db2bf, transparent: true, opacity: 0.7 })));
  } else {
    const pipe = new THREE.CylinderGeometry(r * 1.2, r * 1.2, 0.05, 16); pipe.rotateZ(Math.PI / 2); pipe.translate(-0.026, 0, 0);
    g.add(new THREE.Mesh(pipe, new THREE.MeshStandardMaterial({ color: 0x6f9a7e })));
  }
  const len = opts.arrow ?? 0.16;
  g.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), len, color, 0.035, 0.022));
  g.add(textSprite(multi ? `${p.name} ${k + 1}` : p.name, 0, 0, Math.max(r, 0.02) + 0.03, 0.022, '#123'));
  g.position.set(...n.position);
  g.quaternion.setFromRotationMatrix(pumpBasis(n.aim));
  g.userData = { kind: 'pump', name: p.name, nozzle: k };
  return g;
}

// One object per outlet, plus a thin line linking the nozzles of one pump (manifold hint).
export function pumpObjects(p, lib, opts = {}) {
  const nz = nozzlesOf(p);
  const objs = nz.map((_, k) => nozzleObject(p, k, lib, opts));
  if (nz.length > 1) {
    const pts = nz.map(n => new THREE.Vector3(...n.position));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineDashedMaterial({ color: PUMP_COLOR.return, dashSize: 0.015, gapSize: 0.01 }));
    line.computeLineDistances();
    line.userData = { kind: 'manifold', name: p.name };
    objs.push(line);
  }
  return objs;
}

// Single object for a pump (all its nozzles; used by the Results view).
export function pumpObject(p, lib, opts = {}) {
  const objs = pumpObjects(p, lib, opts);
  if (objs.length === 1) return objs[0];
  const g = new THREE.Group();
  g.add(...objs);
  g.userData = { kind: 'pumpGroup', name: p.name };
  return g;
}

function overflowHandle(o, tank, selected) {
  const b = overflowBox(o, tank), sx = b.x1 - b.x0, sy = b.y1 - b.y0, top = tank.ws - o.crest_below_surface;
  const grp = new THREE.Group();
  const g = new THREE.BoxGeometry(sx + 0.004, sy + 0.004, top + 0.004);
  grp.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: OVERFLOW_SEL, transparent: true, opacity: selected ? 0.22 : 0.0, depthWrite: false })));
  if (selected) grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: OVERFLOW_SEL })));
  grp.position.set(b.x0 + sx / 2, b.y0 + sy / 2, top / 2);
  grp.userData = { kind: 'overflow', name: 'overflow' };
  return grp;
}

function fixtureRotation(f) {
  // Phase-4 convention: R = Ry(-tilt_y) Rx(tilt_x) Rz(rotate_z); beam axis = R (0, 0, -1)
  const m = new THREE.Matrix4().makeRotationY(-f.tilt_y * deg);
  m.multiply(new THREE.Matrix4().makeRotationX(f.tilt_x * deg));
  m.multiply(new THREE.Matrix4().makeRotationZ(f.rotate_z * deg));
  return m;
}

export function fixtureObject(f, lib, tank) {
  const a = lib.fixtures[f.archetype] || { footprint_m: [0.2, 0.2], beam_angle_deg: 90 };
  const g = new THREE.Group();
  const [fl, fw] = a.footprint_m.map(v => Math.max(v, 0.02));
  const body = new THREE.Mesh(new THREE.BoxGeometry(fl, fw, 0.02), new THREE.MeshStandardMaterial({ color: 0x4b5864, roughness: 0.6 }));
  body.position.z = 0.01;
  g.add(body);
  const face = new THREE.Mesh(new THREE.PlaneGeometry(fl * 0.9, fw * 0.9), new THREE.MeshBasicMaterial({ color: 0xfff3b0, side: THREE.DoubleSide }));
  face.position.z = -0.0005;
  g.add(face);
  const beam = (a.beam_angle_deg || 90) * deg;
  const h = Math.max(f.height, 0.01) / Math.max(Math.cos(Math.hypot(f.tilt_x, f.tilt_y) * deg), 0.2);
  const rad = Math.min(h * Math.tan(beam / 2), 2.0);
  const cone = new THREE.ConeGeometry(rad, h, 32, 1, true);
  cone.translate(0, -h / 2, 0);
  cone.rotateX(Math.PI / 2);                                  // apex at the fixture, opening toward local -z
  g.add(new THREE.Mesh(cone, new THREE.MeshBasicMaterial({ color: 0xffe066, transparent: true, opacity: 0.10, side: THREE.DoubleSide, depthWrite: false })));
  g.add(textSprite(f.name, 0, 0, 0.05, 0.022, '#6a5200'));
  g.position.set(f.position[0], f.position[1], tank.ws + f.height);
  g.quaternion.setFromRotationMatrix(fixtureRotation(f));
  g.userData = { kind: 'light', name: f.name };
  return g;
}

const r5 = (v) => +v.toFixed(5);

export class Equipment {
  constructor(app) {
    this.app = app;
    this.group = new THREE.Group();
    app.view.editorRoot.add(this.group);
    this.sel = null;      // {kind: 'pump'|'light'|'overflow', name, nozzle?}
    const tc = app.view.tc;
    tc.addEventListener('objectChange', () => this._fromGizmo(false));
    tc.addEventListener('mouseUp', () => this._fromGizmo(true));
  }

  get pumps() { return this.app.ed.pumps; }
  get fixtures() { return this.app.ed.lighting.fixtures; }

  sync() {
    for (const c of [...this.group.children]) { this.group.remove(c); disposeTree(c); }
    const t = this.app.view.tank;
    if (!t) return;
    for (const p of this.pumps) this.group.add(...pumpObjects(p, this.app.lib));
    for (const f of this.fixtures) this.group.add(fixtureObject(f, this.app.lib, t));
    if (this.app.ed.overflow) this.group.add(overflowHandle(this.app.ed.overflow, t, this.sel?.kind === 'overflow'));
    this._reattach();
    this.app.view.render();
  }

  _obj(sel) {
    if (!sel) return null;
    return this.group.children.find(o => o.userData.kind === sel.kind && o.userData.name === sel.name &&
      (sel.kind !== 'pump' || (o.userData.nozzle || 0) === (sel.nozzle || 0)));
  }

  _reattach() {
    const o = this._obj(this.sel);
    if (this.sel && !o) this.sel = null;
    if (this.app.tab !== 'equip') return;
    const tc = this.app.view.tc;
    tc.showX = tc.showY = tc.showZ = true;
    if (this.sel?.kind === 'overflow') {
      // slide along its own glass wall only
      const along = ['left', 'right'].includes(this.app.ed.overflow.wall) ? 'y' : 'x';
      tc.showX = along === 'x'; tc.showY = along === 'y'; tc.showZ = false;
      this.app.view.attach(o, 'translate');
      return;
    }
    const mode = this.sel?.kind === 'light' ? 'translate' : (this.app.gizmoMode === 'scale' ? 'translate' : this.app.gizmoMode);
    this.app.view.attach(o, mode);
  }

  pickables() { return this.group.children.filter(o => ['pump', 'light', 'overflow'].includes(o.userData.kind)); }

  select(sel) {
    const was = this.sel?.kind === 'overflow', now = sel?.kind === 'overflow';
    this.sel = sel ? { ...sel, nozzle: sel.kind === 'pump' ? (sel.nozzle || 0) : undefined } : null;
    if (was !== now) this.sync(); else this._reattach();
    if (this.app.tab === 'equip') this.renderPanel();
  }

  _clampOverflowCenter(o, c) {
    const Lw = overflowWallLength(o, this.app.view.tank);
    const w = Math.min(o.width, Lw);
    return Math.min(Math.max(c, w / 2), Lw - w / 2);
  }

  _fromGizmo(commit) {
    const o = this.app.view.tc.object;
    if (!o || !['pump', 'light', 'overflow'].includes(o.userData.kind)) return;
    const t = this.app.view.tank;
    if (o.userData.kind === 'pump') {
      const p = this.pumps.find(x => x.name === o.userData.name);
      if (!p) return;
      const k = o.userData.nozzle || 0;
      const a = new THREE.Vector3(1, 0, 0).applyQuaternion(o.quaternion).normalize();
      setNozzle(p, k, o.position.toArray().map(r5), a.toArray().map(v => +v.toFixed(4)));
      if (commit) o.quaternion.setFromRotationMatrix(pumpBasis(p.nozzles[k].aim));   // drop any roll
      if (commit && p.nozzles.length > 1) this.sync();                                 // redraw the manifold line
    } else if (o.userData.kind === 'overflow') {
      const ov = this.app.ed.overflow;
      if (!ov) return;
      const b = overflowBox(ov, t);
      const alongX = !['left', 'right'].includes(ov.wall);
      const c = this._clampOverflowCenter(ov, alongX ? o.position.x : o.position.y);
      ov.center = r5(c);
      // keep the box against its wall and at its height
      if (alongX) { o.position.x = c; o.position.y = (b.y0 + b.y1) / 2; } else { o.position.y = c; o.position.x = (b.x0 + b.x1) / 2; }
      o.position.z = (t.ws - ov.crest_below_surface) / 2;
      if (commit) { this.app.changed('move overflow'); this.app.refreshShell(); this.sync(); this.renderPanel(); }
      return;
    } else {
      const f = this.fixtures.find(x => x.name === o.userData.name);
      if (!f) return;
      f.position = [r5(o.position.x), r5(o.position.y)];
      f.height = +Math.max(o.position.z - t.ws, 0.005).toFixed(5);
      if (commit) o.position.z = t.ws + f.height;
    }
    if (commit) { this.app.changed('move'); this.renderPanel(); }
  }

  uniqueName(base, list) {
    let i = 1, n = base;
    const names = new Set(list.map(x => x.name));
    while (names.has(n)) n = `${base}-${++i}`;
    return n;
  }

  // Default spot for a return nozzle: just in front of the overflow's tank-side face,
  // 5 cm below the surface, aimed away from the overflow and slightly down.
  // offset: shift along the overflow's wall (for extra nozzles).
  returnNozzleSpot(offset = 0) {
    const t = this.app.view.tank, o = this.app.ed.overflow;
    const z = t.ws - 0.05;
    if (!o) return { position: [0.06, t.W * 0.6, z], aim: [1, 0, -0.1] };
    const b = overflowBox(o, t), gap = 0.03;
    const clampX = (x) => Math.min(Math.max(x, 0.02), t.L - 0.02), clampY = (y) => Math.min(Math.max(y, 0.02), t.W - 0.02);
    const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    const side = (v, mid) => (v < mid ? 1 : -1);
    switch (o.wall) {
      case 'front': return { position: [clampX(cx + offset), clampY(b.y1 + gap), z], aim: [0.3 * side(cx, t.L / 2), 1, -0.25] };
      case 'left': return { position: [clampX(b.x1 + gap), clampY(cy + offset), z], aim: [1, 0.3 * side(cy, t.W / 2), -0.25] };
      case 'right': return { position: [clampX(b.x0 - gap), clampY(cy + offset), z], aim: [-1, 0.3 * side(cy, t.W / 2), -0.25] };
      default: return { position: [clampX(cx + offset), clampY(b.y0 - gap), z], aim: [0.3 * side(cx, t.L / 2), -1, -0.25] };
    }
  }

  addPump(archetype) {
    const a = this.app.lib.pumps[archetype], t = this.app.view.tank;
    const isRet = a.kind === 'return';
    const spot = isRet ? this.returnNozzleSpot() : { position: [0.06, t.W * 0.6, t.sand + 0.75 * (t.ws - t.sand)], aim: [1, 0, -0.1] };
    const p = {
      name: this.uniqueName(isRet ? 'return' : 'PH', this.pumps), archetype, kind: a.kind, rated_flow: a.rated_flow,
      derating: a.derating, position: spot.position, aim: spot.aim,
      mode: { type: 'constant', period: 10, min: 0.3, max: 1, duty: 0.5, ramp: 0.1, phase: 0 }, turbulence: a.turbulence,
      static_head: a.static_head, fittings_allowance: a.fittings_allowance, delivered_flow: null, extra: {},
    };
    p.nozzles = [{ position: p.position, aim: p.aim, diameter: null, share: null, extra: {} }];
    this.pumps.push(p);
    this.app.changed('add pump');
    this.sync();
    this.select({ kind: 'pump', name: p.name, nozzle: 0 });
  }

  addNozzle(p) {
    const nz = nozzlesOf(p);
    const last = nz[nz.length - 1];
    const o = this.app.ed.overflow;
    let spot;
    if (o) {
      // next nozzle 8 cm further along the overflow's wall, stepping toward the middle of the tank
      const t = this.app.view.tank, b = overflowBox(o, t);
      const alongX = !['left', 'right'].includes(o.wall);
      const toMid = alongX ? Math.sign(t.L / 2 - (b.x0 + b.x1) / 2) || 1 : Math.sign(t.W / 2 - (b.y0 + b.y1) / 2) || 1;
      spot = this.returnNozzleSpot(toMid * 0.08 * nz.length);
    } else {
      const t = this.app.view.tank;
      spot = { position: [Math.min(last.position[0] + 0.08, t.L - 0.02), last.position[1], last.position[2]], aim: [...last.aim] };
    }
    nz.push({ position: spot.position.map(r5), aim: spot.aim, diameter: last.diameter ?? null, share: null, extra: {} });
    this.app.changed('add nozzle');
    this.sync();
    this.select({ kind: 'pump', name: p.name, nozzle: nz.length - 1 });
  }

  removeNozzle(p, k) {
    const nz = nozzlesOf(p);
    if (nz.length <= 1) { status('a pump needs at least one outlet', 'err'); return; }
    nz.splice(k, 1);
    setNozzle(p, 0);
    this.app.changed('delete nozzle');
    this.sync();
    this.select({ kind: 'pump', name: p.name, nozzle: Math.min(k, nz.length - 1) });
  }

  addFixture(archetype) {
    const t = this.app.view.tank;
    const f = { name: this.uniqueName('light', this.fixtures), archetype, position: [t.L / 2, t.W / 2], height: 0.2032,
      tilt_x: 0, tilt_y: 0, rotate_z: 0, intensity: 100, extra: {} };
    this.fixtures.push(f);
    this.app.changed('add light');
    this.sync();
    this.select({ kind: 'light', name: f.name });
  }

  duplicate() {
    if (!this.sel || this.sel.kind === 'overflow') return;
    if (this.sel.kind === 'pump') {
      const s = this.pumps.find(x => x.name === this.sel.name);
      const p = JSON.parse(JSON.stringify(s));
      p.name = this.uniqueName(s.name, this.pumps);
      const W = this.app.view.tank.W;
      for (const n of nozzlesOf(p)) n.position = [n.position[0], Math.min(n.position[1] + 0.05, W - 0.02), n.position[2]];
      setNozzle(p, 0);
      this.pumps.push(p);
      this.app.changed('duplicate pump');
      this.sync(); this.select({ kind: 'pump', name: p.name, nozzle: 0 });
    } else {
      const s = this.fixtures.find(x => x.name === this.sel.name);
      const f = JSON.parse(JSON.stringify(s));
      f.name = this.uniqueName(s.name, this.fixtures);
      f.position = [Math.min(s.position[0] + 0.2, this.app.view.tank.L), s.position[1]];
      this.fixtures.push(f);
      this.app.changed('duplicate light');
      this.sync(); this.select({ kind: 'light', name: f.name });
    }
  }

  remove() {
    if (!this.sel) return;
    if (this.sel.kind === 'overflow') {
      if (this.pumps.some(p => p.kind === 'return') && !confirm('Remove the overflow? A return pump needs one.')) return;
      this.app.ed.overflow = null;
      this.sel = null;
      this.app.changed('overflow'); this.app.refreshShell(); this.sync(); this.renderPanel();
      return;
    }
    if (this.sel.kind === 'pump') {
      const p = this.pumps.find(x => x.name === this.sel.name);
      if (p && nozzlesOf(p).length > 1) { this.removeNozzle(p, this.sel.nozzle || 0); return; }
    }
    const list = this.sel.kind === 'pump' ? this.pumps : this.fixtures;
    const i = list.findIndex(x => x.name === this.sel.name);
    if (i >= 0) list.splice(i, 1);
    this.sel = null;
    this.app.changed('delete');
    this.sync();
    this.renderPanel();
  }

  snapToGlass() {
    const p = this.pumps.find(x => this.sel?.kind === 'pump' && x.name === this.sel.name);
    if (!p) return;
    const k = this.sel.nozzle || 0, n = nozzlesOf(p)[k];
    const t = this.app.view.tank, off = 0.045;      // outlet ~45 mm from the glass (typical magnet mount)
    const pos = [...n.position];
    let aim = [...n.aim];
    const d = [pos[0], t.L - pos[0], pos[1], t.W - pos[1]];
    const i = d.indexOf(Math.min(...d));
    if (i === 0) { pos[0] = off; aim = [1, 0, aim[2]]; }
    if (i === 1) { pos[0] = t.L - off; aim = [-1, 0, aim[2]]; }
    if (i === 2) { pos[1] = off; aim = [0, 1, aim[2]]; }
    if (i === 3) { pos[1] = t.W - off; aim = [0, -1, aim[2]]; }
    setNozzle(p, k, pos, aim);
    this.app.changed('snap pump');
    this.sync(); this.renderPanel();
  }

  // ---------------------------------------------------------------- panel
  renderPanel() {
    const app = this.app, P = app.panel, U = app.units, lib = app.lib;
    clear(P);
    P.append(el('h2', {}, 'Pumps, overflow and lights'));
    P.append(el('div', { class: 'hint' }, 'Click a pump nozzle, the overflow or a light in the view to select it; drag the gizmo to move it (Rotate mode aims a nozzle; the overflow slides along its glass). Numbers can be typed below.'));

    // pumps
    const addSel = modelSelect(Object.entries(lib.pumps), 'narrow_jet_propeller', null, { suffix: (k, a) => (a.kind === 'return' ? ' · return' : ''), title: 'Pick a real pump to use its published specs, or a generic one' });
    P.append(el('h3', {}, `Pumps (${this.pumps.length})`, el('span', {}, addSel, ' ', el('button', { class: 'small', onclick: () => this.addPump(addSel.value) }, '+ Add'))));
    const summ = Object.fromEntries((app.validation?.pumps || []).map(s => [s.name, s]));
    const list = el('div', { class: 'list' });
    if (!this.pumps.length) list.append(el('div', { class: 'empty' }, 'No pumps yet.'));
    for (const p of this.pumps) {
      const s = summ[p.name], nn = nozzlesOf(p).length;
      list.append(el('div', { class: 'item' + (this.sel?.kind === 'pump' && this.sel.name === p.name ? ' sel' : ''), onclick: () => this.select({ kind: 'pump', name: p.name, nozzle: 0 }) },
        el('span', {}, p.name), el('span', { class: 'sub' }, `${modelName(p.archetype, lib.pumps[p.archetype])}${nn > 1 ? ` · ${nn} nozzles` : ''}${s ? ' · ' + U.fmt(s.delivered_flow_m3s, 'flow', 3) + ' delivered' : ''}`)));
    }
    P.append(list);
    if (this.sel?.kind === 'pump') {
      const p = this.pumps.find(x => x.name === this.sel.name);
      if (p) this._pumpForm(P, p, summ[p.name]);
    }

    this._overflowForm(P);

    // lights
    const lt = modelSelect(Object.entries(lib.fixtures).filter(([, a]) => !a.verification_only), 'led_puck', null, { title: 'Pick a real fixture to use its published specs, or a generic one' });
    P.append(el('h3', {}, `Light fixtures (${this.fixtures.length})`, el('span', {}, lt, ' ', el('button', { class: 'small', onclick: () => this.addFixture(lt.value) }, '+ Add'))));
    if (!lib.lighting_available) P.append(el('div', { class: 'warnbox' }, 'The lighting module (Phase 4) is not installed: fixtures are stored in the project but PAR cannot be computed yet.'));
    const fl = el('div', { class: 'list' });
    if (!this.fixtures.length) fl.append(el('div', { class: 'empty' }, 'No lights yet.'));
    for (const f of this.fixtures) {
      fl.append(el('div', { class: 'item' + (this.sel?.kind === 'light' && this.sel.name === f.name ? ' sel' : ''), onclick: () => this.select({ kind: 'light', name: f.name }) },
        el('span', {}, f.name), el('span', { class: 'sub' }, `${modelName(f.archetype, lib.fixtures[f.archetype])} · ${U.fmt(f.height, 'length', 3)} above water`)));
    }
    P.append(fl);
    if (this.sel?.kind === 'light') {
      const f = this.fixtures.find(x => x.name === this.sel.name);
      if (f) this._fixtureForm(P, f);
    }
  }

  _overflowForm(P) {
    const app = this.app, U = app.units, lib = app.lib, t = app.view.tank;
    const o = app.ed.overflow;
    P.append(el('h3', {}, 'Overflow (where the return water leaves)', o ? el('span', {},
      el('button', { class: 'small' + (this.sel?.kind === 'overflow' ? ' on' : ''), onclick: () => this.select(this.sel?.kind === 'overflow' ? null : { kind: 'overflow', name: 'overflow' }), title: 'Select the box to drag it along its glass in the view' }, 'Drag in view')) : ''));
    P.append(checkRow('Tank has an overflow', !!o, (v) => {
      app.ed.overflow = v ? { wall: 'back', center: t ? t.L - 0.0762 : 0.5, width: 0.1524, depth: 0.1524, crest_below_surface: 0.0127 } : null;
      if (!v && this.sel?.kind === 'overflow') this.sel = null;
      app.changed('overflow'); app.refreshShell(); this.sync(); this.renderPanel();
    }));
    if (!o || !t) return;
    const ch = (rerender = false) => { app.changed('overflow'); app.refreshShell(); this.sync(); if (rerender) this.renderPanel(); };
    const Lw = () => overflowWallLength(o, t);
    const ovs = lib.overflows || {}, op = o.product ? ovs[o.product] : null;
    if (Object.keys(ovs).length) {
      P.append(el('div', { class: 'row' }, el('label', {}, 'Model'), modelSelect(Object.entries(ovs), o.product || '', (v) => {
        o.product = v || null;
        const q = ovs[v];
        if (q) {
          if (q.width) o.width = Math.min(q.width, Lw());
          if (q.depth) o.depth = q.depth;
          o.capacity = q.capacity || null;
          o.center = r5(this._clampOverflowCenter(o, o.center));
        }
        ch(true);
      }, { custom: 'Custom size', title: 'A real overflow box sets the size and its rated capacity' }), el('span')));
      if (op) P.append(productCard(op, [
        ['Box (width × depth)', [op.width, op.depth].filter(Boolean).map(v => U.fmt(v, 'length', 3)).join(' × ') || '–', op.height ? `height ${U.fmt(op.height, 'length', 3)}` : null],
        ['Rated capacity', op.capacity ? U.fmt(op.capacity, 'flow', 4) : 'not published', op.rated_flow_basis],
        op.drains || op.design ? ['Drains', [op.design, op.drains].filter(Boolean).join(': ')] : null]));
    }
    const along = ['left', 'right'].includes(o.wall) ? 'front' : 'left';
    P.append(selectRow('On the glass', lib.overflow_walls || [['back', 'back glass'], ['front', 'front glass'], ['left', 'left glass'], ['right', 'right glass']], o.wall, (v) => {
      const frac = o.center / Lw();
      o.wall = v;
      o.width = Math.min(o.width, Lw());
      o.center = r5(this._clampOverflowCenter(o, frac * Lw()));
      ch(true);
    }));
    P.append(numRow(`Centre, from the ${along} glass`, U, 'length', () => o.center, (v) => { o.center = r5(this._clampOverflowCenter(o, v)); ch(true); },
      { min: 0, title: `Position of the box centre along the ${o.wall} glass (clamped so the box stays inside)` }));
    P.append(numRow('Width (along the glass)', U, 'length', () => o.width, (v) => {
      o.width = Math.min(Math.max(v, 0.01), Lw()); o.center = r5(this._clampOverflowCenter(o, o.center)); ch(true);
    }, { min: 0, title: 'Equal to the glass length = coast-to-coast overflow' }));
    P.append(numRow('Depth (out from the glass)', U, 'length', () => o.depth, (v) => { o.depth = Math.max(v, 0.01); ch(); }, { min: 0 }));
    P.append(numRow('Weir crest below water', U, 'length', () => o.crest_below_surface, (v) => { o.crest_below_surface = v; ch(); }, { min: 0 }));
    P.append(numRow('Drain capacity', U, 'flow', () => o.capacity, (v) => { o.capacity = v || null; ch(); }, { allowEmpty: true, placeholder: 'unknown', min: 0,
      title: 'Most flow the overflow and its drains can take. Only used to warn when the return flow is higher; the flow model does not limit it.' }));
    const dc = lib.drain_capacity || [];
    if (dc.length) P.append(el('details', {}, el('summary', {}, 'Typical drain capacity by pipe size'),
      el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Pipe'), el('th', {}, 'Full siphon'), el('th', {}, 'Gravity (Durso)'), el('th', {}, 'Source')),
        dc.map(r => el('tr', { title: r.notes || '' }, el('td', {}, r.pipe), el('td', {}, r.siphon || '–'), el('td', {}, r.gravity || '–'), el('td', {}, host(r.source || '')))))));
    const [l, r] = ['left', 'right'].includes(o.wall) ? ['front', 'back'] : ['left', 'right'];
    P.append(el('div', { class: 'btns' },
      el('button', { class: 'small', onclick: () => { o.center = r5(o.width / 2); ch(true); } }, `Flush ${l} corner`),
      el('button', { class: 'small', onclick: () => { o.center = r5(Lw() / 2); ch(true); } }, 'Centre of glass'),
      el('button', { class: 'small', onclick: () => { o.center = r5(Lw() - o.width / 2); ch(true); } }, `Flush ${r} corner`),
      el('button', { class: 'small', onclick: () => { o.width = Lw(); o.center = r5(Lw() / 2); ch(true); }, title: 'Full-length overflow along this glass' }, 'Coast to coast')));
  }

  _pumpForm(P, p, s) {
    const app = this.app, U = app.units, lib = app.lib;
    const a = lib.pumps[p.archetype];
    const ch = (why, rebuild = true) => { app.changed(why); if (rebuild) this.sync(); };
    const nz = nozzlesOf(p), multi = nz.length > 1;
    const k = Math.min(this.sel.nozzle || 0, nz.length - 1), n = nz[k];
    P.append(el('h3', {}, `Pump: ${p.name}`, el('span', {},
      el('button', { class: 'small', onclick: () => this.snapToGlass(), title: 'Move the selected outlet 45 mm off the nearest side glass, aimed across the tank' }, 'Snap to glass'), ' ',
      el('button', { class: 'small', onclick: () => this.duplicate() }, 'Duplicate'), ' ',
      el('button', { class: 'small danger', onclick: () => { this.sel = { kind: 'pump', name: p.name, nozzle: 0 }; const keep = p.nozzles; p.nozzles = [keep[0]]; this.remove(); } }, 'Delete pump'))));
    P.append(textRow('Name', p.name, (v) => {
      if (!v || this.pumps.some(x => x !== p && x.name === v)) { status('pump names must be unique and not empty', 'err'); return; }
      p.name = v; this.sel = { kind: 'pump', name: v, nozzle: k }; ch('rename'); this.renderPanel();
    }));
    P.append(el('div', { class: 'row' }, el('label', {}, 'Model'), modelSelect(Object.entries(lib.pumps), p.archetype, (v) => {
      const na = lib.pumps[v], wasReturn = p.kind === 'return';
      p.archetype = v; p.kind = na.kind; p.rated_flow = na.rated_flow; p.derating = na.derating; p.turbulence = na.turbulence;
      if (!(wasReturn && na.kind === 'return')) { p.static_head = na.static_head; p.fittings_allowance = na.fittings_allowance; }  // keep your plumbing
      if (na.kind !== 'return' && p.nozzles.length > 1) { p.nozzles = [p.nozzles[0]]; status('a powerhead has one outlet: extra nozzles removed', 'err'); }
      this.sel.nozzle = 0;
      ch('archetype'); this.renderPanel();
    }, { suffix: (k2, a2) => (a2.kind === 'return' ? ' · return' : '') }), el('span')));
    const outTxt = a ? (a.outlet.shape === 'circle' ? 'Ø ' + U.fmt(a.outlet.diameter, 'length', 3) : U.fmt(a.outlet.width, 'length', 3) + ' × ' + U.fmt(a.outlet.height, 'length', 3)) : '';
    if (a?.product) {
      P.append(productCard(a, [
        [a.kind === 'return' ? 'Max flow (0 head)' : 'Rated flow', U.fmt(a.rated_flow, 'flow', 4), a.rated_flow_basis],
        a.max_head ? ['Max head', U.fmt(a.max_head, 'head', 3), a.curve?.length ? `curve: ${a.curve.length} points (${a.curve_basis || 'manufacturer'})` : (a.curve_basis || 'no published curve: generic curve shape')] : null,
        a.power_w ? ['Power (max)', `${num(a.power_w, 3)} W`] : null,
        [a.kind === 'return' ? 'Nozzle in the tank' : 'Outlet', outTxt, a.outlet_basis],
      ]));
    } else if (a) P.append(el('div', { class: 'hint' }, `${a.description} Outlet: ${outTxt}.`));

    // nozzles (return pumps)
    if (p.kind === 'return') {
      P.append(el('h3', {}, `Nozzles (${nz.length})`, el('span', {}, el('button', { class: 'small', onclick: () => this.addNozzle(p), title: 'Another outlet on the same return line (the pump flow is shared)' }, '+ Add nozzle'))));
      const sn = s?.nozzles || [];
      const tb = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, '#'), el('th', {}, 'Outlet'), el('th', {}, 'Share'), el('th', {}, 'Flow'), el('th', {}, 'Velocity'), el('th', { title: 'Exit velocity head U²/2g: assumed included in the fittings allowance' }, 'Exit head'), el('th', {})));
      nz.forEach((q, i) => {
        const r = sn[i];
        const o = nozzleOutlet(p, i, lib);
        tb.append(el('tr', { class: 'click' + (i === k ? ' sel' : ''), onclick: () => this.select({ kind: 'pump', name: p.name, nozzle: i }) },
          el('td', {}, String(i + 1)),
          el('td', {}, o.shape === 'circle' ? 'Ø ' + U.fmt(o.diameter, 'length', 3) : U.fmt(o.width, 'length', 2) + '×' + U.fmt(o.height, 'length', 2)),
          el('td', {}, r ? `${num(100 * r.share, 3)} %${q.share == null && multi ? ' (auto)' : ''}` : '–'),
          el('td', {}, r ? U.fmt(r.flow_m3s, 'flow', 3) : '–'),
          el('td', {}, r ? `${num(r.velocity_ms, 3)} m/s` : '–'),
          el('td', {}, r ? U.fmt(r.exit_velocity_head_m, 'head', 2) : '–'),
          el('td', {}, multi ? el('button', { class: 'small', title: 'Delete this nozzle', onclick: (ev) => { ev.stopPropagation(); this.removeNozzle(p, i); } }, '×') : '')));
      });
      P.append(tb);
      if (!s) P.append(el('div', { class: 'hint' }, 'Flows appear after the project validates.'));
      P.append(el('div', { class: 'hint' }, multi ? `Editing nozzle ${k + 1} (click a row or a nozzle in the view).` : 'Add nozzles to split this return over several outlets.'));
    }

    const lab = multi ? `Nozzle ${k + 1} position x, y, z` : 'Outlet position x, y, z';
    P.append(vecRow(lab, U, 'length', () => n.position, (v) => { setNozzle(p, k, v); ch('move'); }));
    const az = Math.atan2(n.aim[1], n.aim[0]) / deg, elv = Math.asin(Math.max(-1, Math.min(1, n.aim[2] / Math.hypot(...n.aim)))) / deg;
    const setAim = (azd, eld) => { setNozzle(p, k, null, [Math.cos(eld * deg) * Math.cos(azd * deg), Math.cos(eld * deg) * Math.sin(azd * deg), Math.sin(eld * deg)].map(v => +v.toFixed(4))); ch('aim'); this.renderPanel(); };
    P.append(numRow('Aim azimuth (plan)', U, null, () => +az.toFixed(1), (v) => setAim(v, elv), { unit: 'deg', step: 5, title: '0 = toward +x (right), 90 = toward the back glass, 180 = left, -90 = toward the front glass' }));
    P.append(numRow('Aim elevation', U, null, () => +elv.toFixed(1), (v) => setAim(az, v), { unit: 'deg', step: 5, min: -89, max: 89, title: '+ up, - down' }));
    P.append(el('div', { class: 'hint num' }, `aim vector [${n.aim.map(v => v.toFixed(3)).join(', ')}]`));
    if (p.kind === 'return') {
      P.append(numRow(`Nozzle ${multi ? k + 1 + ' ' : ''}diameter override`, U, 'length', () => n.diameter, (v) => { n.diameter = v || null; ch('nozzle', true); }, { allowEmpty: true, placeholder: 'archetype', min: 0, title: 'Leave empty to use the archetype outlet' }));
      if (multi) P.append(numRow(`Nozzle ${k + 1} flow share`, U, null, () => n.share, (v) => { n.share = v || null; ch('share', false); }, { allowEmpty: true, placeholder: 'auto', step: 0.05, min: 0, max: 1, title: 'Fraction of the pump flow. Empty = split the rest in proportion to outlet area (common manifold pressure)' }));
    }

    P.append(numRow('Rated flow', U, 'flow', () => p.rated_flow, (v) => { p.rated_flow = v; ch('flow', false); }, { min: 0, title: a?.kind === 'return' ? 'Pump zero-head flow Q0 (curve Q = Q0 sqrt(1 - H/H0))' : 'Manufacturer free-flow rating' }));
    if (p.kind === 'powerhead') {
      P.append(numRow('Derating (delivered / rated)', U, null, () => p.derating, (v) => { p.derating = v; ch('derating', false); }, { step: 0.05, min: 0.05, max: 1 }));
      P.append(el('div', { class: 'note' }, 'Manufacturer ratings are free-flow values and overstate what a powerhead delivers in a tank. Delivered flow = rated × derating (default 0.7, an editable starting assumption; the report shows the sensitivity to ±20 %).'));
    } else {
      P.append(numRow('Static head (sump to tank)', U, 'head', () => p.static_head, (v) => { p.static_head = v; ch('head', false); }, { min: 0 }));
      P.append(numRow('Fittings allowance', U, 'head', () => p.fittings_allowance, (v) => { p.fittings_allowance = v; ch('head', false); }, { min: 0, title: 'pipe, check valve, elbows, manifold and the nozzle exit velocity head' }));
      P.append(el('div', { class: 'note' }, 'The pump-curve flow uses static head + fittings allowance. The nozzle exit velocity head (table above) is assumed to be part of the fittings allowance, so adding nozzles does not change the total flow by itself; lower the allowance if more nozzles reduce your losses.'));
    }
    P.append(numRow('Delivered flow override', U, 'flow', () => p.delivered_flow, (v) => { p.delivered_flow = v || null; ch('override', false); }, { allowEmpty: true, placeholder: 'auto', title: 'Leave empty to compute it; enter a measured value to override' }));
    const m = p.mode;
    P.append(selectRow('Mode', [['constant', 'constant'], ['sine', 'sine (smooth wave)'], ['pulse', 'pulse (square, ramped)']], m.type, (v) => { m.type = v; ch('mode', false); this.renderPanel(); }));
    if (m.type !== 'constant') {
      P.append(numRow('Period', U, null, () => m.period, (v) => { m.period = v; ch('mode', false); }, { unit: 's', step: 0.5, min: 0.1 }));
      P.append(numRow('Min flow fraction', U, null, () => m.min, (v) => { m.min = v; ch('mode', false); }, { step: 0.05, min: 0, max: 1 }));
      P.append(numRow('Max flow fraction', U, null, () => m.max, (v) => { m.max = v; ch('mode', false); }, { step: 0.05, min: 0, max: 1 }));
      P.append(numRow('Phase (0-1)', U, null, () => m.phase, (v) => { m.phase = v; ch('mode', false); }, { step: 0.25, min: 0, max: 1, title: 'Two pumps with the same mode and phases 0 and 0.5 alternate (wave)' }));
      if (m.type === 'pulse') {
        P.append(numRow('Duty (on share)', U, null, () => m.duty, (v) => { m.duty = v; ch('mode', false); }, { step: 0.05, min: 0.05, max: 0.95 }));
        P.append(numRow('Ramp (share of period)', U, null, () => m.ramp, (v) => { m.ramp = v; ch('mode', false); }, { step: 0.02, min: 0, max: 0.45 }));
      }
    }
    P.append(numRow('Inflow turbulence intensity', U, null, () => p.turbulence, (v) => { p.turbulence = v; ch('turb', false); }, { step: 0.01, min: 0, max: 0.5, title: '0.05 reproduces the turbulent round-jet far field (verification 7)' }));
    if (s) {
      P.append(kvTable([
        ['Delivered flow' + (multi ? ' (all nozzles)' : ''), `${U.fmt(s.delivered_flow_m3s, 'flow', 4)}`, s.flow_basis],
        ...(multi ? [] : [['Nozzle velocity', `${num(s.nozzle_velocity_ms, 3)} m/s`]]),
        ['Momentum flux J = ρQ²/A' + (multi ? ' (sum)' : ''), `${num(s.momentum_flux_N, 3)} N`],
        ['Mode', s.mode]]));
    }
  }

  // Channel % settings (kept in 'extra') that the new model does not have would fail validation: drop them.
  _keepChannels(f) {
    const known = new Set(this.app.lib.fixtures[f.archetype]?.channels || []);
    const c = f.extra?.channels;
    if (!c || typeof c !== 'object') return;
    const dropped = Object.keys(c).filter(k => !known.has(k));
    for (const k of dropped) delete c[k];
    if (!Object.keys(c).length) delete f.extra.channels;
    if (dropped.length) status(`channel settings ${dropped.join(', ')} do not exist on this model: removed`, 'ok');
  }

  _fixtureForm(P, f) {
    const app = this.app, U = app.units, lib = app.lib;
    const a = lib.fixtures[f.archetype] || {};
    const ch = (why) => { app.changed(why); this.sync(); };
    P.append(el('h3', {}, `Light: ${f.name}`, el('span', {},
      el('button', { class: 'small', onclick: () => this.duplicate() }, 'Duplicate'), ' ',
      el('button', { class: 'small danger', onclick: () => this.remove() }, 'Delete'))));
    P.append(textRow('Name', f.name, (v) => {
      if (!v || this.fixtures.some(x => x !== f && x.name === v)) { status('fixture names must be unique and not empty', 'err'); return; }
      f.name = v; this.sel = { kind: 'light', name: v }; ch('rename'); this.renderPanel();
    }));
    P.append(el('div', { class: 'row' }, el('label', {}, 'Model'), modelSelect(Object.entries(lib.fixtures).filter(([, x]) => !x.verification_only), f.archetype, (v) => { f.archetype = v; this._keepChannels(f); ch('archetype'); this.renderPanel(); }), el('span')));
    if (a.product) {
      const par = (a.par_data || []).filter(r => r.par != null || r.par_range).map(r => [`Published PAR${r.position ? ' (' + r.position + ')' : ''}`, r.par != null ? `${r.par} µmol/m²/s` : `${r.par_range.join('–')} µmol/m²/s`,
        [r.depth ? `depth: ${r.depth}` : null, r.mount_height ? `light height: ${r.mount_height}` : null, r.settings ? `settings: ${r.settings}` : null, r.source ? `source: ${r.source}` : null].filter(Boolean).join(' · ')]);
      P.append(productCard(a, [
        a.power_w ? ['Power (max)', `${num(a.power_w, 3)} W`] : null,
        ['Photon output at 100 %', a.ppf_umol_s ? `${num(a.ppf_umol_s, 3)} µmol/s` : '–', a.ppf_basis],
        ...par]));
    } else if (a.description) P.append(el('div', { class: 'hint' }, a.description));
    P.append(vecRow('Centre over tank x, y', U, 'length', () => f.position, (v) => { f.position = v; ch('move'); }, { n: 2, axes: 'xy' }));
    P.append(numRow('Height above water', U, 'length', () => f.height, (v) => { f.height = v; ch('height'); }, { min: 0.001, title: 'Emitting face above the water surface' }));
    P.append(numRow('Tilt toward back glass', U, null, () => f.tilt_x, (v) => { f.tilt_x = v; ch('tilt'); }, { unit: 'deg', step: 5, min: -80, max: 80 }));
    P.append(numRow('Tilt toward right (+x)', U, null, () => f.tilt_y, (v) => { f.tilt_y = v; ch('tilt'); }, { unit: 'deg', step: 5, min: -80, max: 80 }));
    P.append(numRow('Footprint rotation', U, null, () => f.rotate_z, (v) => { f.rotate_z = v; ch('rotate'); }, { unit: 'deg', step: 15 }));
    P.append(numRow('Master intensity', U, null, () => f.intensity, (v) => { f.intensity = v; ch('intensity'); }, { unit: '%', step: 5, min: 0, max: 100 }));
    if (f.extra && Object.keys(f.extra).length) P.append(el('div', { class: 'hint num' }, 'other settings kept from the file: ' + JSON.stringify(f.extra)));
    P.append(kvTable([
      ['Footprint', a.footprint_m ? a.footprint_m.map(v => U.fmt(v, 'length', 3)).join(' × ') : '–'],
      ['Beam angle (FWHM)', a.beam_angle_deg ? `${num(a.beam_angle_deg, 3)}°` : '–'],
      ['Photon output at 100 %', a.ppf_umol_s ? `${num(a.ppf_umol_s, 3)} µmol/s` : '–']]));
    P.append(el('div', { class: 'note' }, 'Channel settings, photoperiod, water clarity, glass and calibration are edited in the project file (YAML) for now; the UI keeps them unchanged.'));
  }
}
