// Rockscape designer: placeable rock elements. Geometry comes from the server
// (reefflow.ui.rockgen, same code as the union), so what you see is what gets built.
import * as THREE from 'three';
import { api, el, clear, numRow, vecRow, selectRow, textRow, status, busy, uid, deg, num, kvTable } from './util.js';
import { meshFromBuffers, rockMaterial, disposeTree } from './viewport.js';
import { Sculpt } from './sculpt.js';
import { ScanImport } from './scan.js';

const TYPE_LABEL = { rock: 'Rock', slab: 'Slab / shelf', pillar: 'Pillar', arch: 'Arch', bridge: 'Bridge', mesh: 'Imported mesh' };
const SEL_EMISSIVE = 0x335566;

export class Designer {
  constructor(app) {
    this.app = app;
    this.group = new THREE.Group();
    app.view.editorRoot.add(this.group);
    this.objs = new Map();          // element id -> Object3D
    this.geoCache = new Map();      // key -> {positions, index}
    this.sel = null;
    this.lastReport = null;
    const tc = app.view.tc;
    tc.addEventListener('objectChange', () => this._fromGizmo(false));
    tc.addEventListener('mouseUp', () => this._fromGizmo(true));
    this.sculpt = new Sculpt(app);
    this.scan = new ScanImport(app, this);
  }

  get dz() { return this.app.ed.designer; }
  get els() { return this.dz.elements; }

  // ---------------------------------------------------------------- scene sync
  async sync() {
    const ids = new Set(this.els.map(e => e.id));
    for (const [id, o] of this.objs) if (!ids.has(id)) { this.group.remove(o); this.objs.delete(id); }
    await Promise.all(this.els.map(e => this._ensure(e)));
    if (this.sel && !ids.has(this.sel)) this.select(null);
    this.app.view.render();
  }

  _key(e) { return JSON.stringify({ t: e.type, p: e.params }); }

  async _geometry(e) {
    const k = this._key(e);
    if (!this.geoCache.has(k)) {
      const p = api.mesh(`/api/projects/${this.app.pid}/element_mesh`, { element: { type: e.type, params: e.params } });
      this.geoCache.set(k, p);
      p.catch(() => this.geoCache.delete(k));
    }
    return this.geoCache.get(k);
  }

  async _ensure(e) {
    let o = this.objs.get(e.id);
    const k = this._key(e);
    if (!o || o.userData.key !== k) {
      let m;
      try { m = await this._geometry(e); }
      catch (err) { status(`${e.name || e.type}: ${err.message}`, 'err'); return; }
      if (this._key(e) !== k) return;            // params changed meanwhile
      const mesh = meshFromBuffers(m.positions, m.index, rockMaterial());
      mesh.userData = { kind: 'rock', id: e.id, key: k };
      if (o) { this.group.remove(o); disposeTree(o); }
      o = mesh;
      this.objs.set(e.id, o);
      this.group.add(o);
      if (this.sel === e.id) { this._highlight(); this.app.view.attach(o); }
    }
    this._place(o, e);
  }

  _place(o, e) {
    o.position.set(...e.position);
    o.rotation.set(e.rotation[0] * deg, e.rotation[1] * deg, e.rotation[2] * deg, 'XYZ');
    o.scale.set(...e.scale);
    o.updateMatrixWorld();
  }

  _fromGizmo(commit) {
    const o = this.app.view.tc.object;
    if (!o || o.userData.kind !== 'rock') return;
    const e = this.els.find(x => x.id === o.userData.id);
    if (!e) return;
    e.position = o.position.toArray().map(v => +v.toFixed(5));
    e.rotation = [o.rotation.x, o.rotation.y, o.rotation.z].map(v => +(v / deg).toFixed(3));
    e.scale = o.scale.toArray().map(v => +Math.max(v, 0.05).toFixed(4));
    if (commit) { this.app.changed('move rock'); this.renderPanel(); }
  }

  _highlight() {
    for (const [id, o] of this.objs) o.material.emissive?.setHex(id === this.sel ? SEL_EMISSIVE : 0x000000);
  }

  select(id) {
    this.sel = id;
    this._highlight();
    const o = id ? this.objs.get(id) : null;
    this.app.view.attach(o, this.app.gizmoMode);
    if (this.app.tab === 'rocks') this.renderPanel();
  }

  pickables() { return [...this.objs.values()]; }

  // ---------------------------------------------------------------- editing
  defaultPosition() {
    const t = this.app.view.tank;
    return [t.L * (0.25 + 0.5 * Math.random()), t.W * (0.35 + 0.3 * Math.random()), t.sand];
  }

  add(type, extra = {}) {
    const defs = this.app.lib.element_defaults[type];
    const n = this.els.filter(e => e.type === type).length + 1;
    const e = { id: uid(type[0]), type, name: `${TYPE_LABEL[type].split(' ')[0]} ${n}`, position: this.defaultPosition(),
      rotation: [0, 0, 0], scale: [1, 1, 1], params: JSON.parse(JSON.stringify(defs)), ...extra };
    if ('seed' in e.params) e.params.seed = Math.floor(Math.random() * 9999) + 1;
    this.els.push(e);
    this.app.changed(`add ${type}`);
    this.sync().then(() => { this.select(e.id); if (type !== 'mesh') this.snap(e.id, false); });
    return e;
  }

  duplicate(id) {
    const s = this.els.find(e => e.id === id);
    if (!s) return;
    const e = JSON.parse(JSON.stringify(s));
    e.id = uid(s.type[0]);
    e.name = `${s.name} copy`;
    e.position = [s.position[0] + 0.05, s.position[1] + 0.03, s.position[2]];
    if ('seed' in e.params) e.params.seed = Math.floor(Math.random() * 9999) + 1;
    this.els.push(e);
    this.app.changed('duplicate');
    this.sync().then(() => this.select(e.id));
  }

  remove(id) {
    const i = this.els.findIndex(e => e.id === id);
    if (i < 0) return;
    this.els.splice(i, 1);
    this.select(null);
    this.app.changed('delete');
    this.sync();
  }

  // Drop the element so its base rests on the highest sand/rock surface under its footprint.
  snap(id, commit = true) {
    const e = this.els.find(x => x.id === id);
    const o = this.objs.get(id);
    if (!e || !o) return;
    o.updateMatrixWorld();
    const bb = new THREE.Box3().setFromObject(o);
    const others = [...this.objs.values()].filter(x => x !== o);
    const targets = [...others];
    if (this.app.view.sandTop) targets.push(this.app.view.sandTop);
    const rc = new THREE.Raycaster();
    const top = Math.max(bb.max.z, this.app.view.tank.H) + 0.5;
    let support = -Infinity;
    const cx = (bb.min.x + bb.max.x) / 2, cy = (bb.min.y + bb.max.y) / 2;
    const hx = (bb.max.x - bb.min.x) * 0.3, hy = (bb.max.y - bb.min.y) * 0.3;
    for (const [dx, dy] of [[0, 0], [hx, 0], [-hx, 0], [0, hy], [0, -hy]]) {
      rc.set(new THREE.Vector3(cx + dx, cy + dy, top), new THREE.Vector3(0, 0, -1));
      const sandTopVisible = this.app.view.sandTop.material.visible;
      this.app.view.sandTop.material.visible = true;
      const hits = rc.intersectObjects(targets, false);
      this.app.view.sandTop.material.visible = sandTopVisible;
      if (hits.length) support = Math.max(support, hits[0].point.z);
    }
    if (!Number.isFinite(support)) support = this.app.view.tank.sand;
    if (e.type === 'mesh') {
      e.position[2] += support - bb.min.z;          // imported meshes: bounding-box base on the support
    } else {
      e.position[2] = support;                       // designer elements: local origin = base
    }
    this._place(o, e);
    this.app.view.render();
    if (commit) { this.app.changed('snap'); this.renderPanel(); }
  }

  async importMesh(file, unit) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('unit', unit);
    const r = await busy(`importing ${file.name}`, () => api.form(`/api/projects/${this.app.pid}/meshes`, fd));
    const t = this.app.view.tank;
    const [lo, hi] = r.bounds_m;
    // place the mesh's footprint centre mid-tank, base on the sand
    const pos = [t.L / 2 - (lo[0] + hi[0]) / 2, t.W / 2 - (lo[1] + hi[1]) / 2, t.sand - lo[2]];
    const e = this.add('mesh', { name: r.element.name, params: r.element.params, position: pos });
    const rep = r.report;
    status(`imported ${file.name}: ${rep.triangles} triangles, ${rep.watertight ? 'watertight' : 'NOT watertight'}, ` +
      `size ${hi.map((v, i) => this.app.units.fmt(v - lo[i], 'length', 3)).join(' × ')}`, rep.watertight ? 'ok' : 'err');
    return e;
  }

  async build() {
    const app = this.app;
    const res = await busy('building the rockscape (union + marching cubes)', () =>
      api.post(`/api/projects/${app.pid}/rockscape/build`, { editor: app.ed, resolution: this.dz.resolution }));
    this.lastReport = res.report;
    app.loadPayload(res.project, true);
    status(`rockscape built: ${res.report.triangles} triangles, ${res.report.watertight ? 'watertight' : 'NOT watertight'}, ` +
      `${num(res.report.volume_m3 * 1000, 3)} L of rock`, res.report.watertight ? 'ok' : 'err');
  }

  stale() { return this.app.geomKey(this.els) !== this.app.builtKey; }

  // ---------------------------------------------------------------- panel
  renderPanel() {
    if (this.sculpt.active) { this.sculpt.renderPanel(); return; }
    if (this.scan.active) { this.scan.renderPanel(); return; }
    const app = this.app, U = app.units, P = app.panel;
    clear(P);
    P.append(el('h2', {}, 'Rockscape designer'));
    P.append(el('div', { class: 'hint' }, 'Add rock elements, then move/rotate/scale them in the view (click to select; W/E/R switch the gizmo, Del deletes, Ctrl+D duplicates). Press "Build rockscape" to merge them into the one watertight rock mesh the simulation uses.'));
    const add = el('div', { class: 'btns' });
    for (const t of ['rock', 'slab', 'pillar', 'arch', 'bridge']) add.append(el('button', { onclick: () => this.add(t) }, '+ ' + TYPE_LABEL[t]));
    const unitSel = el('select', { title: 'Unit the mesh file was drawn in' }, ...['mm', 'cm', 'm', 'in'].map(u => el('option', { value: u }, u)));
    const fileIn = el('input', { type: 'file', accept: '.stl,.obj,.ply', hidden: true, onchange: async () => {
      if (fileIn.files[0]) { try { await this.importMesh(fileIn.files[0], unitSel.value); } catch (err) { status(err.message, 'err'); } }
      fileIn.value = '';
    } });
    add.append(el('span', {}, el('button', { onclick: () => fileIn.click(), title: 'Import a finished, watertight rock model (STL/OBJ/PLY), e.g. from CAD' }, '+ Import model…'), ' in ', unitSel, fileIn));
    P.append(add);
    const scanIn = el('input', { type: 'file', accept: '.obj,.glb,.gltf,.ply,.stl,.zip', hidden: true, onchange: async () => {
      if (scanIn.files[0]) { try { await this.scan.start(scanIn.files[0]); } catch (err) { status(err.message, 'err'); } }
      scanIn.value = '';
    } });
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: () => scanIn.click(),
      title: 'A raw 3-D scan of dry rock from a phone app (Polycam, Scaniverse, 3d Scanner App, KIRI…): OBJ, GLB, PLY, STL or the ZIP export. The table is removed, holes are closed and the size is checked.' },
      '📱 Import phone scan…'), scanIn));

    // element list
    const flags = this.quickFlags();
    const list = el('div', { class: 'list' });
    if (!this.els.length) list.append(el('div', { class: 'empty' }, 'No rock yet. Add elements above.'));
    for (const e of this.els) {
      const f = flags[e.id];
      list.append(el('div', { class: 'item' + (e.id === this.sel ? ' sel' : ''), onclick: () => this.select(e.id) },
        el('span', {}, e.name || e.type), el('span', { class: 'sub', title: f || '' }, (f ? '⚠ ' : '') + TYPE_LABEL[e.type])));
    }
    P.append(el('h3', {}, `Elements (${this.els.length})`), list);

    const e = this.els.find(x => x.id === this.sel);
    if (e) this._elementForm(P, e);

    // build
    P.append(el('h3', {}, 'Build rockscape'));
    const stale = this.stale();
    P.append(el('div', { class: stale ? 'warnbox' : 'okbox' }, stale
      ? 'Elements changed since the last build: the simulation would use the old rockscape. Build before running.'
      : (app.ed.rockscape.length ? `Up to date: the project uses ${app.ed.rockscape.map(r => r.file).join(', ')}.` : 'No rockscape in the project.')));
    P.append(selectRow('Mesh resolution', [[0.003, '3 mm (fine)'], [0.004, '4 mm'], [0.005, '5 mm'], [0.006, '6 mm'], [0.008, '8 mm (fast)']].map(([v, t]) => [String(v), t]),
      String(this.dz.resolution), (v) => { this.dz.resolution = parseFloat(v); app.markDirty(); }, { title: 'Grid spacing of the union. Use at most the CFD cell size of the finest run you plan (Fine = 3 mm).' }));
    const b = el('div', { class: 'btns' },
      el('button', { class: 'primary', disabled: !this.els.length, onclick: () => this.build().catch(err => status(err.message, 'err')) }, 'Build rockscape'),
      el('a', { class: 'button', href: `/api/projects/${app.pid}/rockscape.stl`, title: 'Download the built rockscape (STL, millimetres)' }, 'Export STL'));
    P.append(b);
    if (this.lastReport) {
      const r = this.lastReport;
      const dens = this.dz.rock_density || 1600;
      P.append(kvTable([
        ['Rock volume', `${num(r.volume_m3 * 1000, 3)} L`],
        ['Est. dry weight', `${U.fmt(r.volume_m3 * dens, 'mass', 3)}`, 'solid volume × the density below (porous dry rock; adjust to your rock)'],
        ['Triangles', String(r.triangles)], ['Watertight', r.watertight ? 'yes' : 'NO'],
        ['Grid', `${r.grid.join(' × ')} at ${U.fmt(r.resolution_m, 'length', 2)}`]]));
      P.append(numRow('Rock density (dry, bulk)', U, null, () => dens / 1000, (v) => { this.dz.rock_density = v * 1000; app.markDirty(); this.renderPanel(); }, { unit: 'kg/L', step: 0.05, title: 'Used only for the weight estimate' }));
      P.append(el('h3', {}, `Geometry checks (${r.issues.length})`));
      if (!r.issues.length) P.append(el('div', { class: 'okbox' }, 'No issues: rock clear of the glass, supported on the sand, below the water surface.'));
      for (const is of r.issues) {
        P.append(el('div', { class: 'issue ' + is.severity, title: 'click to select the elements involved',
          onclick: () => { if (is.elements?.length) this.select(is.elements[0]); } }, (is.severity === 'warning' ? '⚠ ' : 'ℹ ') + is.message));
      }
      P.append(el('div', { class: 'hint' }, 'Checks are informational (simple footprint / centre-of-mass test, no structural analysis).'));
    }

    this.sculpt.entrySection(P);
  }

  _elementForm(P, e) {
    const app = this.app, U = app.units;
    const upd = (why) => { app.changed(why); this.sync(); };
    P.append(el('h3', {}, `${TYPE_LABEL[e.type]}`, el('span', {},
      el('button', { class: 'small', onclick: () => this.snap(e.id), title: 'Drop onto the sand / rock below' }, 'Snap down'), ' ',
      el('button', { class: 'small', onclick: () => this.duplicate(e.id) }, 'Duplicate'), ' ',
      el('button', { class: 'small danger', onclick: () => this.remove(e.id) }, 'Delete'))));
    P.append(textRow('Name', e.name, (v) => { e.name = v; app.changed('rename'); this.renderPanel(); }));
    P.append(vecRow('Position x, y, z', U, 'length', () => e.position, (v) => { e.position = v; upd('move'); }, { title: 'Element base point in tank coordinates (from the inside front-left-bottom glass corner)' }));
    P.append(vecRow('Rotation x, y, z', U, null, () => e.rotation, (v) => { e.rotation = v; upd('rotate'); }, { unit: 'deg', step: 5 }));
    P.append(vecRow('Scale x, y, z', U, null, () => e.scale, (v) => { e.scale = v.map(s => Math.max(s, 0.05)); upd('scale'); }, { unit: '×', step: 0.05 }));
    const p = e.params;
    const prm = (k, label, kind, opts = {}) => P.append(numRow(label, U, kind, () => p[k], (v) => { p[k] = v; upd('edit'); }, opts));
    if (e.type === 'rock') {
      P.append(vecRow('Size x, y, z', U, 'length', () => p.size, (v) => { p.size = v.map(s => Math.max(s, 0.01)); upd('edit'); }));
      prm('lobes', 'Lobes', null, { step: 1, min: 1, max: 6 });
      prm('roughness', 'Roughness', null, { step: 0.05, min: 0, max: 0.6 });
      prm('bury', 'Buried fraction', null, { step: 0.05, min: 0, max: 0.6, title: 'Share of the height below the base point (sits in the sand)' });
    } else if (e.type === 'slab') {
      P.append(vecRow('Size x, y, thickness', U, 'length', () => p.size, (v) => { p.size = v.map(s => Math.max(s, 0.005)); upd('edit'); }));
      prm('roughness', 'Roughness', null, { step: 0.05, min: 0, max: 0.6 });
    } else if (e.type === 'pillar') {
      prm('radius_bottom', 'Radius, bottom', 'length', { min: 0 }); prm('radius_top', 'Radius, top', 'length', { min: 0 });
      prm('height', 'Height', 'length', { min: 0 }); prm('roughness', 'Roughness', null, { step: 0.05, min: 0, max: 0.6 });
    } else if (e.type === 'arch') {
      prm('span', 'Span (leg to leg)', 'length', { min: 0 }); prm('height', 'Height', 'length', { min: 0 });
      prm('thickness', 'Thickness', 'length', { min: 0 }); prm('roughness', 'Roughness', null, { step: 0.05, min: 0, max: 0.6 });
    } else if (e.type === 'bridge') {
      prm('span', 'Span (leg to leg)', 'length', { min: 0 }); prm('height', 'Height (deck top)', 'length', { min: 0 });
      prm('deck_thickness', 'Deck thickness', 'length', { min: 0 }); prm('width', 'Deck width', 'length', { min: 0 });
      prm('leg_radius', 'Leg radius', 'length', { min: 0 }); prm('roughness', 'Roughness', null, { step: 0.05, min: 0, max: 0.6 });
    } else if (e.type === 'mesh') {
      P.append(kvTable([['File', p.file], ['File unit', p.unit]]));
    }
    if ('seed' in p) {
      P.append(el('div', { class: 'row' }, el('label', {}, 'Random seed'),
        el('span', {}, el('input', { type: 'number', value: p.seed, step: 1, style: 'width:70px', onchange: (ev) => { p.seed = parseInt(ev.target.value, 10) || 1; upd('seed'); } }), ' ',
          el('button', { class: 'small', title: 'New random shape', onclick: () => { p.seed = Math.floor(Math.random() * 9999) + 1; upd('seed'); this.renderPanel(); } }, '🎲')), el('span')));
    }
  }

  quickFlags() {
    const t = this.app.view.tank, out = {};
    if (!t) return out;
    for (const e of this.els) {
      const o = this.objs.get(e.id);
      if (!o) continue;
      const bb = new THREE.Box3().setFromObject(o);
      const gap = Math.min(bb.min.x, t.L - bb.max.x, bb.min.y, t.W - bb.max.y);
      if (gap <= 0.002) out[e.id] = 'touches or passes through the glass';
      else if (bb.max.z > t.ws) out[e.id] = 'rises above the water surface';
    }
    return out;
  }
}
