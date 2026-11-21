// Phone-scan import: upload a raw scan (OBJ / GLB / PLY / STL / ZIP from a phone scanning app),
// check size and upright, cut away the table, and let the server (reefflow.scan) turn it into
// a watertight rock that joins the rockscape as an imported mesh element.
import * as THREE from 'three';
import { api, el, clear, selectRow, status, busy, num, kvTable, unpackMesh } from './util.js';
import { meshFromBuffers, rockMaterial, disposeTree } from './viewport.js';

const UPS = [['auto', 'Auto'], ['+z', 'Z up'], ['+y', 'Y up'], ['-y', 'Y down'], ['+x', 'X up'], ['-x', 'X down'], ['-z', 'Upside down']];
const UNIT_OPTS = [['auto', 'auto'], ['m', 'm'], ['cm', 'cm'], ['mm', 'mm'], ['in', 'in']];
const DX_OPTS = [['', 'Auto'], ['0.0015', '1.5 mm (fine, slow)'], ['0.002', '2 mm'], ['0.003', '3 mm'], ['0.004', '4 mm (fast)']];
const CUT_COLOR = 0xd9534f;

export class ScanImport {
  constructor(app, designer) {
    this.app = app;
    this.designer = designer;
    this.active = false;
    this.group = new THREE.Group();
    this.group.visible = false;
    app.view.editorRoot.add(this.group);
    this._reset();
  }

  _reset() {
    this.scan = null; this.info = null; this.result = null;
    this.s = { unit: 'auto', scale: 1, up: 'auto', ground: 'auto', cutH: null, dx: '', keep: 'pieces' };
    this.measure = { dim: 'max', value: null };
  }

  // ---------------------------------------------------------------- enter / leave
  async start(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await busy(`reading the scan ${file.name}`, () => api.form(`/api/projects/${this.app.pid}/scans`, fd));
    this._reset();
    this.scan = r.scan;
    this.active = true;
    this.designer.select(null);
    document.getElementById('gizmoModes').hidden = true;
    this.app.view.renderer.localClippingEnabled = true;
    this.group.visible = true;
    await this.preview();
    this.app.view.setView('3d');
  }

  leave(deleteRaw = false) {
    if (!this.active) return;
    if (deleteRaw && this.scan) api.del(`/api/projects/${this.app.pid}/scans/${this.scan.id}`).catch(() => {});
    this.active = false;
    this._clearScene();
    this.group.visible = false;
    this._reset();
    document.getElementById('gizmoModes').hidden = this.app.tab !== 'rocks';
    this.app.renderPanel();
    this.app.view.render();
  }

  _clearScene() { for (const c of [...this.group.children]) { this.group.remove(c); disposeTree(c); } }

  // ---------------------------------------------------------------- server calls
  _body() { return { unit: this.s.unit, scale: this.s.scale, up: this.s.up }; }

  async preview() {
    const r = await busy('turning the scan upright', () => api.binary(`/api/projects/${this.app.pid}/scans/${this.scan.id}/preview`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(this._body()) }));
    this.info = r.meta;
    this.result = null;
    if (this.s.ground !== 'none') {
      if (this.info.ground && this.s.ground === 'auto') this.s.cutH = this.info.ground.height;
      if (!this.info.ground && this.s.ground === 'auto') this.s.cutH = null;
    }
    this._raw = unpackMesh(r.buf);
    this._showRaw(this._raw);
    this.renderPanel();
  }

  cutZ() {   // absolute z of the cut in the upright scan frame, or null
    if (this.s.ground === 'none' || this.s.cutH === null) return null;
    return this.info.bounds_m[0][2] + this.s.cutH;
  }

  // where the upright scan is drawn in the tank: footprint centre mid-tank, cut plane (or lowest point) on the sand
  // (the table is drawn 1.5 mm above the sand so the two do not flicker through each other)
  _offset() {
    const t = this.app.view.tank, [lo, hi] = this.keptBounds();
    const zc = this.cutZ();
    const base = this.info.ground ? Math.min(this.info.ground.z - 0.0015, zc ?? Infinity) : (zc ?? lo[2]);
    return new THREE.Vector3(t.L / 2 - (lo[0] + hi[0]) / 2, t.W / 2 - (lo[1] + hi[1]) / 2, t.sand - base);
  }

  // bounds of the part above the cut, from the preview geometry (the table would otherwise dominate)
  keptBounds() {
    const zc = this.cutZ(), P = this._raw?.positions;
    if (!P || zc === null) return this.info.bounds_m;
    const lo = [Infinity, Infinity, zc], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) {
      if (P[i + 2] < zc) continue;
      for (let a = 0; a < 3; a++) { const v = P[i + a]; if (v < lo[a]) lo[a] = v; if (v > hi[a]) hi[a] = v; }
    }
    return Number.isFinite(hi[2]) ? [lo, hi] : this.info.bounds_m;
  }

  _showRaw(m) {
    this._clearScene();
    this._raw = m;
    const off = this._offset();
    const zc = this.cutZ();
    const keepMat = rockMaterial();
    keepMat.side = THREE.DoubleSide;
    const keep = meshFromBuffers(m.positions, m.index, keepMat);
    keep.position.copy(off);
    this.group.add(keep);
    if (zc !== null) {
      const zw = zc + off.z;
      keepMat.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 0, 1), -zw)];
      const cutMat = new THREE.MeshStandardMaterial({ color: CUT_COLOR, roughness: 1, transparent: true, opacity: 0.55, side: THREE.DoubleSide,
        clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 0, -1), zw)] });
      const cut = new THREE.Mesh(keep.geometry, cutMat);
      cut.position.copy(off);
      this.group.add(cut);
      const [lo, hi] = this.keptBounds();
      const w = (hi[0] - lo[0]) * 1.4, d = (hi[1] - lo[1]) * 1.4;
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ color: CUT_COLOR, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
      plane.position.set((lo[0] + hi[0]) / 2 + off.x, (lo[1] + hi[1]) / 2 + off.y, zw);
      this.group.add(plane);
    }
    this.app.view.render();
  }

  async clean() {
    const b = { ...this._body(), keep: this.s.keep, dx: this.s.dx ? parseFloat(this.s.dx) : null,
      ground: this.s.ground === 'none' || this.s.cutH === null ? 'none' : (this.s.ground === 'auto' ? 'auto' : this.s.cutH) };
    const r = await busy('cleaning the scan: closing holes, removing the table, making it watertight', () =>
      api.post(`/api/projects/${this.app.pid}/scans/${this.scan.id}/clean`, b));
    this.result = r;
    const m = await api.mesh(`/api/projects/${this.app.pid}/element_mesh`, { element: { type: 'mesh', params: r.element.params } });
    this._clearScene();
    const mesh = meshFromBuffers(m.positions, m.index, rockMaterial());
    const t = this.app.view.tank, [lo, hi] = r.bounds_m;
    mesh.position.set(t.L / 2 - (lo[0] + hi[0]) / 2, t.W / 2 - (lo[1] + hi[1]) / 2, t.sand - lo[2]);
    this.group.add(mesh);
    this.app.view.render();
    const rep = r.report;
    status(`scan cleaned: watertight, ${num(rep.volume_m3 * 1000, 3)} L of rock, ${rep.triangles} triangles`, 'ok');
    this.renderPanel();
  }

  add() {
    const r = this.result;
    if (!r) return;
    const t = this.app.view.tank, [lo, hi] = r.bounds_m;
    const pos = [t.L / 2 - (lo[0] + hi[0]) / 2, t.W / 2 - (lo[1] + hi[1]) / 2, t.sand - lo[2]];
    this.leave(false);
    const e = this.designer.add('mesh', { name: r.element.name, params: r.element.params, position: pos });
    status(`added ${r.element.name} to the rock elements: move it with the gizmo, then Build rockscape`, 'ok');
    return e;
  }

  // ---------------------------------------------------------------- size helpers
  sizeAboveCut() {   // [x, y, z] of the part that will be kept (from the preview geometry)
    const [lo, hi] = this.keptBounds();
    return [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  }

  applyMeasurement() {
    const v = this.measure.value;
    if (!(v > 0)) { status('enter the real measurement first', 'err'); return; }
    const sz = this.sizeAboveCut();
    const cur = this.measure.dim === 'max' ? Math.max(sz[0], sz[1]) : sz[{ x: 0, y: 1, z: 2 }[this.measure.dim]];
    if (!(cur > 0)) return;
    const k = v / cur;
    this.s.scale = +(this.s.scale * k).toPrecision(5);
    if (this.s.cutH !== null && this.s.ground !== 'auto') this.s.cutH *= k;
    this.preview().catch(e => status(e.message, 'err'));
  }

  // ---------------------------------------------------------------- panel
  renderPanel() {
    const app = this.app, U = app.units, P = app.panel, info = this.info;
    clear(P);
    P.append(el('h2', {}, 'Phone scan clean-up'));
    if (!info) { P.append(el('div', { class: 'hint' }, 'Reading the scan…')); return; }
    P.append(el('div', { class: 'hint' }, `${this.scan.name}: ${info.triangles.toLocaleString()} triangles. ` +
      'Check the size and that the rock stands upright, then press Clean up. Red = cut away (the table or floor).'));
    const dis = !!this.result;

    // 1 size
    P.append(el('h3', {}, '1 · Size'));
    const sz = this.sizeAboveCut();
    P.append(el('div', { class: 'row' }, el('label', {}, 'Scan size (W × D × H)'),
      el('b', {}, sz.map(v => U.fmt(v, 'length', 3)).join(' × ')), el('span')));
    P.append(selectRow('File unit', UNIT_OPTS.map(([v, t]) => [v, v === 'auto' ? `auto (${info.unit_guessed})` : t]), this.s.unit,
      (v) => { this.s.unit = v; this.preview().catch(e => status(e.message, 'err')); },
      { title: 'Phone scanning apps save in metres; auto guesses from the size' }));
    const mDim = el('select', {}, ...[['max', 'longest side'], ['z', 'height'], ['x', 'width (left–right)'], ['y', 'depth (front–back)']]
      .map(([v, t]) => el('option', { value: v, selected: v === this.measure.dim }, t)));
    mDim.onchange = () => { this.measure.dim = mDim.value; };
    const mVal = el('input', { type: 'number', min: 0, step: 'any', style: 'width:70px', placeholder: U.unit('length'),
      value: this.measure.value ? num(U.to(this.measure.value, 'length'), 4) : '' });
    mVal.onchange = () => { const x = parseFloat(mVal.value); this.measure.value = x > 0 ? U.from(x, 'length') : null; };
    P.append(el('div', { class: 'row', title: 'Measure the real rock with a tape and type one dimension here: the scan is scaled to match. Needed when the app had no LiDAR (photo-only scans have no true size).' },
      el('label', {}, 'Real size (tape measure)'),
      el('span', {}, mDim, ' ', mVal, ' ', U.unit('length'), ' ',
        el('button', { class: 'small', disabled: dis, onclick: () => { mVal.onchange(); this.applyMeasurement(); } }, 'Scale to this')), el('span')));
    if (this.s.scale !== 1) {
      P.append(el('div', { class: 'hint' }, `Scaled × ${num(this.s.scale, 4)} `,
        el('button', { class: 'small', onclick: () => { this.s.scale = 1; this.preview().catch(e => status(e.message, 'err')); } }, 'Reset')));
    }
    const t = app.view.tank;
    if (sz[0] > t.L || sz[1] > t.W || sz[2] > t.ws - t.sand) {
      P.append(el('div', { class: 'warnbox' }, 'The scan is bigger than the tank. Check the unit and the real size above.'));
    }

    // 2 upright
    P.append(el('h3', {}, '2 · Upright'));
    const useg = el('div', { class: 'seg tools' });
    for (const [v, label] of UPS) {
      const on = this.s.up === v;
      useg.append(el('button', { class: on ? 'on' : '', disabled: dis, title: v === 'auto' ? 'Find the table under the scan' : `Use the file's ${v} axis as up`,
        onclick: () => { this.s.up = v; if (this.s.ground !== 'none') this.s.ground = 'auto'; this.preview().catch(e => status(e.message, 'err')); } }, label));
    }
    P.append(useg);
    P.append(el('div', { class: 'hint' }, `Using ${info.up} as up: ${info.up_reason}. If the rock looks tipped over in the view, pick another button.`));

    // 3 table / floor
    P.append(el('h3', {}, '3 · Table or floor'));
    const hasCut = this.s.ground !== 'none' && this.s.cutH !== null;
    const cb = el('input', { type: 'checkbox', disabled: dis });
    cb.checked = hasCut;
    cb.onchange = () => {
      if (cb.checked) { this.s.ground = info.ground ? 'auto' : 'manual'; this.s.cutH = info.ground ? info.ground.height : 0.005; }
      else this.s.ground = 'none';
      this._showRaw(this._raw); this.renderPanel();
    };
    P.append(el('div', { class: 'row', title: 'Cut everything below a flat plane: the table the rock sat on, and a few millimetres of the rock base so it stands flat' },
      el('label', {}, 'Remove the table / floor'), cb, el('span')));
    P.append(el('div', { class: 'hint' }, info.ground
      ? `Found a flat surface under the scan (${Math.round(info.ground.area_fraction * 100)} % of the scanned area).`
      : 'No table found: the scan may already be cropped. You can still cut the base flat at a chosen height.'));
    if (hasCut) {
      const ch = el('input', { type: 'number', step: 'any', min: 0, style: 'width:80px', disabled: dis, value: num(U.to(this.s.cutH, 'length'), 4) });
      ch.onchange = () => { const x = parseFloat(ch.value); if (x >= 0) { this.s.cutH = U.from(x, 'length'); this.s.ground = 'manual'; this._showRaw(this._raw); this.renderPanel(); } };
      const nudge = (k) => () => { this.s.cutH = Math.max(0, this.s.cutH + k); this.s.ground = 'manual'; this._showRaw(this._raw); this.renderPanel(); };
      P.append(el('div', { class: 'row', title: 'Height of the cut above the lowest point of the scan' }, el('label', {}, 'Cut height'),
        el('span', {}, ch, ' ', U.unit('length'), ' ',
          el('button', { class: 'small', disabled: dis, onclick: nudge(-0.002), title: 'lower by 2 mm' }, '▼'), ' ',
          el('button', { class: 'small', disabled: dis, onclick: nudge(0.002), title: 'raise by 2 mm' }, '▲')), el('span')));
    }

    // 4 clean up
    P.append(el('h3', {}, '4 · Clean up'));
    P.append(selectRow('Detail', DX_OPTS, this.s.dx, (v) => { this.s.dx = v; }, { title: 'Voxel size of the solid model. Auto: about 1/150 of the rock size (1.5–4 mm). Finer is slower and makes bigger files; the CFD grid is 3–5 mm anyway.' }));
    P.append(selectRow('Loose pieces', [['pieces', 'keep pieces ≥ 5 % of the main rock'], ['main', 'keep the largest piece only']], this.s.keep, (v) => { this.s.keep = v; }));
    P.append(el('div', { class: 'btns' },
      el('button', { class: 'primary', disabled: dis, onclick: () => this.clean().catch(e => status(e.message, 'err')) }, 'Clean up'),
      el('button', { onclick: () => this.leave(true) }, 'Cancel')));

    if (this.result) {
      const r = this.result.report, dens = app.ed.designer.rock_density || 1600;
      P.append(el('div', { class: 'okbox' }, 'Watertight rock ready. Add it, then place it with the move/rotate gizmo like any element.'));
      const tbl = kvTable([
        ['Size (W × D × H)', r.size_m.map(v => U.fmt(v, 'length', 3)).join(' × ')],
        ['Rock volume', `${num(r.volume_m3 * 1000, 3)} L`],
        ['Est. dry weight', U.fmt(r.volume_m3 * dens, 'mass', 3), 'solid volume × the rock density of the designer (porous dry rock)'],
        ['Holes closed', r.raw_open_edge_length_m > 0 ? `open scan edges ${U.fmt(r.raw_open_edge_length_m, 'length', 3)} long, all closed` : 'none (the scan was closed)'],
        ['Table cut', r.ground_cut ? `${U.fmt(r.ground_cut.height, 'length', 2)} above the lowest point; ${Math.round(r.ground_cut.removed_area_fraction * 100)} % of the scan removed` : 'none'],
        ['Loose pieces removed', String(r.removed_pieces), 'pieces of 1 cm³ or more that are not part of the rock (bits of the room, floating noise)'],
        r.filled_void_volume_m3 > 1e-6 ? ['Sealed voids filled', U.fmt(r.filled_void_volume_m3, 'volume', 2), 'hollow pockets with no opening; water cannot reach them'] : null,
        r.normals_flipped ? ['Note', 'the scan was inside-out (faces reversed); fixed'] : null,
        ['Triangles', r.triangles.toLocaleString()],
        ['Detail', `${num(r.dx * 1000, 2)} mm voxels`],
      ]);
      tbl.classList.add('wrap');
      P.append(tbl);
      P.append(el('div', { class: 'btns' },
        el('button', { class: 'primary', onclick: () => this.add() }, 'Add to rockscape'),
        el('button', { onclick: () => { this.result = null; this.preview().catch(e => status(e.message, 'err')); } }, 'Back to settings'),
        el('button', { onclick: () => this.leave(true) }, 'Cancel')));
    }
    P.append(el('h3', {}, 'Scanning tips'));
    P.append(el('ul', { class: 'hint tips' },
      el('li', {}, 'Scan dry rock on a plain table; walk all the way round, then low and high.'),
      el('li', {}, 'Crop the scan to the rock in the app if it caught the room; the table is fine (it is removed here).'),
      el('li', {}, 'Export OBJ or GLB (Polycam, Scaniverse, 3d Scanner App, KIRI…). ZIP exports work too.'),
      el('li', {}, 'Without LiDAR the size is a guess: measure the rock and use "Real size".')));
  }
}
