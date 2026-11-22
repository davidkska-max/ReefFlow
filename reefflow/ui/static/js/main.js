// ReefFlow UI: application shell (projects, tabs, save/validate, undo/redo, tank panel).
import { api, Units, el, clear, numRow, selectRow, textRow, kvTable, num, status, busy, modal, closeModal } from './util.js';
import { Viewport } from './viewport.js';
import { Designer } from './designer.js';
import { Equipment } from './equipment.js';
import { RunPanel } from './run.js';
import { Results, Compare } from './results.js';
import { Photos } from './photos.js';

const EDITOR_TABS = ['tank', 'rocks', 'equip', 'run'];
const OPTIONAL_TABS = ['photos', 'compare'];
const SUPPORT_URL = 'https://buymeacoffee.com/davidkska';   // plain link; opens in the browser, nothing is sent from the app
const LOGO_SVG = '<svg viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="lgw" x1="0" y1="0" x2="1" y2="1">'
  + '<stop offset="0" stop-color="#19b6d2"/><stop offset="1" stop-color="#0a5f80"/></linearGradient></defs>'
  + '<rect width="32" height="32" rx="8" fill="url(#lgw)"/>'
  + '<path d="M5 13c4-4 7 2 11-1.5S22 7 27 10" stroke="#fff" stroke-width="2.4" fill="none" stroke-linecap="round"/>'
  + '<path d="M5 19.5c4-4 7 2 11-1.5s6-4.5 11-1.5" stroke="#fff" stroke-opacity=".7" stroke-width="2" fill="none" stroke-linecap="round"/>'
  + '<path d="M9 27c0-3 1.2-4.6 2.6-5.6M11.6 27c0-2 .6-3.2 1.6-4M22 27c0-2.6 1.4-4 3-4.6" stroke="#ffd7a8" stroke-width="1.8" fill="none" stroke-linecap="round"/></svg>';
const LS = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } } };

class App {
  constructor() {
    this.panel = document.getElementById('panel');
    this.view = new Viewport(document.getElementById('view'));
    this.pid = null; this.ed = null; this.validation = null; this.dirty = false;
    this.tab = 'tank'; this.gizmoMode = 'translate';
    this.history = []; this.future = []; this._snap = null;
    this.builtKey = null;
    this.collapsed = new Set(JSON.parse(LS.get('reefflow.collapsed') || '[]'));
  }

  async init() {
    this.lib = await api.get('/api/library');
    this.units = new Units(this.lib, 'US');
    this.designer = new Designer(this);
    this.equipment = new Equipment(this);
    this.runPanel = new RunPanel(this);
    this.results = new Results(this);
    this.compare = new Compare(this);
    this.photos = new Photos(this);
    this.view.onPick = (hit) => this.onPick(hit);
    this.view.onHover = (e) => this.results.hover(e);
    this._wire();
    // panel sections fold open / closed by their heading; applied to whatever any tab renders
    this._sectObs = new MutationObserver(() => { if (!this._sectRaf) this._sectRaf = requestAnimationFrame(() => { this._sectRaf = 0; this._decorate(); }); });
    this._sectObs.observe(this.panel, { childList: true, subtree: true });
    document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('optional', OPTIONAL_TABS.includes(b.dataset.tab)));
    const last = LS.get('reefflow.project');
    if (last) { try { await this.openProject(last); } catch (e) { this.projectsDialog(); } }
    else this.projectsDialog();
    this.runPanel.refreshJobs();
    setInterval(() => this.runPanel.refreshJobs(), 15000);
  }

  _wire() {
    document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => this.setTab(b.dataset.tab)));
    document.getElementById('btnProjects').onclick = () => this.projectsDialog();
    document.getElementById('btnSave').onclick = () => this.save();
    document.getElementById('btnUndo').onclick = () => this.undo();
    document.getElementById('btnRedo').onclick = () => this.redo();
    document.getElementById('validBadge').onclick = () => this.showValidation();
    document.querySelectorAll('#unitsToggle button').forEach(b => b.addEventListener('click', () => {
      if (!this.ed || this.ed.display_units === b.dataset.u) return;
      this.ed.display_units = b.dataset.u;
      this.units = new Units(this.lib, b.dataset.u);
      this.changed('units');
      this._unitsButtons();
      this.renderPanel();
    }));
    document.querySelectorAll('#camViews button').forEach(b => b.addEventListener('click', () => this.view.setView(b.dataset.view)));
    document.querySelectorAll('#gizmoModes button').forEach(b => b.addEventListener('click', () => this.setGizmo(b.dataset.mode)));
    window.addEventListener('keydown', (e) => this.onKey(e));
    window.addEventListener('beforeunload', (e) => { if (this.dirty) { e.preventDefault(); e.returnValue = ''; } });
  }

  onKey(e) {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); this.save(); return; }
    if (this.tab === 'photos' && this.photos.onKey(e)) return;
    if (typing) return;
    if (this.tab === 'rocks' && this.designer.sculpt.onKey(e)) return;
    if ((e.ctrlKey || e.metaKey) && k === 'z' && !e.shiftKey) { e.preventDefault(); this.undo(); return; }
    if ((e.ctrlKey || e.metaKey) && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); this.redo(); return; }
    if ((e.ctrlKey || e.metaKey) && k === 'd') {
      e.preventDefault();
      if (this.tab === 'rocks' && this.designer.sel) this.designer.duplicate(this.designer.sel);
      if (this.tab === 'equip') this.equipment.duplicate();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (k === 'w') this.setGizmo('translate');
    if (k === 'e') this.setGizmo('rotate');
    if (k === 'r' && this.tab === 'rocks') this.setGizmo('scale');
    if (k === 'delete' || k === 'backspace') {
      if (this.tab === 'rocks' && this.designer.sel) this.designer.remove(this.designer.sel);
      if (this.tab === 'equip' && this.equipment.sel) this.equipment.remove();
    }
    if (k === 'escape') { this.designer.select(null); this.equipment.select(null); }
  }

  setGizmo(m) {
    this.gizmoMode = m;
    document.querySelectorAll('#gizmoModes button').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
    if (this.view.tc.object) {
      const isLight = ['light', 'overflow'].includes(this.view.tc.object.userData.kind);   // translate only
      const isPump = this.view.tc.object.userData.kind === 'pump';
      this.view.tc.setMode(isLight ? 'translate' : (isPump && m === 'scale') ? 'translate' : m);
    }
    this.view.render();
  }

  // ---------------------------------------------------------------- projects
  async openProject(pid) {
    if (this.dirty && this.pid && pid !== this.pid && !confirm('Discard unsaved changes to the current project?')) return;
    if (this.designer.sculpt.active && !(await this.designer.sculpt.leave())) return;
    const payload = await busy('opening project', () => api.get(`/api/projects/${encodeURIComponent(pid)}`));
    this.pid = pid;
    LS.set('reefflow.project', pid);
    this.history = []; this.future = [];
    this.results.cur = null; this.results.sum = null; this.results.clearScene(); this.results.runs = [];
    this.runPanel.opts = null; this.runPanel.est = null;
    this.designer.sel = null; this.equipment.sel = null;
    this.view._framed = false;
    this.loadPayload(payload, false);
    this.results.refreshRuns();
    closeModal();
    status(`opened ${payload.editor.name}`, 'ok');
  }

  // payload = server project response (editor model + yaml + validation)
  loadPayload(p, keepHistory) {
    this.ed = p.editor;
    this.yaml = p.yaml;
    this.validation = p.validation;
    this.builtKey = p.designer_stale ? null : this.geomKey(this.ed.designer.elements);
    this.units = new Units(this.lib, this.ed.display_units);
    this._snap = JSON.stringify(this.ed);
    if (!keepHistory) { this.history = []; this.future = []; }
    this.setDirty(false);
    document.getElementById('projName').textContent = this.ed.name;
    document.getElementById('btnYaml').hidden = false;
    document.getElementById('btnYaml').href = `/api/projects/${this.pid}/yaml`;
    document.getElementById('btnSave').disabled = false;
    this._unitsButtons();
    this.refreshShell();
    this.designer.sync();
    this.equipment.sync();
    this.updateBadge();
    this.setTab(this.tab);
  }

  geomKey(els) { return JSON.stringify((els || []).map(e => [e.type, e.position, e.rotation, e.scale, e.params])); }

  _unitsButtons() { document.querySelectorAll('#unitsToggle button').forEach(b => b.classList.toggle('on', b.dataset.u === this.ed?.display_units)); }

  async save() {
    if (!this.pid) return;
    try {
      const p = await busy('saving', () => api.put(`/api/projects/${this.pid}`, { editor: this.ed }));
      const hist = this.history, fut = this.future;
      this.loadPayload(p, true);
      this.history = hist; this.future = fut;
      this.updateUndo();
      status(p.validation.ok ? 'saved' : `saved, but the project has ${p.validation.errors.length} error(s)`, p.validation.ok ? 'ok' : 'err');
    } catch (e) { status(`save failed: ${e.message}`, 'err'); throw e; }
  }

  setDirty(d) { this.dirty = d; document.getElementById('dirtyMark').hidden = !d; }
  markDirty() { this.setDirty(true); this.validateSoon(); }

  // record an edit for undo, mark dirty, re-validate
  changed(why) {
    const now = JSON.stringify(this.ed);
    if (this._snap && now !== this._snap) { this.history.push(this._snap); if (this.history.length > 100) this.history.shift(); this.future = []; }
    this._snap = now;
    this.updateUndo();
    this.markDirty();
  }

  updateUndo() {
    document.getElementById('btnUndo').disabled = !this.history.length;
    document.getElementById('btnRedo').disabled = !this.future.length;
  }

  _restore(snap) {
    this.ed = JSON.parse(snap);
    this._snap = snap;
    this.units = new Units(this.lib, this.ed.display_units);
    this._unitsButtons();
    this.refreshShell();
    this.designer.sync();
    this.equipment.sync();
    this.updateUndo();
    this.markDirty();
    this.renderPanel();
  }
  undo() { if (!this.history.length) return; this.future.push(this._snap); this._restore(this.history.pop()); status('undone'); }
  redo() { if (!this.future.length) return; this.history.push(this._snap); this._restore(this.future.pop()); status('redone'); }

  validateSoon() {
    clearTimeout(this._vt);
    this._vt = setTimeout(async () => {
      if (!this.pid) return;
      try {
        this.validation = await api.post(`/api/projects/${this.pid}/validate`, { editor: this.ed });
        this.updateBadge();
        if (this.tab === 'tank') this._tankDerived();
        // pumps panel shows derived flows from the validation: refresh it unless you are typing in it
        if (this.tab === 'equip' && !this.panel.contains(document.activeElement)) {
          const top = this.panel.scrollTop;
          this.equipment.renderPanel();
          this.panel.scrollTop = top;
        }
      } catch (e) { /* server busy; next edit retries */ }
    }, 400);
  }

  updateBadge() {
    const b = document.getElementById('validBadge'), v = this.validation;
    if (!v) { b.hidden = true; return; }
    b.hidden = false;
    const stale = this.designer.stale();
    if (!v.ok) { b.className = 'badge bad'; b.textContent = `${v.errors.length} error${v.errors.length > 1 ? 's' : ''}`; }
    else if (stale || v.warnings.length) { b.className = 'badge warn'; b.textContent = stale ? 'rock not built' : `${v.warnings.length} note${v.warnings.length > 1 ? 's' : ''}`; }
    else { b.className = 'badge ok'; b.textContent = 'valid'; }
    this.updateSteps();
  }

  showValidation() {
    const v = this.validation;
    if (!v) return;
    modal(el('div', {}, el('h2', {}, 'Project check'),
      v.errors.length ? el('div', { class: 'errbox' }, 'Errors (must fix before running):\n• ' + v.errors.join('\n• ')) : el('div', { class: 'okbox' }, 'No errors: the core project parser accepts this project.'),
      v.warnings.length ? el('div', { class: 'warnbox' }, '• ' + v.warnings.join('\n• ')) : null,
      el('div', { class: 'btns' }, el('button', { onclick: closeModal }, 'Close'))));
  }

  updateJobBadge(jobs) {
    const live = jobs.filter(j => ['queued', 'running'].includes(j.status));
    const b = document.querySelector('#tabs button[data-tab=run] .x');
    const r = live.find(j => j.status === 'running');
    b.textContent = live.length ? (r ? `${Math.round(r.pct)} %` : 'queued') : '';
  }

  // Step tabs: a tick on the steps that are done for this project.
  updateSteps() {
    const ed = this.ed, v = this.validation, runs = this.results?.runs || [];
    const done = {
      tank: !!(ed && v && v.tank),
      rocks: !!(ed && (ed.rockscape || []).length && !this.designer?.stale()),
      equip: !!(ed && (ed.pumps || []).length),
      run: runs.length > 0,
      results: runs.some(r => r.status === 'analysed'),
    };
    document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('done', !!done[b.dataset.tab] && b.dataset.tab !== this.tab));
  }

  _h3Key(h) {
    const t = [...h.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('') || h.textContent;
    return t.replace(/\(\d+\)/g, '').replace(/:.*/, ':').trim();
  }

  _decorate() {
    for (const h of this.panel.querySelectorAll('h3')) {
      if (h.closest('.spec')) continue;
      const key = `${this.tab}|${this._h3Key(h)}`;
      if (!h.classList.contains('sect')) {
        h.classList.add('sect');
        h.title = h.title || 'Click to fold this section';
        h.addEventListener('click', (e) => {
          if (e.target.closest('button, select, input, a, label, textarea')) return;
          if (this.collapsed.has(key)) this.collapsed.delete(key); else this.collapsed.add(key);
          LS.set('reefflow.collapsed', JSON.stringify([...this.collapsed]));
          this._applySect(h, this.collapsed.has(key));
        });
      }
      this._applySect(h, this.collapsed.has(key));
    }
  }

  _applySect(h, col) {
    h.classList.toggle('collapsed', col);
    for (let n = h.nextElementSibling; n && !/^H[23]$/.test(n.tagName); n = n.nextElementSibling) n.classList.toggle('sect-hidden', col);
  }

  refreshShell() { if (this.ed && this.tab !== 'results') this.view.setTank(this.ed.tank, this.ed.overflow); }

  setTab(tab) {
    if (!this.pid && tab !== 'compare') { this.tab = tab; this.projectsDialog(); return; }
    const prev = this.tab;
    this.tab = tab;
    document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
    const editor = EDITOR_TABS.includes(tab);
    this.view.editorRoot.visible = editor;
    this.view.resultsRoot.visible = tab === 'results' || tab === 'compare';
    document.getElementById('gizmoModes').hidden = !(tab === 'rocks' || tab === 'equip') || (tab === 'rocks' && (this.designer.sculpt.active || this.designer.scan.active));
    document.querySelector('#gizmoModes button[data-mode=scale]').hidden = tab !== 'rocks';
    document.getElementById('legend').hidden = true;
    document.getElementById('readout').hidden = true;
    this.view.attach(null);
    if (prev === 'photos' && tab !== 'photos') this.photos.leave();
    if (prev === 'rocks' && tab !== 'rocks' && this.designer.sculpt.active) this.designer.sculpt.leave();
    if (prev === 'rocks' && tab !== 'rocks' && this.designer.scan.active) this.designer.scan.leave(false);
    document.getElementById('viewTools').hidden = false;
    if ((editor || tab === 'photos') && (prev === 'results' || prev === 'compare' || !this.view.tank)) this.refreshShell();
    if (tab === 'rocks') { this.view.pickables = () => this.designer.pickables(); if (!this.designer.sculpt.active && !this.designer.scan.active) this.designer.select(this.designer.sel); }
    else if (tab === 'equip') { this.view.pickables = () => this.equipment.pickables(); this.equipment.select(this.equipment.sel); }
    else if (tab === 'results') { this.view.pickables = []; this.results.showShell(); if (this.results.sum && !this.results.group.children.length) this.results.buildScene(); }
    else if (tab === 'photos') { this.view.pickables = () => this.photos.pickables(); this.photos.enter(); }
    else this.view.pickables = [];
    this.setGizmo(this.gizmoMode);
    this.renderPanel();
    this.updateSteps();
    this.view.render();
  }

  renderPanel() {
    if (!this.ed && this.tab !== 'compare') { clear(this.panel); return; }
    if (this.tab === 'tank') this.tankPanel();
    else if (this.tab === 'rocks') this.designer.renderPanel();
    else if (this.tab === 'equip') this.equipment.renderPanel();
    else if (this.tab === 'run') this.runPanel.render();
    else if (this.tab === 'results') this.results.render();
    else if (this.tab === 'compare') this.compare.render();
    else if (this.tab === 'photos') this.photos.render();
  }

  onPick(hit) {
    let o = hit?.object;
    if (this.tab === 'photos') { this.photos.pick(hit); return; }
    if (this.tab === 'rocks' && (this.designer.sculpt.active || this.designer.scan.active)) return;
    if (this.tab === 'rocks') {
      while (o && o.userData?.kind !== 'rock') o = o.parent;
      this.designer.select(o ? o.userData.id : null);
    } else if (this.tab === 'equip') {
      while (o && !['pump', 'light', 'overflow'].includes(o.userData?.kind)) o = o.parent;
      this.equipment.select(o ? { kind: o.userData.kind, name: o.userData.name, nozzle: o.userData.nozzle } : null);
    }
  }

  async openResults(pid, rid) {
    if (pid !== this.pid) await this.openProject(pid);
    await this.results.refreshRuns();
    this.setTab('results');
    await this.results.open(pid, rid);
  }

  // ---------------------------------------------------------------- tank panel
  tankPanel() {
    const P = this.panel, U = this.units, t = this.ed.tank, s = this.ed.simulation;
    clear(P);
    P.append(el('h2', {}, 'Tank'));
    const ch = (why) => { this.changed(why); this.refreshShell(); this.equipment.sync(); };
    P.append(textRow('Project name', this.ed.name, (v) => { this.ed.name = v || 'untitled'; document.getElementById('projName').textContent = this.ed.name; this.changed('name'); }));
    P.append(el('h3', {}, 'Dimensions'));
    P.append(selectRow('Tank style', [['', '— choose —'], ['rimmed', 'Rimmed: black frame at the top and bottom'], ['rimless_top', 'Rimless top, with a base frame'], ['rimless', 'Rimless: no frame (glass edges)']], t.style || '', (v) => { t.style = v || null; this.changed('tank'); },
      { title: 'Which edges have a plastic frame. The photo calibration then asks you to click the frame corners or the glass corners.' }));
    P.append(el('div', { class: 'hint' }, 'External size: measure the outside of the tank with a tape, over the frame if it has one (height from the stand to the top edge). The internal size is computed from the glass thickness. Coordinates everywhere start at the inside front-left-bottom glass corner: x along the length (left → right seen from the front), y front → back, z up.'));
    P.append(numRow('External length', U, 'length', () => t.length, (v) => { t.length = v; ch('tank'); }, { min: 0 }));
    P.append(numRow('External width (front-back)', U, 'length', () => t.width, (v) => { t.width = v; ch('tank'); }, { min: 0 }));
    P.append(numRow('External height', U, 'length', () => t.height, (v) => { t.height = v; ch('tank'); }, { min: 0 }));
    P.append(numRow('Glass thickness (sides)', U, 'length', () => t.glass, (v) => { t.glass = v; ch('tank'); }, { min: 0 }));
    P.append(numRow('Bottom glass thickness', U, 'length', () => t.bottom_glass, (v) => { t.bottom_glass = v; ch('tank'); }, { min: 0 }));
    P.append(numRow('Water level below rim', U, 'length', () => t.water_gap, (v) => { t.water_gap = v; ch('tank'); }, { min: 0 }));
    P.append(numRow('Sand depth', U, 'length', () => t.sand_depth, (v) => { t.sand_depth = v; ch('tank'); }, { min: 0 }));
    this.derivedBox = el('div');
    P.append(el('h3', {}, 'Derived'), this.derivedBox);
    this._tankDerived();

    P.append(el('h3', {}, 'Simulation defaults'));
    P.append(selectRow('Default preset', Object.entries(this.lib.presets).map(([k, dx]) => [k, `${k} (${U.fmt(dx, 'length', 2)})`]), s.preset, (v) => { s.preset = v; this.changed('preset'); }));
    P.append(numRow('Spin-up time', U, null, () => s.spinup, (v) => { s.spinup = v; this.changed('spinup'); }, { unit: 's', step: 10, min: 0 }));
    P.append(numRow('Averaging time', U, null, () => s.average, (v) => { s.average = v; this.changed('average'); }, { unit: 's', step: 10, min: 0.1 }));
    P.append(numRow('Dead-zone speed threshold', U, 'velocity', () => this.ed.criteria.dead_zone_speed, (v) => { this.ed.criteria.dead_zone_speed = v; this.changed('criteria'); }, { min: 0, title: 'Starting assumption (ASSUMPTIONS A-210), not a husbandry standard' }));

    P.append(el('h3', {}, 'Project file'));
    P.append(el('div', { class: 'hint' }, 'Everything is stored in one human-readable YAML file. Advanced settings (fluid, collision, coral profiles, lighting details) can be edited directly.'));
    P.append(el('div', { class: 'btns' }, el('button', { onclick: () => this.yamlDialog() }, 'Edit YAML…'), el('a', { class: 'button', href: `/api/projects/${this.pid}/yaml` }, 'Download YAML')));
  }

  _tankDerived() {
    const box = this.derivedBox;
    if (!box || !box.isConnected) return;
    clear(box);
    const v = this.validation, U = this.units;
    if (v?.errors?.length) box.append(el('div', { class: 'errbox' }, v.errors.join('\n')));
    if (!v?.tank) return;
    const t = v.tank;
    box.append(kvTable([
      ['Internal L × W × H', t.internal_m.map(x => U.fmtv(x, 'length', 4)).join(' × ') + ' ' + U.unit('length')],
      ['Water surface above bottom glass', U.fmt(t.water_surface_m, 'length', 4)],
      ['Water column above sand', U.fmt(t.water_column_m, 'length', 4)],
      ['Gross internal volume', U.fmt(t.gross_volume_m3, 'volume', 4)],
      ['Water above the sand (before rock)', U.fmt(t.water_volume_above_sand_m3, 'volume', 4)]]));
  }

  yamlDialog() {
    const ta = el('textarea', { class: 'yaml', spellcheck: 'false' }, this.yaml);
    const err = el('div');
    const close = modal(el('div', {}, el('h2', {}, 'Project file (YAML)'),
      el('div', { class: 'hint' }, 'Quantities are "<number> <unit>" (in, mm, GPH, L/h, s, ...). Saving replaces the project file; the core parser checks it.'),
      ta, err,
      el('div', { class: 'btns' },
        el('button', { class: 'primary', onclick: async () => {
          if (this.dirty && !confirm('This discards unsaved edits made in the other tabs. Continue?')) return;
          try { const p = await api.put(`/api/projects/${this.pid}/yaml`, { text: ta.value }); this.loadPayload(p, false); close(); status(p.validation.ok ? 'YAML saved' : 'YAML saved with errors', p.validation.ok ? 'ok' : 'err'); }
          catch (e) { clear(err).append(el('div', { class: 'errbox' }, e.message)); }
        } }, 'Save YAML'),
        el('button', { onclick: () => close() }, 'Cancel'))), { wide: true, sticky: true });
  }

  // ---------------------------------------------------------------- projects dialog
  async projectsDialog() {
    let list = [];
    let ws = '';
    try { const r = await api.get('/api/projects'); list = r.projects; ws = r.workspace; } catch (e) { status(e.message, 'err'); }
    const name = el('input', { type: 'text', placeholder: 'Name (optional)', style: 'width: 100%' });
    const box = el('div', { class: 'list projlist', style: 'max-height: 300px' });
    if (!list.length) box.append(el('div', { class: 'empty' }, 'No projects yet: start from the 75-gallon demo or a blank tank below.'));
    const ago = (t) => {
      const d = (Date.now() / 1000 - t) / 86400;
      return d < 1 / 24 ? 'just now' : d < 1 ? `${Math.round(d * 24)} h ago` : d < 30 ? `${Math.round(d)} day${Math.round(d) === 1 ? '' : 's'} ago` : new Date(t * 1000).toLocaleDateString();
    };
    for (const p of [...list].sort((a, b) => b.modified - a.modified)) {
      box.append(el('div', { class: 'item' + (p.id === this.pid ? ' sel' : '') },
        el('span', { style: 'cursor:pointer;flex:1;min-width:0', onclick: () => this.openProject(p.id).catch(e => status(e.message, 'err')) },
          el('b', {}, p.name),
          el('div', { class: 'projmeta' }, el('span', { class: 'chip runs' }, `${p.runs} run${p.runs === 1 ? '' : 's'}`),
            el('span', { class: 'chip', title: new Date(p.modified * 1000).toLocaleString() }, `saved ${ago(p.modified)}`),
            p.id === this.pid ? el('span', { class: 'chip' }, 'open now') : null)),
        el('span', { style: 'white-space:nowrap' },
          el('button', { class: 'small', title: 'Make a copy of this project', onclick: async () => { const n = prompt('Name of the copy', `${p.name} copy`); if (!n) return; try { const r = await api.post(`/api/projects/${p.id}/copy`, { name: n }); await this.openProject(r.id); } catch (e) { status(e.message, 'err'); } } }, 'Copy'), ' ',
          el('button', { class: 'small danger', title: 'Delete this project and its runs', onclick: async () => { if (!confirm(`Delete project "${p.name}" and all its runs?`)) return; try { await api.del(`/api/projects/${p.id}`); if (p.id === this.pid) { this.pid = null; this.ed = null; LS.set('reefflow.project', ''); } this.projectsDialog(); } catch (e) { status(e.message, 'err'); } } }, 'Delete'))));
    }
    const create = async (template) => {
      try { const r = await api.post('/api/projects', { name: name.value.trim() || (template === 'demo' ? 'Demo 75-gallon reef' : 'New tank'), template }); await this.openProject(r.id); }
      catch (e) { status(e.message, 'err'); }
    };
    const yfile = el('input', { type: 'file', accept: '.yaml,.yml' });
    const mfiles = el('input', { type: 'file', accept: '.stl,.obj,.ply', multiple: true });
    const hero = el('div', { class: 'welcome' },
      el('div', { class: 'logo', html: LOGO_SVG + '<h1>Reef<span>Flow</span></h1>' }),
      el('p', {}, 'Model your reef tank, then see where the water moves, where it stalls and how much light reaches each part of the rock, before you move a thing.'),
      el('div', { class: 'ws', title: 'Projects are stored in this folder on your computer' }, `Projects folder: ${ws}`),
      el('div', { class: 'support' }, 'ReefFlow is free. If it helps your tank, you can ',
        el('a', { href: SUPPORT_URL, target: '_blank', rel: 'noopener' }, 'buy me a coffee ☕'), '.'));
    modal(el('div', {}, hero,
      el('h3', {}, list.length ? 'Your projects' : 'Projects'), box,
      el('h3', {}, 'Start a new project'),
      el('div', { class: 'row full' }, name),
      el('div', { class: 'choices' },
        el('button', { class: 'choice primary', onclick: () => create('demo') }, el('b', {}, 'From the 75-gallon demo'), el('span', {}, 'A complete example: rock, pumps, overflow and lights, ready to run.')),
        el('button', { class: 'choice', onclick: () => create('blank') }, el('b', {}, 'Blank tank'), el('span', {}, 'Enter your own tank size, then add rock and equipment.'))),
      el('details', {}, el('summary', {}, 'Import a project file'),
        el('div', { class: 'hint' }, 'Choose the YAML file and, if it references rock meshes, the mesh files too.'),
        el('div', { class: 'row full' }, el('label', {}, 'Project YAML ', yfile)),
        el('div', { class: 'row full' }, el('label', {}, 'Rock meshes (optional) ', mfiles)),
        el('div', { class: 'btns' }, el('button', { onclick: async () => {
          if (!yfile.files[0]) { status('choose a YAML file', 'err'); return; }
          const fd = new FormData(); fd.append('project_file', yfile.files[0]);
          for (const f of mfiles.files) fd.append('meshes', f);
          try { const r = await api.form('/api/projects/import', fd); await this.openProject(r.id); if (!r.validation.ok) status('imported with errors: ' + r.validation.errors.join('; '), 'err'); }
          catch (e) { status(e.message, 'err'); }
        } }, 'Import'))),
      this.pid ? el('div', { class: 'btns', style: 'justify-content:flex-end' }, el('button', { onclick: closeModal }, 'Close')) : null), { sticky: !this.pid });
  }
}

const app = new App();
window.reefflow = app;          // for debugging from the browser console
app.init().catch((e) => { status(`startup failed: ${e.message}`, 'err'); console.error(e); });
