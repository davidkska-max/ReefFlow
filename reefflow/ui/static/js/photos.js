// Photos tab: rockscape from photos (spec Feature 1). Photo editor (2-D canvas: calibration
// clicks, rock-mask painting) + 3-D result with carve/add tools. All computation is done by
// reefflow.photo on the server; this module only collects clicks and strokes.
import * as THREE from 'three';
import { api, el, clear, num, status, busy, modal, selectRow, numRow, kvTable } from './util.js';
import { meshFromBuffers, rockMaterial, disposeTree } from './viewport.js';

const LEVEL_COLOR = { rim: '#ffd400', water: '#3ad0ff', sand: '#ffb070', bottom: '#ff6a6a', vertical: '#c58cff', seam: '#c58cff', top: '#ffd400', base: '#ff6a6a', outer: '#e8e8e8' };
const SIDE_NAME = { FL: 'front left', FR: 'front right', BR: 'back right', BL: 'back left' };
const LEVEL_NAME = { top: 'Top outside corner', base: 'Bottom outside corner', water: 'Water line at the corner', seam: 'Corner joint (anywhere)', rim: 'Inside rim corner', sand: 'Sand line at the corner', bottom: 'Inside bottom corner' };
// Tank style (D-841): which edges have a frame. Set by the Photos panel before drawing.
let STYLE = null;
const TOP_FRAMED = () => STYLE === 'rimmed' || STYLE === null;
const BASE_FRAMED = () => STYLE !== 'rimless';
const STYLE_OPTS = [['rimmed', 'Rimmed: black frame at the top and bottom'], ['rimless_top', 'Rimless top, with a base frame'], ['rimless', 'Rimless: no frame (glass edges)']];
function levelName(lv) {
  if (lv === 'top' && STYLE) return TOP_FRAMED() ? 'Top of the frame' : 'Top of the glass';
  if (lv === 'base' && STYLE) return BASE_FRAMED() ? 'Bottom of the base frame' : 'Bottom of the glass';
  return LEVEL_NAME[lv] || lv;
}
// Exactly what to click, for the current tank style.
function styleHint(n, fallback) {
  const [lv, c] = n.split('_'), side = SIDE_NAME[c] || c;
  if (lv === 'top') return `${side}: ` + (STYLE === null ? 'top outside corner of the tank: the top of the frame if it has one, else the top edge of the glass'
    : TOP_FRAMED() ? 'the outside top corner of the black frame (its top edge, at the very corner), not the glass under it'
      : 'the top corner of the glass itself: where the top edge of the front glass meets the side glass');
  if (lv === 'base') return `${side}: ` + (STYLE === null ? 'bottom outside corner: where the tank, or its base frame, sits on the stand'
    : BASE_FRAMED() ? 'the bottom outside corner of the base frame, where the frame sits on the stand (not where the glass meets the frame)'
      : 'the bottom corner of the glass, where it sits on the stand or leveling mat');
  return fallback;
}
// Plain-language label for a click name, e.g. top_FL -> "Top of the frame · front left".
function pointLabel(n) {
  const [lv, c, two] = n.split('_');
  return `${levelName(lv)}${two ? ' (2nd point)' : ''} · ${SIDE_NAME[c] || c}`;
}
// Where each point sits on the little "where to click" diagram (viewBox 0 0 240 136).
// Side views: the near face (outer box) is the big rectangle, the far face the inner one.
function sketchPos(kind, n) {
  const [lv, c, two] = n.split('_');
  if (kind === 'top') {
    const o = { FL: [12, 124], FR: [228, 124], BR: [228, 12], BL: [12, 12] }, i = { FL: [40, 104], FR: [200, 104], BR: [200, 32], BL: [40, 32] };
    if (lv === 'top' || lv === 'rim') return lv === 'rim' ? [o[c][0] + (c[1] === 'L' ? 6 : -6), o[c][1] + (c[0] === 'F' ? -6 : 6)] : o[c];
    if (lv === 'sand' || lv === 'water' || lv === 'bottom') return i[c];
    return null;
  }
  const near = { front: ['FL', 'FR'], back: ['BR', 'BL'], left: ['BL', 'FL'], right: ['FR', 'BR'] }[kind];
  const far = { front: ['BL', 'BR'], back: ['FR', 'FL'], left: ['BR', 'FR'], right: ['FL', 'BL'] }[kind];
  if (near.includes(c)) {
    const x = near.indexOf(c) ? 228 : 12;
    if (lv === 'top') return [x, 12];
    if (lv === 'base') return [x, 124];
    const xi = near.indexOf(c) ? 220 : 20;
    return { rim: [xi, 22], water: [xi, 30], sand: [xi, 104], bottom: [xi, 114] }[lv] || null;
  }
  const x = far.indexOf(c) ? 192 : 48;
  return { top: [x, 34], rim: [x, 34], water: [x, 42], seam: [x, two ? 62 : 82], sand: [x, 98], bottom: [x, 104], base: [x, 104] }[lv] || null;
}
const KIND_LABEL = { top: 'top (from above)', front: 'front', back: 'back', left: 'left side', right: 'right side' };
const DX_CHOICES = [[0.003, '3 mm (slow)'], [0.004, '4 mm'], [0.005, '5 mm (default)'], [0.0075, '7.5 mm'], [0.01, '10 mm (fast)']];

// ------------------------------------------------------------------ 2-D photo canvas
class PhotoCanvas {
  constructor(container, owner) {
    this.owner = owner;
    this.box = container;
    this.canvas = el('canvas');
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.img = null; this.maskCanvas = null; this.overlay = null;
    this.zoom = 1; this.panX = 0; this.panY = 0; this.opacity = 0.45; this.showMask = true;
    this.stroke = null; this.poly = []; this.mouse = null; this._pan = null; this.autoFit = true;
    const c = this.canvas;
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const k = Math.exp(-e.deltaY * 0.0015), r = c.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      this.panX = sx - (sx - this.panX) * k; this.panY = sy - (sy - this.panY) * k; this.zoom *= k; this.autoFit = false;
      this.draw();
    }, { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('mousedown', (e) => this._down(e));
    window.addEventListener('mousemove', (e) => this._move(e));
    window.addEventListener('mouseup', (e) => this._up(e));
    c.addEventListener('dblclick', () => this.owner.finishPolygon());
    new ResizeObserver(() => this.resize()).observe(container);
  }

  resize() {
    const r = this.box.getBoundingClientRect();
    if (r.width < 20 || r.height < 20) return;           // hidden: keep the last size
    this.canvas.width = r.width; this.canvas.height = r.height;
    if (this.autoFit) this.fit(); else this.draw();
  }

  fit() {
    this.autoFit = true;
    if (!this.img) { this.draw(); return; }
    const W = this.canvas.width, H = this.canvas.height;
    this.zoom = Math.min(W / this.img.width, H / this.img.height) * 0.97;
    this.panX = (W - this.img.width * this.zoom) / 2; this.panY = (H - this.img.height * this.zoom) / 2;
    this.draw();
  }

  async setImage(url) {
    const img = new Image();
    img.src = url;
    await img.decode();
    const first = !this.img || this.img.width !== img.width || this.img.height !== img.height;
    this.img = img;
    if (first) this.fit(); else this.draw();
  }

  async setMask(url) {
    if (!url) { this.maskCanvas = null; this.draw(); return; }
    const img = new Image();
    img.src = url;
    try { await img.decode(); } catch (e) { this.maskCanvas = null; this.draw(); return; }
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const x = c.getContext('2d');
    x.drawImage(img, 0, 0);
    const d = x.getImageData(0, 0, c.width, c.height), p = d.data;
    for (let i = 0; i < p.length; i += 4) {
      if (p[i] > 127) { p[i] = 40; p[i + 1] = 235; p[i + 2] = 120; p[i + 3] = 255; } else p[i + 3] = 0;
    }
    x.putImageData(d, 0, 0);
    this.maskCanvas = c;
    this.draw();
  }

  toImg(e) {
    const r = this.canvas.getBoundingClientRect();
    return [(e.clientX - r.left - this.panX) / this.zoom - 0.5, (e.clientY - r.top - this.panY) / this.zoom - 0.5];
  }
  toScr(u, v) { return [this.panX + (u + 0.5) * this.zoom, this.panY + (v + 0.5) * this.zoom]; }

  _down(e) {
    if (!this.img) return;
    if (e.button === 1 || e.button === 2 || e.altKey || this.owner.spaceDown) {
      this._pan = [e.clientX, e.clientY, this.panX, this.panY]; this.autoFit = false; e.preventDefault(); return;
    }
    if (e.button !== 0) return;
    const p = this.toImg(e);
    const tool = this.owner.tool2d();
    if (tool === 'click') this.owner.placePoint(p);
    else if (tool === 'brush+' || tool === 'brush-') { this.stroke = [p]; this.draw(); }
    else if (tool === 'poly+' || tool === 'poly-') { this.poly.push(p); this.draw(); }
  }

  _move(e) {
    if (this._pan) {
      this.panX = this._pan[2] + e.clientX - this._pan[0]; this.panY = this._pan[3] + e.clientY - this._pan[1];
      this.draw(); return;
    }
    if (!this.img || e.target !== this.canvas) { if (this.mouse) { this.mouse = null; this.draw(); } return; }
    this.mouse = this.toImg(e);
    if (this.stroke) this.stroke.push(this.mouse);
    this.owner.readout(this.mouse);
    this.draw();
  }

  _up() {
    if (this._pan) { this._pan = null; return; }
    if (this.stroke) {
      const s = this.stroke; this.stroke = null;
      this.owner.strokeDone(s.length === 1 ? [s[0], s[0]] : s);
    }
  }

  draw() {
    const x = this.ctx, c = this.canvas;
    x.setTransform(1, 0, 0, 1, 0, 0);
    x.fillStyle = '#1c2328'; x.fillRect(0, 0, c.width, c.height);
    if (!this.img) {
      x.fillStyle = '#9ab'; x.font = '14px system-ui'; x.fillText('Add a photo (left panel) to start.', 20, 60); return;
    }
    const w = this.img.width * this.zoom, h = this.img.height * this.zoom;
    x.imageSmoothingEnabled = this.zoom < 2;
    x.drawImage(this.img, this.panX, this.panY, w, h);
    const o = this.owner;
    if (this.maskCanvas && this.showMask && o.sub === 'mask') {
      x.globalAlpha = this.opacity; x.drawImage(this.maskCanvas, this.panX, this.panY, w, h); x.globalAlpha = 1;
    }
    if (o.sub === 'calibrate') this._drawCalibration(x);
    if (o.sub === 'mask') this._drawMaskTools(x);
  }

  _drawCalibration(x) {
    const o = this.owner, v = o.view(), ov = this.overlay;
    if (ov) {
      x.lineWidth = 1.5;
      for (const e of ov.edges) {
        x.strokeStyle = LEVEL_COLOR[e.level] || '#fff'; x.globalAlpha = 0.85; x.beginPath();
        e.uv.forEach(([u, vv], i) => { const [sx, sy] = this.toScr(u, vv); if (i) x.lineTo(sx, sy); else x.moveTo(sx, sy); });
        x.stroke();
      }
      x.globalAlpha = 1;
      // model positions only for what was clicked, plus a dashed "expected here" ring for
      // the point being placed (from the current fit) -- not every corner, which clutters
      for (const [n, p] of Object.entries(ov.points)) {
        if (!p || (!v.clicks[n] && n !== o.active)) continue;
        const [sx, sy] = this.toScr(p[0], p[1]);
        x.strokeStyle = n === o.active && !v.clicks[n] ? '#ff00d0' : LEVEL_COLOR[n.split('_')[0]]; x.lineWidth = 1.5;
        if (n === o.active && !v.clicks[n]) x.setLineDash([4, 3]);
        x.beginPath(); x.arc(sx, sy, n === o.active && !v.clicks[n] ? 11 : 6, 0, 2 * Math.PI); x.stroke();
        x.setLineDash([]);
      }
    }
    for (const [n, p] of Object.entries(v.clicks)) {
      const [sx, sy] = this.toScr(p[0], p[1]);
      const mp = ov?.points?.[n];
      if (mp) { const [mx, my] = this.toScr(mp[0], mp[1]); x.strokeStyle = '#ff3b3b'; x.lineWidth = 2; x.beginPath(); x.moveTo(sx, sy); x.lineTo(mx, my); x.stroke(); }
      x.strokeStyle = n === o.active ? '#ff00d0' : (n.startsWith('seam_') ? LEVEL_COLOR.seam : '#ffff00'); x.lineWidth = 2;
      x.beginPath();
      if (n.startsWith('seam_')) { x.moveTo(sx, sy - 9); x.lineTo(sx + 6, sy); x.lineTo(sx, sy + 9); x.lineTo(sx - 6, sy); x.closePath(); }
      else { x.moveTo(sx - 8, sy); x.lineTo(sx + 8, sy); x.moveTo(sx, sy - 8); x.lineTo(sx, sy + 8); }
      x.stroke();
      x.fillStyle = 'rgba(0,0,0,.55)'; x.font = '600 12px system-ui';
      const lab = pointLabel(n), w = x.measureText(lab).width;
      const lx = sx + 14 + w > this.canvas.width ? sx - 14 - w : sx + 8;      // keep labels inside the canvas
      x.fillRect(lx, sy + 3, w + 6, 16); x.fillStyle = '#ffff88'; x.fillText(lab, lx + 3, sy + 15);
    }
    if (this.mouse && o.active) {
      const [sx, sy] = this.toScr(...this.mouse);
      const lab = pointLabel(o.active);
      x.font = '600 12px system-ui';
      x.fillStyle = 'rgba(0,0,0,.6)'; x.fillRect(sx + 12, sy + 8, x.measureText(lab).width + 10, 18);
      x.fillStyle = '#ff9df0'; x.fillText(lab, sx + 16, sy + 21);
      this._drawLoupe(x, sx, sy);
    }
  }

  // Magnifier for placing corners: 3x of the photo around the cursor, in the canvas
  // corner away from the cursor. Hidden when the view is already zoomed in that far.
  _drawLoupe(x, sx, sy) {
    const S = 180, k = 3, c = this.canvas;
    if (this.zoom >= k) return;
    const [u, v] = this.mouse, half = S / (2 * k);
    const left = sx > c.width / 2 ? 10 : c.width - S - 10, top = 10;
    x.save();
    x.fillStyle = '#000'; x.fillRect(left - 2, top - 2, S + 4, S + 4);
    x.beginPath(); x.rect(left, top, S, S); x.clip();
    x.imageSmoothingEnabled = false;
    x.drawImage(this.img, u + 0.5 - half, v + 0.5 - half, 2 * half, 2 * half, left, top, S, S);
    x.strokeStyle = '#ff00d0'; x.lineWidth = 1;
    x.beginPath(); x.moveTo(left + S / 2, top); x.lineTo(left + S / 2, top + S); x.moveTo(left, top + S / 2); x.lineTo(left + S, top + S / 2); x.stroke();
    x.restore();
  }

  _drawMaskTools(x) {
    const o = this.owner, t = o.tool2d(), r = o.brush * this.zoom;
    const add = t.endsWith('+');
    if (this.stroke && this.stroke.length) {
      x.strokeStyle = add ? 'rgba(40,235,120,.6)' : 'rgba(255,60,60,.6)'; x.lineWidth = 2 * r; x.lineCap = 'round'; x.lineJoin = 'round';
      x.beginPath(); this.stroke.forEach(([u, v], i) => { const [sx, sy] = this.toScr(u, v); if (i) x.lineTo(sx, sy); else x.moveTo(sx, sy); });
      if (this.stroke.length === 1) { const [sx, sy] = this.toScr(...this.stroke[0]); x.lineTo(sx + 0.1, sy); }
      x.stroke();
    }
    if (this.poly.length) {
      x.strokeStyle = add ? '#28eb78' : '#ff4040'; x.fillStyle = add ? 'rgba(40,235,120,.25)' : 'rgba(255,60,60,.25)'; x.lineWidth = 2;
      x.beginPath(); this.poly.forEach(([u, v], i) => { const [sx, sy] = this.toScr(u, v); if (i) x.lineTo(sx, sy); else x.moveTo(sx, sy); });
      if (this.mouse) x.lineTo(...this.toScr(...this.mouse));
      x.closePath(); x.fill(); x.stroke();
    }
    if (this.mouse && t.startsWith('brush')) {
      const [sx, sy] = this.toScr(...this.mouse);
      x.strokeStyle = add ? '#28eb78' : '#ff4040'; x.lineWidth = 1.5; x.beginPath(); x.arc(sx, sy, r, 0, 2 * Math.PI); x.stroke();
    }
  }
}

// ------------------------------------------------------------------ Photos tab
export class Photos {
  constructor(app) {
    this.app = app;
    this.st = null; this.sel = null; this.sub = 'calibrate'; this.display = '2d';
    this.active = null; this.maskTool = 'brush+'; this.brush = 12; this.spaceDown = false;
    this.caps = null; this.job = null; this.es = null; this.jobLine = '';
    this.group = new THREE.Group();
    app.view.scene.add(this.group);
    this.group.visible = false;
    this.meshObj = null; this.markers = new THREE.Group(); this.group.add(this.markers);
    this.tool3d = { type: 'capsule', op: 'carve', radius: 0.02, span: 0.08, rise: 0.06, length: 0.12, azimuth: 0, size: 0.05 };
    this.picks = [];
    this.box = document.getElementById('photoView');
    this.canvas = new PhotoCanvas(this.box, this);
    window.addEventListener('keydown', (e) => { if (e.code === 'Space' && this.app.tab === 'photos' && !this._typing()) { this.spaceDown = true; } });
    window.addEventListener('keyup', (e) => { if (e.code === 'Space') this.spaceDown = false; });
  }

  _typing() { return ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName); }
  base() { return `/api/projects/${this.app.pid}/photos`; }
  view() { return this.st?.views.find(v => v.id === this.sel); }
  tool2d() { return this.sub === 'calibrate' ? 'click' : this.maskTool; }

  // ---------------------------------------------------------------- lifecycle
  async enter() {
    if (!this.caps) api.get('/api/photo/capabilities').then(c => { this.caps = c; if (this.app.tab === 'photos') this.render(); }).catch(() => {});
    await this.load();
    this.show();
  }

  leave() {
    this.box.hidden = true; this.group.visible = false;
    document.getElementById('readout').hidden = true; document.getElementById('viewTools').hidden = false;
  }

  show() {
    const two = this.display === '2d';
    this.box.hidden = !two;
    this.group.visible = !two;
    document.getElementById('viewTools').hidden = two;
    this.app.view.editorRoot.visible = false;
    if (two) this.canvas.resize();
    this.app.view.render();
  }

  async load(keepSel = true) {
    try { this.st = await api.get(this.base()); } catch (e) { status(e.message, 'err'); this.st = null; this.render(); return; }
    if (!keepSel || !this.view()) { this.sel = this.st.views[0]?.id || null; this.active = null; }
    if (this.st.job && !this.es) this.follow(this.st.job.id);
    await this.loadPhoto();
    if (this.st.recon) await this.loadMesh(); else this.clearMesh();
    this.render();
  }

  async loadPhoto() {
    const v = this.view();
    if (!v) { this.canvas.img = null; this.canvas.maskCanvas = null; this.canvas.overlay = null; this.canvas.draw(); return; }
    const t = Date.now();
    await this.canvas.setImage(`${this.base()}/views/${v.id}/image?t=${v.added}`);
    await this.canvas.setMask(v.mask ? `${this.base()}/views/${v.id}/mask.png?t=${t}` : null);
    this.canvas.overlay = null;
    if (v.calibration) { try { this.canvas.overlay = (await api.get(`${this.base()}/views/${v.id}/overlay`)).overlay; } catch (e) { /* ignore */ } }
    if (!this.active) this.active = this.nextPoint();
    this.canvas.draw();
  }

  select(vid) { this.sel = vid; this.active = null; this.canvas.poly = []; this.display = '2d'; this.show(); this.loadPhoto().then(() => this.render()); }

  // ---------------------------------------------------------------- calibration
  points() { const v = this.view(); return v ? this.st.recommended[v.kind] : []; }
  nextPoint(after) {
    const v = this.view(); if (!v) return null;
    const names = this.points().map(p => p.name);
    const start = after ? names.indexOf(after) + 1 : 0;
    for (let i = 0; i < names.length; i++) { const n = names[(start + i) % names.length]; if (!v.clicks[n]) return n; }
    return null;
  }

  async placePoint(p) {
    const v = this.view();
    if (!v || !this.active) { status('choose a point in the list first (all recommended points are placed)', 'err'); return; }
    if (p[0] < -0.5 || p[1] < -0.5 || p[0] > v.width - 0.5 || p[1] > v.height - 0.5) return;
    const clicks = { ...v.clicks, [this.active]: p.map(c => +c.toFixed(2)) };
    const placed = this.active;
    v.clicks = clicks;
    this.active = this.nextPoint(placed);
    this.canvas.draw();
    await this.sendClicks(clicks);
  }

  async sendClicks(clicks, focal, sizeOk) {
    const v = this.view();
    try {
      const r = await api.put(`${this.base()}/views/${v.id}/clicks`, { clicks, focal, size_ok: sizeOk });
      Object.assign(v, r.view);
      this.canvas.overlay = r.overlay;
    } catch (e) { status(e.message, 'err'); }
    this.canvas.draw();
    this.renderPanelOnly();
  }

  removePoint(n) { const v = this.view(); const c = { ...v.clicks }; delete c[n]; v.clicks = c; this.active = n; this.sendClicks(c); }

  clearClicks() {
    const v = this.view(); if (!v || !Object.keys(v.clicks).length) return;
    if (!confirm('Remove every click on this photo and start the corners again?')) return;
    v.clicks = {}; this.active = this.nextPoint(); this.sendClicks({});
  }

  // ---------------------------------------------------------------- masks
  async strokeDone(stroke) {
    const mode = this.maskTool.endsWith('+') ? 'add' : 'subtract';
    await this.maskOps([{ op: 'brush', stroke: stroke.map(p => p.map(c => +c.toFixed(1))), radius: this.brush, mode }]);
  }

  async finishPolygon() {
    if (this.sub !== 'mask' || !this.maskTool.startsWith('poly')) return;
    const pts = this.canvas.poly; this.canvas.poly = [];
    if (pts.length < 3) { this.canvas.draw(); status('a polygon needs at least 3 points', 'err'); return; }
    await this.maskOps([{ op: 'polygon', vertices: pts.map(p => p.map(c => +c.toFixed(1))), mode: this.maskTool.endsWith('+') ? 'add' : 'subtract' }]);
  }

  async maskOps(ops, url = 'mask_ops') {
    const v = this.view();
    try {
      const r = url === 'mask_ops' ? await api.post(`${this.base()}/views/${v.id}/mask_ops`, { ops }) : await api.post(`${this.base()}/views/${v.id}/${url}`);
      Object.assign(v, r.view);
      await this.canvas.setMask(`${this.base()}/views/${v.id}/mask.png?t=${Date.now()}`);
    } catch (e) { status(e.message, 'err'); }
    this.renderPanelOnly();
  }

  // Sensitivity slider: move the outline of the last suggestion (fast, no GrabCut re-run);
  // if the server has none cached (restart, colour cue), suggest again with the setting.
  async setSensitivity(val) {
    const v = this.view(); if (!v) return;
    this.sensitivity = val;
    if (v.mask_edited && !confirm('Moving the sensitivity replaces this mask, including your brush edits. Continue?')) { this.render(); return; }
    try {
      const r = await busy('adjusting the outline', () => api.post(`${this.base()}/views/${v.id}/sensitivity`, { sensitivity: val }));
      Object.assign(v, r.view);
      await this.canvas.setMask(`${this.base()}/views/${v.id}/mask.png?t=${Date.now()}`);
      this.renderPanelOnly();
    } catch (e) {
      if (/Suggest/.test(e.message || '')) await this.suggest(false); else status(e.message, 'err');
    }
  }

  async suggest(all) {
    try {
      const cue = this.cue || 'auto', sensitivity = this.sensitivity ?? 50;
      const r = await busy(all ? 'segmenting all calibrated photos (GrabCut)' : 'segmenting this photo', () =>
        api.post(`${this.base()}/segment`, all ? { cue, sensitivity } : { views: [this.sel], cue, sensitivity }));
      this.st = r.state;
      const methods = Object.entries(r.result).map(([k, x]) => `${k}: ${x.method}`).join(', ');
      status(methods ? `rock mask suggested (${methods}); check and correct it with the brush` : 'nothing to do (edited masks are kept)', 'ok');
    } catch (e) { status(e.message, 'err'); }
    await this.loadPhoto();
    this.render();
  }

  readout(p) {
    const ro = document.getElementById('readout');
    const v = this.view();
    if (!v || !p) { ro.hidden = true; return; }
    ro.textContent = `pixel u ${p[0].toFixed(0)}, v ${p[1].toFixed(0)}  (${v.width} × ${v.height})   wheel = zoom, right-drag / Alt-drag = pan`;
    ro.hidden = false;
  }

  onKey(e) {
    if (this._typing()) return false;
    if (this.sub === 'mask' && this.display === '2d') {
      if (e.key === 'Enter') { this.finishPolygon(); return true; }
      if (e.key === 'Escape') { this.canvas.poly = []; this.canvas.draw(); return true; }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); this.maskOps(null, 'mask_undo'); return true; }
    }
    if (e.key === 'f' || e.key === 'F') { this.canvas.fit(); return true; }
    return false;
  }

  // ---------------------------------------------------------------- carving job
  async carve() {
    try {
      const r = await api.post(`${this.base()}/carve`);
      this.follow(r.job.id);
      status('carving started', 'ok');
    } catch (e) { status(e.message, 'err'); }
    this.render();
  }

  follow(id) {
    if (this.es) this.es.close();
    this.job = { id, pct: 0, status: 'queued' };
    const es = new EventSource(`/api/jobs/${id}/events`);
    this.es = es;
    es.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      Object.assign(this.job, { status: d.status, pct: d.pct, error: d.error });
      if (d.lines.length) this.jobLine = d.lines[d.lines.length - 1];
      this.renderJobBox();
    };
    es.addEventListener('end', async () => {
      es.close(); this.es = null;
      const ok = this.job.status === 'done';
      status(ok ? 'reconstruction ready: check it in the 3D view and against the photos' : `carving ${this.job.status}: ${this.job.error || ''}`, ok ? 'ok' : 'err');
      this.job = null;
      await this.load();
      if (ok && this.app.tab === 'photos') { this.display = '3d'; this.show(); this.render(); }
    });
    es.onerror = () => { es.close(); this.es = null; };
  }

  async cancelJob() { if (this.job) { try { await api.post(`/api/jobs/${this.job.id}/cancel`); } catch (e) { status(e.message, 'err'); } } }

  // ---------------------------------------------------------------- 3-D result + edits
  clearMesh() { if (this.meshObj) { this.group.remove(this.meshObj); disposeTree(this.meshObj); this.meshObj = null; } }

  async loadMesh() {
    try {
      const m = await api.mesh(`${this.base()}/recon/mesh`);
      this.clearMesh();
      if (!m.nv) return;
      this.meshObj = meshFromBuffers(m.positions, m.index, rockMaterial());
      this.meshObj.userData = { kind: 'photoRock' };
      this.group.add(this.meshObj);
      this.app.view.render();
    } catch (e) { this.clearMesh(); }
  }

  pick(hit) {
    if (!hit) return;
    const p = hit.point.toArray().map(v => +v.toFixed(4));
    const t = this.tool3d.type;
    if (t === 'capsule') this.picks.push(p); else this.picks = [p];
    this.drawPicks();
    this.renderPanelOnly();
  }

  pickables() { const o = []; if (this.meshObj) o.push(this.meshObj); if (this.app.view.sandTop) o.push(this.app.view.sandTop); return o; }

  drawPicks() {
    for (const c of [...this.markers.children]) { this.markers.remove(c); disposeTree(c); }
    const T = this.tool3d;
    for (const p of this.picks) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.005, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff00c8, depthTest: false }));
      s.position.set(...p); s.renderOrder = 5; this.markers.add(s);
    }
    const ghost = new THREE.MeshBasicMaterial({ color: T.op === 'carve' ? 0xff3355 : 0x33ff88, transparent: true, opacity: 0.35, depthWrite: false });
    if (T.type === 'capsule' && this.picks.length) {
      for (let i = 0; i < this.picks.length; i++) {
        const a = new THREE.Vector3(...this.picks[i]);
        const sp = new THREE.Mesh(new THREE.SphereGeometry(T.radius, 16, 12), ghost); sp.position.copy(a); this.markers.add(sp);
        if (i) {
          const b = new THREE.Vector3(...this.picks[i - 1]), len = a.distanceTo(b);
          const cy = new THREE.Mesh(new THREE.CylinderGeometry(T.radius, T.radius, len, 16, 1, true), ghost);
          cy.position.copy(a).add(b).multiplyScalar(0.5);
          cy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
          this.markers.add(cy);
        }
      }
    } else if (this.picks.length) {
      const p = new THREE.Vector3(...this.picks[0]);
      let m;
      if (T.type === 'sphere') m = new THREE.Mesh(new THREE.SphereGeometry(T.radius, 20, 14), ghost);
      else if (T.type === 'box') m = new THREE.Mesh(new THREE.BoxGeometry(T.size, T.size, T.size), ghost);
      else if (T.type === 'arch') {
        const g = new THREE.BoxGeometry(T.length, T.span, T.rise); g.translate(0, 0, T.rise / 2);
        m = new THREE.Mesh(g, ghost); m.rotation.z = T.azimuth * Math.PI / 180;
      }
      if (m) { m.position.copy(p); this.markers.add(m); }
    }
    this.app.view.render();
  }

  buildEdit() {
    const T = this.tool3d, P = this.picks;
    if (!P.length) throw new Error('click on the rock (or sand) in the 3D view first');
    let shape;
    if (T.type === 'capsule') { if (P.length < 2) throw new Error('a tunnel needs at least 2 points (entry and exit)'); shape = { type: 'capsule', points: P, radius: T.radius }; }
    else if (T.type === 'sphere') shape = { type: 'sphere', center: P[0], radius: T.radius };
    else if (T.type === 'box') { const h = T.size / 2; shape = { type: 'box', lo: P[0].map(c => c - h), hi: P[0].map(c => c + h) }; }
    else if (T.type === 'arch') {
      const a = T.azimuth * Math.PI / 180;
      shape = { type: 'arch', base_center: P[0], axis: [+Math.cos(a).toFixed(5), +Math.sin(a).toFixed(5)], span: T.span, rise: Math.max(T.rise, T.span / 2), length: T.length };
    }
    return { op: T.op, shape };
  }

  async setEdits(edits) {
    try {
      const r = await busy('applying edits', () => api.put(`${this.base()}/edits`, { edits }));
      this.st.edits = r.edits;
      if (this.st.recon) this.st.recon.volume = r.volume;
      await this.loadMesh();
    } catch (e) { status(e.message, 'err'); }
    this.render();
  }

  async applyEdit() {
    let e;
    try { e = this.buildEdit(); } catch (err) { status(err.message, 'err'); return; }
    this.picks = []; this.drawPicks();
    await this.setEdits([...this.st.edits, e]);
    status('edit applied (the comparison images update when you press "Update comparison")', 'ok');
  }

  async compare() {
    try { const r = await busy('reprojecting the reconstruction into each photo', () => api.post(`${this.base()}/compare`)); if (this.st.recon) this.st.recon.comparison = r; }
    catch (e) { status(e.message, 'err'); }
    this.render();
  }

  async useAsRockscape() {
    if (!confirm('Replace the project\'s rockscape with this reconstruction? (Rockscape-designer elements are replaced by the reconstruction; you can add designer pieces on top afterwards.)')) return false;
    let ok = false;
    try {
      const r = await busy('exporting the reconstruction', () => api.post(`${this.base()}/use`));
      const p = await api.get(`/api/projects/${this.app.pid}`);
      this.app.loadPayload(p, false);
      status(`${r.note} Mesh: ${r.export.mesh.triangles} triangles, ${r.export.mesh.watertight ? 'watertight' : 'NOT watertight'}.`, 'ok');
      ok = true;
    } catch (e) { status(e.message, 'err'); }
    this.load();
    return ok;
  }

  // Hand the reconstruction to the brush sculpt tool (Rockscape tab).
  async sculptIt() {
    const used = this.st.used_as_rockscape && !this.st.recon?.stale;
    if (!used || confirm('Use the current reconstruction (with its edits) as the rockscape first? Choose Cancel to sculpt the rockscape the project has now.')) {
      if (!(await this.useAsRockscape())) return;
    }
    this.app.setTab('rocks');
    await this.app.designer.sculpt.enter({ fresh: true });
  }

  // ---------------------------------------------------------------- phone link
  async phoneDialog() {
    let st;
    try { st = await busy('opening the phone link', () => api.post('/api/phone/start', { project: this.app.pid })); }
    catch (e) { status(e.message, 'err'); return; }
    const box = el('div', { class: 'phone-dlg' });
    const recv = el('div', { class: 'list' });
    let seen = (st.received || []).length, timer = null;
    const stop = async () => { clearInterval(timer); try { await api.post('/api/phone/stop'); } catch (e) { /* already stopped */ } close(); status('phone link stopped', 'ok'); };
    const paint = (s) => {
      clear(recv);
      if (!s.received?.length) recv.append(el('div', { class: 'empty' }, 'Waiting for photos…'));
      for (const r of [...s.received].reverse()) recv.append(el('div', { class: 'item' }, el('span', {}, `✓ ${r.title}`), el('span', { class: 'sub' }, `${r.view} · ${r.size.join(' × ')} px`)));
    };
    const qr = el('div', { class: 'qr' });
    qr.innerHTML = st.qr_svg || '';
    box.append(el('h2', {}, 'Take the photos with your phone'),
      st.warning ? el('div', { class: 'warnbox' }, st.warning) : '',
      el('div', { class: 'phone-grid' }, qr, el('div', {},
        el('ol', {},
          el('li', {}, 'Connect the phone to the same Wi-Fi as this computer.'),
          el('li', {}, 'Open the camera app and point it at the QR code; tap the link.'),
          el('li', {}, 'Follow the shot list on the phone. Each photo appears in the list below and in the Photos tab.')),
        el('div', { class: 'hint' }, 'Or type this address in the phone browser:'),
        el('div', { class: 'mono' }, st.url || '–'),
        st.urls?.length > 1 ? el('div', { class: 'hint' }, `If it does not load, try: ${st.urls.slice(1).join('  ·  ')}`) : '',
        el('div', { class: 'hint' }, 'If Windows asks whether Python may use the network, allow Private networks. The link works only on your home network, only for this project, and stops after an hour unused.'))),
      el('h3', {}, 'Received'), recv,
      el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: () => { clearInterval(timer); close(); } }, 'Done (keep the link open)'),
        el('button', { onclick: stop }, 'Stop the link')));
    paint(st);
    const close = modal(box, { wide: true, sticky: true });
    timer = setInterval(async () => {
      if (!box.isConnected) { clearInterval(timer); return; }
      let s;
      try { s = await api.get('/api/phone/status'); } catch (e) { return; }
      if (!s.active) { clearInterval(timer); paint({ received: [] }); recv.prepend(el('div', { class: 'warnbox' }, 'The link has stopped.')); return; }
      paint(s);
      if (s.received.length !== seen) { seen = s.received.length; await this.load(); status(`photo received from the phone: ${s.received[s.received.length - 1].view}`, 'ok'); }
    }, 2000);
  }

  // ---------------------------------------------------------------- panel
  renderPanelOnly() { if (this.app.tab === 'photos') this.render(); }

  render() {
    STYLE = this.app.ed?.tank?.style || null;
    const app = this.app, P = app.panel, U = app.units;
    clear(P);
    P.append(el('h2', {}, 'Rockscape from photos'));
    P.append(el('div', { class: 'hint' }, 'For an existing tank: photograph it from the top (pumps off, still surface) and from any sides; click the tank corners on each photo; check the rock outline; carve the 3-D rock; fix caves and arches; use it as the rockscape.'));
    if (!this.st) { P.append(el('div', { class: 'errbox' }, 'Photo session not available (fix the tank dimensions first).')); return; }
    if (this.st.notice) P.append(el('div', { class: 'warnbox' }, this.st.notice));
    if (this.caps) P.append(el('div', { class: 'hint' }, `Segmentation: ${this.caps.sam?.length ? 'SAM (' + this.caps.sam.join(', ') + ') + GrabCut' : 'GrabCut (SAM model not installed: optional, the classical method is used)'}.`));

    // photo list
    const kindSel = el('select', {}, ...Object.entries(KIND_LABEL).map(([k, t]) => el('option', { value: k }, t)));
    const file = el('input', { type: 'file', accept: '.jpg,.jpeg,.png,.tif,.tiff,.webp', hidden: true, onchange: async () => {
      const f = file.files[0]; file.value = '';
      if (!f) return;
      const fd = new FormData(); fd.append('file', f); fd.append('kind', kindSel.value);
      try { const r = await busy(`uploading ${f.name}`, () => api.form(`${this.base()}/views`, fd)); this.st = r.state; this.select(r.view.id); status(`added ${r.view.id} (${r.view.width} × ${r.view.height} px${r.view.scale < 1 ? ', reduced from ' + r.view.original_size.join(' × ') : ''})`, 'ok'); }
      catch (e) { status(e.message, 'err'); }
    } });
    P.append(el('h3', {}, `1 Photos (${this.st.views.length})`, el('span', {}, kindSel, ' ', el('button', { class: 'small', onclick: () => file.click() }, '+ Add photo'), file)));
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: () => this.phoneDialog(),
      title: 'Show a QR code: your phone opens a guided photo page and every photo lands here' }, '📱 Use my phone')));
    const list = el('div', { class: 'list' });
    if (!this.st.views.length) list.append(el('div', { class: 'empty' }, 'No photos yet. A top view is needed; front and one side are recommended; an extra oblique photo reduces the overfill.'));
    for (const v of this.st.views) {
      const cal = v.calibration ? `✓ ${num(v.calibration.rms_px, 2)} px` : (Object.keys(v.clicks).length ? `${Object.keys(v.clicks).length} pts` : 'not calibrated');
      list.append(el('div', { class: 'item' + (v.id === this.sel ? ' sel' : ''), onclick: () => this.select(v.id) },
        el('span', {}, v.id), el('span', { class: 'sub' }, `${cal} · ${v.mask ? 'mask ✓' : 'no mask'}`)));
    }
    P.append(list);
    P.append(el('div', { class: 'seg', style: 'margin:8px 0' },
      el('button', { class: this.display === '2d' ? 'on' : '', onclick: () => { this.display = '2d'; this.show(); this.render(); } }, 'Photo editor'),
      el('button', { class: this.display === '3d' ? 'on' : '', disabled: !this.st.recon, onclick: () => { this.display = '3d'; this.show(); this.render(); } }, '3D result')));

    const v = this.view();
    if (this.display === '2d' && v) this._photoPanel(P, v);
    this._reconPanel(P);
  }

  _photoPanel(P, v) {
    P.append(el('div', { class: 'btns' },
      el('div', { class: 'seg' },
        el('button', { class: this.sub === 'calibrate' ? 'on' : '', onclick: () => { this.sub = 'calibrate'; this.canvas.draw(); this.render(); } }, '2 Calibrate'),
        el('button', { class: this.sub === 'mask' ? 'on' : '', disabled: !v.calibration && !v.mask, onclick: () => { this.sub = 'mask'; this.canvas.draw(); this.render(); } }, '3 Rock mask')),
      el('button', { class: 'small', onclick: () => this.canvas.fit(), title: 'Fit the photo to the view (F)' }, 'Fit'),
      el('button', { class: 'small danger', onclick: async () => {
        if (!confirm(`Remove photo ${v.id} (clicks and mask too)?`)) return;
        try { this.st = await api.del(`${this.base()}/views/${v.id}`); this.sel = null; await this.load(false); } catch (e) { status(e.message, 'err'); }
      } }, 'Remove photo')));
    if (this.sub === 'calibrate') this._calibrationPanel(P, v); else this._maskPanel(P, v);
  }

  // "Where to click" diagram: the listed points on a sketch of this kind of view; the
  // point being placed is pink, placed ones green. Clicking a dot selects that point.
  _sketch(v) {
    const kind = v.kind, act = this.active;
    const L = [];
    if (kind === 'top') {
      L.push('<rect x="12" y="12" width="216" height="112" fill="#2b3b48" stroke="#ddd" stroke-width="2"/>',
        '<rect x="40" y="32" width="160" height="72" fill="#c9b48a" opacity=".55" stroke="#ffb070"/>',
        '<text x="120" y="133" fill="#9ab" font-size="9" text-anchor="middle">front</text>');
    } else {
      L.push('<rect x="12" y="12" width="216" height="112" fill="#1f4a73" stroke="#e8e8e8" stroke-width="1.5"/>',
        TOP_FRAMED() ? '<rect x="12" y="12" width="216" height="7" fill="#111"/>' : '', BASE_FRAMED() ? '<rect x="12" y="116" width="216" height="8" fill="#111"/>' : '',
        '<rect x="48" y="34" width="144" height="70" fill="#245a8a" stroke="#c58cff" stroke-width="1"/>',
        '<line x1="12" y1="12" x2="48" y2="34" stroke="#8aa"/><line x1="228" y1="12" x2="192" y2="34" stroke="#8aa"/>',
        '<line x1="12" y1="124" x2="48" y2="104" stroke="#8aa"/><line x1="228" y1="124" x2="192" y2="104" stroke="#8aa"/>',
        '<line x1="48" y1="42" x2="192" y2="42" stroke="#3ad0ff"/>', '<rect x="48" y="96" width="144" height="8" fill="#c9b48a" opacity=".7"/>',
        '<text x="120" y="70" fill="#cde" font-size="9" text-anchor="middle">back glass (seen through the water)</text>',
        `<text x="120" y="133" fill="#9ab" font-size="8" text-anchor="middle">${STYLE === null ? 'black bands = frame, if your tank has one' : STYLE === 'rimmed' ? 'black bands = your frame: click its outside corners' : STYLE === 'rimless_top' ? 'top: glass corners · bottom: base-frame corners' : 'no frame: click the glass corners'}</text>`);
    }
    const names = new Set([...this.points().map(p => p.name), ...Object.keys(v.clicks), act].filter(Boolean));
    for (const n of names) {
      const p = sketchPos(kind, n); if (!p) continue;
      const on = n === act, done = !!v.clicks[n];
      if (n.startsWith('seam_')) L.push(`<line x1="${p[0]}" y1="${p[1] - 9}" x2="${p[0]}" y2="${p[1] + 9}" stroke="${on ? '#ff00d0' : '#c58cff'}" stroke-width="${on ? 4 : 2.5}"/>`);
      L.push(`<circle data-n="${n}" cx="${p[0]}" cy="${p[1]}" r="${on ? 7 : 5}" fill="${on ? '#ff00d0' : done ? '#28eb78' : '#fff'}" stroke="#000" stroke-width="1" style="cursor:pointer"><title>${pointLabel(n)}</title></circle>`);
    }
    const box = el('div', { class: 'calsketch' });
    box.innerHTML = `<svg viewBox="0 0 240 136" width="100%" style="max-width:300px;display:block">${L.join('')}</svg>`;
    box.addEventListener('click', (e) => { const n = e.target.getAttribute?.('data-n'); if (n) { this.active = n; this.canvas.draw(); this.render(); } });
    return box;
  }

  _sizeWarning(sc) {
    const U = this.app.units, fl = (m) => U.fmtv(m, 'length', 3) + ' ' + U.unit('length');
    const parts = [];
    if (sc.height_changed) parts.push(`about ${fl(sc.external_m[2])} tall (± ${fl(sc.height_std_m)}), not the ${fl(sc.external_now_m[2])} entered`);
    if (sc.width_changed) parts.push(`about ${fl(sc.external_m[1])} front to back (± ${fl(sc.width_std_m)}), not the ${fl(sc.external_now_m[1])} entered`);
    const v = this.view();
    return el('div', { class: 'errbox' }, el('b', {}, 'The tank size may be wrong. '),
      `This photo fits better with a tank ${parts.join(' and ')} (fit error ${num(sc.rms_now_px, 2)} → ${num(sc.rms_px, 2)} px). Measure the outside of the tank with a tape and correct it on the Tank tab (then press Save): the photos are calibrated again automatically. One photo judges depth only roughly (glass thickness and clicks shift it by about an inch), so if your tape says otherwise, trust the tape.`,
      el('div', { class: 'btns', style: 'margin-top:6px' }, el('button', { class: 'small', onclick: () => this.sendClicks(v.clicks, undefined, true) }, 'My measurements are right: stop asking')));
  }

  // The Photos tab calibrates with the SAVED project. Say so when the Tank tab has
  // different, unsaved dimensions, and offer to save.
  _unsavedTank() {
    const st = this.st?.tank, t = this.app.ed?.tank;
    if (!st || !t) return null;
    const gb = st.bottom_glass ?? st.glass;
    const saved = [st.length + 2 * st.glass, st.width + 2 * st.glass, st.height + gb];
    const now = [t.length, t.width, t.height];
    if (!saved.some((x, i) => Math.abs(x - now[i]) > 5e-4)) return null;
    const U = this.app.units, f = (a) => a.map(x => U.fmtv(x, 'length', 3)).join(' × ') + ' ' + U.unit('length');
    return el('div', { class: 'errbox' }, el('b', {}, 'Unsaved tank size. '),
      `The Tank tab says ${f(now)}, but photos use the saved project (${f(saved)}). Save to use the new size; the photos are calibrated again with your clicks.`,
      el('div', { class: 'btns', style: 'margin-top:6px' }, el('button', { class: 'primary small', onclick: async () => { await this.app.save(); await this.load(); } }, 'Save project now')));
  }

  _calibrationPanel(P, v) {
    const ut = this._unsavedTank();
    if (ut) P.append(ut);
    P.append(selectRow('Your tank', [['', '— choose the style —'], ...STYLE_OPTS], STYLE || '', (m) => {
      if (!this.app.ed) return;
      this.app.ed.tank.style = m || null; this.app.changed('tank'); STYLE = m || null; this.canvas.draw(); this.render();
    }, { title: 'Tells the calibration which corners to ask for: the frame corners, or the glass corners. Saved with the project (Tank tab).' }));
    const topTxt = STYLE === null ? 'the top corners (top of the frame, if your tank has one)' : TOP_FRAMED() ? 'the top outside corners of the black frame' : 'the top corners of the glass';
    const baseTxt = STYLE === null ? 'where the tank sits on the stand' : BASE_FRAMED() ? 'the bottom outside corners of the base frame, where it sits on the stand' : 'the bottom corners of the glass, where it sits on the stand';
    P.append(el('div', { class: 'hint' }, v.kind === 'top'
      ? `Click ${topTxt} (4), then the 4 corners where the sand meets the glass. The pink point is next; skip any you cannot see.`
      : `Click ${topTxt} and ${baseTxt} at both ends. Then, through the water: the back corners at the water line, and anywhere on the back vertical corner joints. The pink point is next (see the sketch); skip any you cannot see.`));
    const sc0 = v.calibration?.size_check;
    if (sc0 && sc0.suggest) P.append(this._sizeWarning(sc0));
    if (this.active) P.append(el('div', { class: 'okbox', style: 'margin:4px 0' }, `Next: ${pointLabel(this.active)}`));
    P.append(this._sketch(v));
    const old = Object.keys(v.clicks).filter(n => /^(rim|bottom)_/.test(n));
    if (old.length) P.append(el('div', { class: 'warnbox' }, `This photo has inside-corner clicks from the old list (${old.map(pointLabel).join(', ')}). On a tank with a frame those corners are hidden, so clicks on the frame pull the fit off. Remove them (×) or press Clear all clicks and use the list below.`));
    const f = v.focal || { mode: 'free' };
    const opts = [['free', v.exif_f35_mm ? 'solve from the clicks' : 'solve from the clicks (phone camera, 26 mm, as a guide)'], ['px', 'known, in pixels'], ['mm35', 'known, 35-mm equivalent (mm)']];
    if (v.exif_f35_mm) opts.splice(1, 0, ['exif', `from EXIF: ${num(v.exif_f35_mm, 3)} mm (35-mm eq.) = ${num(v.exif_f_px, 4)} px`]);
    P.append(selectRow('Focal length', opts, f.mode, (m) => this.sendClicks(v.clicks, { mode: m, value: f.value }),
      { title: 'Photos sent from a phone browser lose the focal length; the fit then starts from a typical phone main camera (26 mm equivalent) and adjusts it' }));
    if (f.mode === 'px' || f.mode === 'mm35') P.append(numRow(f.mode === 'px' ? 'Focal length (px)' : 'Focal length (35-mm eq.)', this.app.units, null, () => f.value, (val) => this.sendClicks(v.clicks, { mode: f.mode, value: val }), { unit: f.mode === 'px' ? 'px' : 'mm', step: 1, min: 1 }));
    const res = {};
    for (const p of v.calibration?.points || []) res[p.name] = p.residual_px;
    for (const p of v.calibration?.seams || []) res[p.name] = p.residual_px;
    const t = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Point'), el('th', {}, 'Clicked'), el('th', {}, 'Fit error'), el('th', {})));
    const row = (name, hint) => {
      const c = v.clicks[name];
      return el('tr', { class: 'click', title: hint || '', style: name === this.active ? 'background:#fde6fa' : '', onclick: () => { this.active = name; this.canvas.draw(); this.render(); } },
        el('td', {}, pointLabel(name), hint ? el('div', { class: 'hint', style: 'margin:0' }, hint) : null), el('td', {}, c ? '✓' : (name === this.active ? '← click' : '')),
        el('td', {}, res[name] !== undefined ? `${num(res[name], 2)} px` : ''),
        el('td', {}, c ? el('button', { class: 'small', title: 'remove this point', onclick: (e) => { e.stopPropagation(); this.removePoint(name); } }, '×') : ''));
    };
    for (const p of this.points()) t.append(row(p.name, styleHint(p.name, p.hint)));
    P.append(t);
    const rec = new Set(this.points().map(p => p.name));
    const extra = this.st.all_points.filter(p => !rec.has(p.name));
    P.append(el('details', {}, el('summary', {}, 'More points (if visible): a second point on each corner joint, sand line, inside corners'),
      el('table', { class: 'grid' }, ...extra.map(p => row(p.name, styleHint(p.name, p.hint))))));
    P.append(el('div', { class: 'btns' }, el('button', { disabled: !Object.keys(v.clicks).length, onclick: () => this.clearClicks() }, 'Clear all clicks')));
    const cal = v.calibration;
    if (v.calib_error) P.append(el('div', { class: cal ? 'warnbox' : 'errbox' }, v.calib_error));
    if (cal) {
      P.append(el('div', { class: cal.rms_px <= 3 ? 'okbox' : 'warnbox' }, `Calibrated: the model lines sit on your clicks within ${num(cal.rms_px, 2)} px on average (worst ${num(cal.max_px, 2)} px). Under 3 px is good.`));
      P.append(kvTable([
        ['Focal length', `${num(cal.focal_px, 4)} px${cal.param_std?.f_px ? ' ± ' + num(cal.param_std.f_px, 2) : ' (fixed)'}`],
        ['Camera position x, y, z', cal.camera_centre_m.map(c => this.app.units.fmtv(c, 'length', 3)).join(', ') + ' ' + this.app.units.unit('length')],
        ['Camera position 1σ', cal.camera_centre_std_mm.map(c => num(c, 2)).join(', ') + ' mm'],
        ['Same points without refraction', cal.pinhole_only_rms_px !== null ? `${num(cal.pinhole_only_rms_px, 2)} px RMS` : '–', 'how much the refraction model matters for this photo']]));
      for (const w of cal.warnings || []) P.append(el('div', { class: 'warnbox' }, w));
      P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: () => { this.sub = 'mask'; this.canvas.draw(); this.render(); } }, 'Next: rock mask →')));
    }
  }

  _maskPanel(P, v) {
    P.append(el('div', { class: 'hint' }, 'Green = rock. Get an automatic suggestion, then correct it: paint with the brush (drag), or click polygon corners (double-click or Enter to close). Equipment in front of the rock, shadows and coralline on the glass usually need a few strokes. Ctrl+Z undoes the last mask edit.'));
    P.append(selectRow('Find rock by', [['auto', 'texture (side photos), colour (top photo)'], ['texture', 'texture: rough rock vs smooth water'], ['colour', 'colour difference from the water']], this.cue || 'auto', (m) => { this.cue = m; },
      { title: 'Under blue reef lights rock and water have nearly the same colour, so side photos use texture: porous rock is rough, water and glass are smooth. Only rough areas that reach the sand count, so hanging equipment is left out.' }));
    {
      const sv = v.mask_sensitivity ?? this.sensitivity ?? 50;
      const rng = el('input', { type: 'range', min: 0, max: 100, step: 5, value: sv, title: 'Like a selection tolerance: higher takes more of the rock edge (and of what touches it), lower less. Only moves the outline near the rock, so it cannot fill the open water.' });
      const val = el('span', { class: 'val' }, `${sv}`);
      rng.addEventListener('input', () => { val.textContent = rng.value; });
      rng.addEventListener('change', () => this.setSensitivity(+rng.value));
      P.append(el('div', { class: 'layer' }, el('span'), el('span', {}, 'Sensitivity'), el('span', {}, el('span', { class: 'hint', style: 'margin:0 4px 0 0' }, 'less'), rng, el('span', { class: 'hint', style: 'margin:0 0 0 4px' }, 'more'), ' ', val)));
    }
    P.append(el('div', { class: 'btns' },
      el('button', { class: 'primary', disabled: !v.calibration, onclick: () => this.suggest(false) }, 'Suggest mask (this photo)'),
      el('button', { disabled: !this.st.views.some(x => x.calibration), onclick: () => this.suggest(true), title: 'Top view first; its outline then guides the side views. Masks you edited are kept.' }, 'Suggest all')));
    if (v.mask_method) P.append(el('div', { class: 'hint' }, `Mask: ${v.mask_method}${v.mask_edited ? ' + your edits' : ''}${v.mask_fraction !== undefined ? ` · ${num(100 * v.mask_fraction, 3)} % of the photo` : ''}`));
    for (const n of v.mask_notes || []) P.append(el('div', { class: 'hint', style: 'margin:0' }, '· ' + n));
    const tools = [['brush+', 'Brush +'], ['brush-', 'Brush −'], ['poly+', 'Polygon +'], ['poly-', 'Polygon −']];
    P.append(el('div', { class: 'seg', style: 'margin:6px 0' }, ...tools.map(([k, t]) => el('button', { class: this.maskTool === k ? 'on' : '', onclick: () => { this.maskTool = k; this.canvas.poly = []; this.canvas.draw(); this.render(); } }, t))));
    const rng = el('input', { type: 'range', min: 2, max: 80, step: 1, value: this.brush });
    const rv = el('span', { class: 'val' }, `${this.brush} px`);
    rng.addEventListener('input', () => { this.brush = +rng.value; rv.textContent = `${this.brush} px`; });
    P.append(el('div', { class: 'layer' }, el('span'), el('span', {}, 'Brush radius'), el('span', {}, rng, ' ', rv)));
    const op = el('input', { type: 'range', min: 0, max: 1, step: 0.05, value: this.canvas.opacity });
    op.addEventListener('input', () => { this.canvas.opacity = +op.value; this.canvas.draw(); });
    P.append(el('div', { class: 'layer' }, el('span'), el('span', {}, 'Mask opacity'), el('span', {}, op)));
    if (this.maskTool.startsWith('poly')) P.append(el('div', { class: 'btns' }, el('button', { class: 'small', onclick: () => this.finishPolygon() }, 'Close polygon (Enter)'), el('button', { class: 'small', onclick: () => { this.canvas.poly = []; this.canvas.draw(); } }, 'Cancel (Esc)')));
    P.append(el('div', { class: 'btns' },
      el('button', { class: 'small', onclick: () => this.maskOps([{ op: 'fill_holes' }]) }, 'Fill holes'),
      el('button', { class: 'small', onclick: () => this.maskOps([{ op: 'remove_small', min_area: 200 }]) }, 'Remove specks'),
      el('button', { class: 'small', onclick: () => this.maskOps([{ op: 'smooth', radius: 2 }]) }, 'Smooth'),
      el('button', { class: 'small', onclick: () => this.maskOps(null, 'mask_undo') }, 'Undo'),
      el('button', { class: 'small danger', onclick: () => { if (confirm('Clear the whole mask?')) this.maskOps([{ op: 'clear' }]); } }, 'Clear')));
  }

  _reconPanel(P) {
    const st = this.st, U = this.app.units, rec = st.recon;
    P.append(el('h3', {}, '4 Reconstruct (carve)'));
    P.append(el('div', { class: 'hint' }, `Ready: ${st.ready_views.length ? st.ready_views.join(', ') : 'none'} (a photo needs a calibration and a mask).`));
    const ge = st.grid_estimate;
    P.append(selectRow('Voxel size', DX_CHOICES.map(([d, t]) => [String(d), t]), String(st.settings.dx), async (val) => {
      try { this.st = { ...this.st, ...(await api.put(`${this.base()}/settings`, { dx: parseFloat(val) })) }; } catch (e) { status(e.message, 'err'); }
      this.render();
    }, { title: 'Use at most the CFD cell size of the runs you plan (Standard = 5 mm)' }));
    if (ge) P.append(el('div', { class: 'hint' }, `${ge.shape.join(' × ')} = ${num(ge.voxels / 1e6, 3)} M voxels, about ${Math.max(5, Math.round(ge.est_seconds_per_view * Math.max(st.ready_views.length, 1)))} s.`));
    P.append(numRow('Mask tolerance', U, null, () => st.settings.tolerance_px, async (val) => {
      try { this.st = { ...this.st, ...(await api.put(`${this.base()}/settings`, { tolerance_px: Math.round(val) })) }; } catch (e) { status(e.message, 'err'); }
    }, { unit: 'px', step: 1, min: 0, max: 10, title: 'Masks are grown by this many pixels before carving to absorb calibration error (can only add volume)' }));
    this.jobBox = el('div');
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', disabled: !st.ready_views.length || !!this.job, onclick: () => this.carve() }, rec ? 'Carve again' : 'Carve')), this.jobBox);
    this.renderJobBox();
    if (!rec) return;
    if (rec.stale) P.append(el('div', { class: 'warnbox' }, 'Photos, calibrations, masks or settings changed since this carving: carve again to update.'));
    const s = rec.stats;
    P.append(kvTable([['Voxels occupied', `${s.occupied.toLocaleString()} of ${s.voxels.toLocaleString()}`], ['Views used', s.views.join(', ')],
      ['Not seen by any view', s.unconstrained.toLocaleString(), 'left empty'], ['Carving time', `${num(s.time_s, 2)} s`]]));

    // edit tools
    const T = this.tool3d;
    P.append(el('h3', {}, '5 Fix caves and arches (3D)'));
    P.append(el('div', { class: 'hint' }, 'Silhouettes cannot see caves, arches, overhangs or gaps between rocks, so the carving fills them. Easiest: sculpt it with brushes (push to dig caves, pull to add, smooth, flatten shelves). Or use the precise shapes below: pick a tool, click points on the rock in the 3D view, then Apply.'));
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', onclick: () => this.sculptIt(), title: 'Uses the reconstruction as the rockscape and opens the brush sculpt tool in the Rockscape tab' }, 'Sculpt it with brushes →')));
    P.append(el('div', { class: 'btns' }, el('button', { class: this.display === '3d' ? 'on' : '', onclick: () => { this.display = '3d'; this.show(); this.render(); } }, 'Show 3D result')));
    const upd = () => { this.drawPicks(); };
    P.append(selectRow('Tool', [['capsule', 'tunnel along clicked points'], ['arch', 'arch opening'], ['sphere', 'sphere'], ['box', 'cube']], T.type, (val) => { T.type = val; this.picks = []; upd(); this.render(); }));
    P.append(selectRow('Action', [['carve', 'carve (open water)'], ['add', 'add rock']], T.op, (val) => { T.op = val; upd(); }));
    if (T.type === 'capsule' || T.type === 'sphere') P.append(numRow('Radius', U, 'length', () => T.radius, (val) => { T.radius = Math.max(val, 0.002); upd(); }, { min: 0 }));
    if (T.type === 'box') P.append(numRow('Cube size', U, 'length', () => T.size, (val) => { T.size = Math.max(val, 0.004); upd(); }, { min: 0 }));
    if (T.type === 'arch') {
      P.append(numRow('Opening width', U, 'length', () => T.span, (val) => { T.span = val; upd(); }, { min: 0 }));
      P.append(numRow('Opening height', U, 'length', () => T.rise, (val) => { T.rise = val; upd(); }, { min: 0, title: 'at least half the width (semicircular top)' }));
      P.append(numRow('Passage length', U, 'length', () => T.length, (val) => { T.length = val; upd(); }, { min: 0 }));
      P.append(numRow('Passage direction (plan)', U, null, () => T.azimuth, (val) => { T.azimuth = val; upd(); }, { unit: 'deg', step: 15, title: '0 = along x (left-right), 90 = front-back' }));
      P.append(el('div', { class: 'hint' }, 'Click the floor point at the middle of the opening (on the sand or rock base).'));
    }
    P.append(el('div', { class: 'hint num' }, this.picks.length ? `${this.picks.length} point(s): ` + this.picks.map(p => '(' + p.map(c => U.fmtv(c, 'length', 3)).join(', ') + ')').join(' ') : 'no points clicked yet'));
    P.append(el('div', { class: 'btns' }, el('button', { class: 'primary', disabled: !this.picks.length, onclick: () => this.applyEdit() }, 'Apply'),
      el('button', { class: 'small', onclick: () => { this.picks = []; upd(); this.render(); } }, 'Clear points')));
    if (st.edits.length) {
      const tl = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, `Edits (${st.edits.length})`), el('th', {})));
      st.edits.forEach((e, i) => tl.append(el('tr', {}, el('td', {}, `${e.op} ${e.shape.type}` + (e.shape.radius ? ` r = ${U.fmt(e.shape.radius, 'length', 3)}` : '')),
        el('td', {}, el('button', { class: 'small', onclick: () => this.setEdits(st.edits.filter((_, j) => j !== i)) }, '×')))));
      P.append(tl);
    }

    // comparison
    P.append(el('h3', {}, '6 Check against the photos'));
    P.append(el('div', { class: 'hint' }, 'Each photo with the reconstruction reprojected into it: green = agree, red = reconstruction only (overfill or a mask/calibration problem), blue = rock in the mask that the reconstruction lacks.'));
    P.append(el('div', { class: 'btns' }, el('button', { onclick: () => this.compare() }, 'Update comparison')));
    const cmp = rec.comparison;
    if (cmp) {
      const tc = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Photo'), el('th', {}, 'Silhouette IoU'), el('th', {}, 'Overfill px'), el('th', {}, 'Underfill px')));
      for (const [k, r] of Object.entries(cmp.views)) tc.append(el('tr', {}, el('td', {}, k), el('td', {}, num(r.silhouette_iou, 3)), el('td', {}, r.overfill_px.toLocaleString()), el('td', {}, r.underfill_px.toLocaleString())));
      P.append(tc);
      if (cmp.edits !== st.edits.length) P.append(el('div', { class: 'hint' }, 'Edits changed since this comparison: press Update comparison.'));
      for (const k of Object.keys(cmp.views)) {
        const src = `${this.base()}/recon/compare/${k}?t=${cmp.time}`;
        P.append(el('img', { class: 'fig', src, alt: k, onclick: () => modal(el('img', { src, style: 'max-width:100%' }), { wide: true }) }));
      }
    }

    // volume
    const vol = rec.volume;
    P.append(el('h3', {}, '7 Rock volume and weight'));
    P.append(el('div', { class: 'warnbox' }, 'Upper bound: silhouette carving cannot see caves, overhangs or gaps between stacked rocks and fills them. In the validation on a known synthetic rockscape, 3 photos (top, front, side) overestimated the volume by +47 %; a 4th, oblique photo helps, and so does opening known caves above.'));
    if (vol) {
      P.append(numRow('Bulk density (dry rock)', U, null, () => st.settings.bulk_density / 1000, async (val) => {
        try { rec.volume = await api.post(`${this.base()}/volume`, { bulk_density: val * 1000 }); st.settings.bulk_density = val * 1000; } catch (e) { status(e.message, 'err'); }
        this.render();
      }, { unit: 'kg/L', step: 0.05, min: 0.3, max: 3, title: 'Dry porous reef rock is typically 1.1-1.8 kg/L (default 1.4, an assumption)' }));
      P.append(numRow('Solid fraction', U, null, () => st.settings.solid_fraction, async (val) => {
        try { rec.volume = await api.post(`${this.base()}/volume`, { solid_fraction: val }); st.settings.solid_fraction = val; } catch (e) { status(e.message, 'err'); }
        this.render();
      }, { step: 0.05, min: 0.05, max: 1, title: 'Share of the reconstructed envelope that is rock (lower for loosely stacked rubble)' }));
      P.append(kvTable([
        ['Volume (after edits)', `${num(vol.volume_L, 3)} L (${num(vol.volume_gal, 3)} gal)`],
        ['Volume before edits', `${num(vol.volume_before_edits_m3 * 1000, 3)} L`],
        ['Dry weight estimate', `${U.fmt(vol.weight_kg, 'mass', 3)} (range ${U.fmtv(vol.weight_range_kg[0], 'mass', 3)}–${U.fmtv(vol.weight_range_kg[1], 'mass', 3)} ${U.unit('mass')})`, `bulk density ${num(vol.density_range_kg_m3[0] / 1000, 2)}-${num(vol.density_range_kg_m3[1] / 1000, 2)} kg/L`]]));
      P.append(el('div', { class: 'hint' }, 'Compare with what you know you bought: if the estimate is much higher, open hidden caves or add a photo.'));
    }
    P.append(el('h3', {}, '8 Use it'));
    P.append(el('div', { class: 'btns' },
      el('button', { class: 'primary', onclick: () => this.useAsRockscape() }, 'Use as rockscape'),
      el('a', { class: 'button', href: `${this.base()}/recon.stl` }, 'Download STL')));
    if (st.used_as_rockscape) P.append(el('div', { class: 'okbox' }, `Used as the project's rockscape (${st.used_as_rockscape.file}, ${new Date(st.used_as_rockscape.time * 1000).toLocaleString()}). Edits made after that need "Use as rockscape" again.`));
  }

  renderJobBox() {
    const b = this.jobBox;
    if (!b || !b.isConnected) return;
    clear(b);
    if (!this.job) return;
    b.append(el('div', { class: 'job running' }, el('div', { class: 'head' }, el('b', {}, 'Carving'), el('span', { class: 'status-running' }, this.job.status)),
      el('div', { class: 'progress' }, el('div', { style: `width:${this.job.pct || 0}%` })),
      el('div', { class: 'last' }, `${num(this.job.pct || 0, 3)} %  ${this.jobLine}`),
      el('div', { class: 'btns' }, el('button', { class: 'small danger', onclick: () => this.cancelJob() }, 'Cancel'))));
  }
}
