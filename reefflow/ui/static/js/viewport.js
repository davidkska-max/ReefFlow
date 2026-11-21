// three.js viewport in tank coordinates: metres, origin at the inside bottom-left-front
// glass corner, x along the length, y front -> back, z up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

export class Viewport {
  constructor(container) {
    this.container = container;
    // transparent canvas: the deep-water gradient behind it is CSS (#view in app.css)
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.3, -1.4, 0.9);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = false;
    this.orbit.addEventListener('change', () => this.render());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7f70, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(0.4, -0.8, 2.0);
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-1, 1, 0.5);
    this.scene.add(fill);
    this.shell = new THREE.Group(); this.scene.add(this.shell);
    this.editorRoot = new THREE.Group(); this.scene.add(this.editorRoot);
    this.resultsRoot = new THREE.Group(); this.scene.add(this.resultsRoot);
    this.tc = new TransformControls(this.camera, this.renderer.domElement);
    this.tc.setSize(0.8);
    this.tc.addEventListener('change', () => this.render());
    this.tc.addEventListener('dragging-changed', (e) => { this.orbit.enabled = !e.value; this.dragging = e.value; });
    this.scene.add(this.tc);
    this.raycaster = new THREE.Raycaster();
    this.tank = null;
    this.pickables = [];
    this.onPick = null;
    this.onHover = null;
    this._down = null;
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => { this._down = [e.clientX, e.clientY]; });
    dom.addEventListener('pointerup', (e) => {
      if (!this._down || this.tc.dragging) { this._down = null; return; }
      const d = Math.hypot(e.clientX - this._down[0], e.clientY - this._down[1]);
      this._down = null;
      if (d < 5 && this.onPick) this.onPick(this.pick(e, typeof this.pickables === 'function' ? this.pickables() : this.pickables), e);
    });
    dom.addEventListener('pointermove', (e) => { if (this.onHover && !this.dragging && e.buttons === 0) this.onHover(e); });
    dom.addEventListener('pointerleave', () => { if (this.onHover) this.onHover(null); });
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth || 800, h = this.container.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this.renderer.render(this.scene, this.camera); });
  }

  ndc(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  pick(e, objects) {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const hits = this.raycaster.intersectObjects(objects.filter(o => o.visible), true);
    return hits.length ? hits[0] : null;
  }

  // ---------------------------------------------------------------- tank shell
  // t = editor tank (external dims + glass, SI). Draws the internal volume.
  setTank(t, overflow) {
    for (const c of [...this.shell.children]) { this.shell.remove(c); disposeTree(c); }
    const L = t.length - 2 * t.glass, W = t.width - 2 * t.glass, H = t.height - t.bottom_glass;
    const ws = H - t.water_gap, sand = t.sand_depth;
    this.tank = { L, W, H, ws, sand };
    const box = new THREE.BoxGeometry(L, W, H); box.translate(L / 2, W / 2, H / 2);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0x8ec9dc }));
    this.shell.add(edges);
    // outer glass edges
    const g = t.glass, gb = t.bottom_glass;
    const ob = new THREE.BoxGeometry(L + 2 * g, W + 2 * g, H + gb); ob.translate(L / 2, W / 2, (H + gb) / 2 - gb);
    this.shell.add(new THREE.LineSegments(new THREE.EdgesGeometry(ob), new THREE.LineBasicMaterial({ color: 0x6aa6bc, transparent: true, opacity: 0.55 })));
    // back and side panes faintly tinted (front left clear so the user looks in)
    const paneMat = new THREE.MeshBasicMaterial({ color: 0xa6dcee, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false });
    const back = new THREE.Mesh(new THREE.PlaneGeometry(L, H), paneMat); back.rotation.x = Math.PI / 2; back.position.set(L / 2, W, H / 2);
    const left = new THREE.Mesh(new THREE.PlaneGeometry(W, H), paneMat); left.rotation.set(Math.PI / 2, Math.PI / 2, 0); left.position.set(0, W / 2, H / 2);
    const right = left.clone(); right.position.x = L;
    this.shell.add(back, left, right);
    // sand
    if (sand > 0) {
      const sg = new THREE.BoxGeometry(L, W, sand); sg.translate(L / 2, W / 2, sand / 2);
      this.sandMesh = new THREE.Mesh(sg, new THREE.MeshStandardMaterial({ color: 0xe8dcc0, roughness: 1 }));
      this.sandMesh.name = 'sand';
      this.shell.add(this.sandMesh);
    } else this.sandMesh = null;
    // sand top plane (used for snapping and for the sand-shear texture)
    const st = new THREE.Mesh(new THREE.PlaneGeometry(L, W), new THREE.MeshBasicMaterial({ visible: false }));
    st.position.set(L / 2, W / 2, sand); st.name = 'sandTop';
    this.sandTop = st; this.shell.add(st);
    // water surface
    const wg = new THREE.PlaneGeometry(L, W);
    this.water = new THREE.Mesh(wg, new THREE.MeshBasicMaterial({ color: 0x5fd0f2, transparent: true, opacity: 0.10, side: THREE.DoubleSide, depthWrite: false }));
    this.water.position.set(L / 2, W / 2, ws);
    this.shell.add(this.water);
    const wl = new THREE.LineSegments(new THREE.EdgesGeometry(wg), new THREE.LineBasicMaterial({ color: 0x6fdcf6 }));
    wl.position.copy(this.water.position);
    this.shell.add(wl);
    // overflow box
    if (overflow) this.shell.add(overflowMesh(overflow, this.tank));
    // labels
    this.shell.add(textSprite('FRONT', L / 2, -0.05, 0, 0.035));
    this.shell.add(textSprite('x', L + 0.05, -0.02, 0, 0.03, '#ff7b7b'));
    this.shell.add(textSprite('y', -0.03, W + 0.04, 0, 0.03, '#7be38a'));
    this.shell.add(axesLines(Math.min(L, W) * 0.25));
    this.orbit.target.set(L / 2, W / 2, H * 0.4);
    if (!this._framed) { this.setView('3d'); this._framed = true; }
    this.render();
  }

  setView(v) {
    if (!this.tank) return;
    const { L, W, H } = this.tank;
    const c = new THREE.Vector3(L / 2, W / 2, H * 0.45);
    const d = Math.max(L, H * 1.6) * 1.35;
    const pos = { '3d': [c.x - 0.25 * L, c.y - d, c.z + 0.6 * d], top: [c.x, c.y - 0.0001, c.z + d * 1.1], front: [c.x, c.y - d, c.z], side: [c.x + d * 0.75, c.y, c.z] }[v];
    this.camera.position.set(...pos);
    this.orbit.target.copy(c);
    this.camera.up.set(0, 0, 1);
    this.orbit.update();
    this.render();
  }

  attach(obj, mode) {
    if (!obj) { this.tc.detach(); this.render(); return; }
    this.tc.attach(obj);
    if (mode) this.tc.setMode(mode);
    this.render();
  }
}

// Overflow footprint (same rule as reefflow.project.Overflow.box, wall form):
// o = {wall: back|front|left|right, center (along the wall), width (along), depth (out)}.
export function overflowWallLength(o, tank) { return (o.wall === 'left' || o.wall === 'right') ? tank.W : tank.L; }

export function overflowBox(o, tank) {
  const { L, W } = tank;
  const Lw = overflowWallLength(o, tank);
  const c = o.center ?? Lw / 2;
  const a0 = Math.max(c - o.width / 2, 0), a1 = Math.min(c + o.width / 2, Lw);
  if (o.wall === 'front') return { x0: a0, x1: a1, y0: 0, y1: o.depth };
  if (o.wall === 'left') return { x0: 0, x1: o.depth, y0: a0, y1: a1 };
  if (o.wall === 'right') return { x0: L - o.depth, x1: L, y0: a0, y1: a1 };
  return { x0: a0, x1: a1, y0: W - o.depth, y1: W };          // back (default)
}

export function overflowMesh(o, tank) {
  const { ws } = tank;
  const b = overflowBox(o, tank), sx = b.x1 - b.x0, sy = b.y1 - b.y0;
  const top = ws - o.crest_below_surface;
  const g = new THREE.BoxGeometry(sx, sy, top); g.translate(b.x0 + sx / 2, b.y0 + sy / 2, top / 2);
  const grp = new THREE.Group();
  grp.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x8aa3b0, transparent: true, opacity: 0.28, depthWrite: false })));
  grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: 0xa9c6d2 })));
  grp.add(textSprite('overflow', b.x0 + sx / 2, b.y0 + sy / 2, top + 0.03, 0.025, '#333'));
  return grp;
}

export function axesLines(s) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(s, 0, 0), new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, s, 0), new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, s)]);
  g.setAttribute('color', new THREE.Float32BufferAttribute([1, 0.4, 0.4, 1, 0.4, 0.4, 0.4, 0.88, 0.45, 0.4, 0.88, 0.45, 0.4, 0.6, 1, 0.4, 0.6, 1], 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
}

// Label that reads on the dark water backdrop: light text with a dark halo. Dark colours passed
// by callers (written for the old light background) are lightened, keeping their hue.
function readableOnDark(color) {
  const c = new THREE.Color(color);
  const lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  if (lum < 0.5) c.lerp(new THREE.Color(0xffffff), 0.78 - lum);
  return `#${c.getHexString()}`;
}

export function textSprite(text, x, y, z, h = 0.03, color = '#e6f3f8', bg = null) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const fs = 48, pad = 14;
  const font = `600 ${fs}px "Segoe UI Variable Text", "Segoe UI", system-ui, Arial`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 2 * pad;
  c.width = w; c.height = fs + 2 * pad;
  ctx.font = font;
  ctx.textBaseline = 'middle';
  if (bg) {
    const r = c.height / 2;
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.roundRect ? ctx.roundRect(0, 0, c.width, c.height, r) : ctx.rect(0, 0, c.width, c.height); ctx.fill();
    ctx.fillStyle = color;
  } else {
    ctx.lineJoin = 'round';
    ctx.lineWidth = 10;
    ctx.strokeStyle = 'rgba(4, 18, 27, 0.82)';
    ctx.strokeText(text, pad, c.height / 2);
    ctx.fillStyle = readableOnDark(color);
  }
  ctx.fillText(text, pad, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set(h * c.width / c.height, h, 1);
  s.position.set(x, y, z);
  s.renderOrder = 10;
  return s;
}

export function disposeTree(o) {
  o.traverse?.((c) => {
    c.geometry?.dispose?.();
    const m = c.material;
    if (Array.isArray(m)) m.forEach(x => { x.map?.dispose?.(); x.dispose?.(); });
    else if (m) { m.map?.dispose?.(); m.dispose?.(); }
  });
}

export function meshFromBuffers(positions, index, material) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (index && index.length) g.setIndex(new THREE.BufferAttribute(index, 1));
  g.computeVertexNormals();
  g.computeBoundingBox();
  return new THREE.Mesh(g, material);
}

export const ROCK_COLOR = 0xb9a688;
export function rockMaterial() { return new THREE.MeshStandardMaterial({ color: ROCK_COLOR, roughness: 0.95, metalness: 0 }); }
