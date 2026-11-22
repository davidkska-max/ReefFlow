// Particle animation in the Results view (D-849): tracer particles carried by the run's
// time-averaged velocity field. The field (m/s, NaN in rock) comes from /flowfield; particles are
// integrated in the browser (midpoint rule, sub-stepped so no step moves more than half a cell),
// drawn as fading trails coloured by speed, and respawn when old, stuck in rock or out of the water.
// They show where the AVERAGE flow carries water; real water also swirls on top of it (turbulence).
import * as THREE from 'three';
import { api } from './util.js';

const SLOW = new THREE.Color(0x2f7fc1), MID = new THREE.Color(0x4fe3ee), FAST = new THREE.Color(0xfff2a6);

export class Particles {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.field = null;        // {U: Float32Array, nx, ny, nz, o: [3], h, vmax, fluid: Int32Array}
    this.key = null;
    this.opts = { count: 3000, speed: 1, trail: 16, release: 'everywhere', life: 8 };
    this.on = false;
    this.obj = null;
    this.raf = 0;
    this.pumps = [];          // [{p: [3], aim: [3]}]
  }

  async load(pid, rid) {
    const key = `${pid}/${rid}`;
    if (this.key === key && this.field) return this.field;
    const { meta, buf } = await api.binary(`/api/runs/${pid}/${rid}/flowfield`);
    const [nx, ny, nz] = meta.shape;
    const U = new Float32Array(buf);
    const fluid = [];
    const sp = [];
    for (let n = 0; n < nx * ny * nz; n++) {
      const u = U[3 * n];
      if (Number.isFinite(u)) { fluid.push(n); sp.push(Math.hypot(u, U[3 * n + 1], U[3 * n + 2])); }
    }
    sp.sort((a, b) => a - b);
    this.field = { U, nx, ny, nz, o: meta.origin, h: meta.dx, vmax: Math.max(sp[Math.floor(0.95 * (sp.length - 1))] || 0.05, 0.01),
      fluid: Int32Array.from(fluid) };
    this.key = key;
    return this.field;
  }

  // trilinear velocity at p (m); null when every surrounding node is rock / outside
  vel(x, y, z, out) {
    const F = this.field, h = F.h;
    const fx = (x - F.o[0]) / h, fy = (y - F.o[1]) / h, fz = (z - F.o[2]) / h;
    const i = Math.floor(fx), j = Math.floor(fy), k = Math.floor(fz);
    if (i < 0 || j < 0 || k < 0 || i >= F.nx - 1 || j >= F.ny - 1 || k >= F.nz - 1) return null;
    const tx = fx - i, ty = fy - j, tz = fz - k;
    let wsum = 0, ux = 0, uy = 0, uz = 0;
    for (let c = 0; c < 8; c++) {
      const di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
      const n = ((i + di) * F.ny + (j + dj)) * F.nz + (k + dk);
      const u = F.U[3 * n];
      if (!Number.isFinite(u)) continue;
      const w = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty) * (dk ? tz : 1 - tz);
      wsum += w; ux += w * u; uy += w * F.U[3 * n + 1]; uz += w * F.U[3 * n + 2];
    }
    if (wsum < 0.25) return null;               // mostly in rock: treat as solid
    out[0] = ux / wsum; out[1] = uy / wsum; out[2] = uz / wsum;
    return out;
  }

  _spawn(q) {
    const F = this.field, P = this.P, a = 3 * q;
    if (this.opts.release === 'pumps' && this.pumps.length && Math.random() < 0.85) {
      const pm = this.pumps[Math.floor(Math.random() * this.pumps.length)];
      const r = 0.012;
      P[a] = pm.p[0] + pm.aim[0] * 0.01 + (Math.random() - 0.5) * r;
      P[a + 1] = pm.p[1] + pm.aim[1] * 0.01 + (Math.random() - 0.5) * r;
      P[a + 2] = pm.p[2] + pm.aim[2] * 0.01 + (Math.random() - 0.5) * r;
    } else {
      const n = F.fluid[Math.floor(Math.random() * F.fluid.length)];
      const k = n % F.nz, j = Math.floor(n / F.nz) % F.ny, i = Math.floor(n / (F.nz * F.ny));
      P[a] = F.o[0] + (i + Math.random() - 0.5) * F.h;
      P[a + 1] = F.o[1] + (j + Math.random() - 0.5) * F.h;
      P[a + 2] = F.o[2] + (k + Math.random() - 0.5) * F.h;
    }
    this.age[q] = 0;
    this.life[q] = this.opts.life * (0.5 + Math.random());
    const T = this.opts.trail, b = q * T * 3;
    for (let t = 0; t < T; t++) { this.tr[b + 3 * t] = P[a]; this.tr[b + 3 * t + 1] = P[a + 1]; this.tr[b + 3 * t + 2] = P[a + 2]; }
  }

  _build() {
    this._disposeObj();
    const N = this.opts.count, T = this.opts.trail;
    this.P = new Float32Array(3 * N);
    this.age = new Float32Array(N);
    this.life = new Float32Array(N);
    this.spd = new Float32Array(N);
    this.tr = new Float32Array(N * T * 3);              // trail ring, newest first
    for (let q = 0; q < N; q++) { this._spawn(q); this.age[q] = Math.random() * this.life[q]; }
    const segs = N * (T - 1);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs * 6), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segs * 8), 4).setUsage(THREE.DynamicDrawUsage));
    const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    lines.frustumCulled = false;
    lines.renderOrder = 5;
    const hg = new THREE.BufferGeometry();
    hg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage));
    hg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage));
    const heads = new THREE.Points(hg, new THREE.PointsMaterial({ size: 3, sizeAttenuation: false, vertexColors: true, transparent: true, depthWrite: false }));
    heads.frustumCulled = false;
    heads.renderOrder = 6;
    this.obj = new THREE.Group();
    this.obj.add(lines, heads);
    this.lines = lines; this.heads = heads;
    this.root.add(this.obj);
  }

  step(dtReal) {
    const F = this.field, N = this.opts.count, T = this.opts.trail;
    const dt = Math.min(dtReal, 0.05) * this.opts.speed;
    const v1 = [0, 0, 0], v2 = [0, 0, 0], P = this.P;
    for (let q = 0; q < N; q++) {
      const a = 3 * q;
      this.age[q] += dt;
      let alive = this.age[q] < this.life[q];
      let rem = dt, guard = 0;
      while (alive && rem > 1e-6 && guard++ < 20) {
        if (!this.vel(P[a], P[a + 1], P[a + 2], v1)) { alive = false; break; }
        const sp = Math.hypot(v1[0], v1[1], v1[2]);
        const h = Math.min(rem, sp > 1e-6 ? 0.5 * F.h / sp : rem);
        const mx = P[a] + 0.5 * h * v1[0], my = P[a + 1] + 0.5 * h * v1[1], mz = P[a + 2] + 0.5 * h * v1[2];
        if (!this.vel(mx, my, mz, v2)) { alive = false; break; }
        P[a] += h * v2[0]; P[a + 1] += h * v2[1]; P[a + 2] += h * v2[2];
        this.spd[q] = Math.hypot(v2[0], v2[1], v2[2]);
        rem -= h;
      }
      if (!alive) { this._spawn(q); continue; }
      const b = q * T * 3;
      this.tr.copyWithin(b + 3, b, b + 3 * (T - 1));
      this.tr[b] = P[a]; this.tr[b + 1] = P[a + 1]; this.tr[b + 2] = P[a + 2];
    }
    this._upload();
  }

  _color(s, out) {
    const f = Math.min(s / this.field.vmax, 1);
    if (f < 0.5) out.copy(SLOW).lerp(MID, f / 0.5); else out.copy(MID).lerp(FAST, (f - 0.5) / 0.5);
    return out;
  }

  _upload() {
    const N = this.opts.count, T = this.opts.trail;
    const pos = this.lines.geometry.attributes.position.array, col = this.lines.geometry.attributes.color.array;
    const hp = this.heads.geometry.attributes.position.array, hc = this.heads.geometry.attributes.color.array;
    const c = new THREE.Color();
    let s = 0;
    for (let q = 0; q < N; q++) {
      this._color(this.spd[q], c);
      // fade in after a respawn and out before the end of life
      const life = Math.min(this.age[q] / 0.4, 1, (this.life[q] - this.age[q]) / 0.4);
      const b = q * T * 3;
      for (let t = 0; t < T - 1; t++, s++) {
        const p = s * 6, o = s * 8;
        pos[p] = this.tr[b + 3 * t]; pos[p + 1] = this.tr[b + 3 * t + 1]; pos[p + 2] = this.tr[b + 3 * t + 2];
        pos[p + 3] = this.tr[b + 3 * t + 3]; pos[p + 4] = this.tr[b + 3 * t + 4]; pos[p + 5] = this.tr[b + 3 * t + 5];
        const a0 = life * (1 - t / (T - 1)) * 0.95, a1 = life * (1 - (t + 1) / (T - 1)) * 0.95;
        col[o] = c.r; col[o + 1] = c.g; col[o + 2] = c.b; col[o + 3] = a0;
        col[o + 4] = c.r; col[o + 5] = c.g; col[o + 6] = c.b; col[o + 7] = a1;
      }
      hp[3 * q] = this.P[3 * q]; hp[3 * q + 1] = this.P[3 * q + 1]; hp[3 * q + 2] = this.P[3 * q + 2];
      hc[3 * q] = c.r * life; hc[3 * q + 1] = c.g * life; hc[3 * q + 2] = c.b * life;
    }
    this.lines.geometry.attributes.position.needsUpdate = true;
    this.lines.geometry.attributes.color.needsUpdate = true;
    this.heads.geometry.attributes.position.needsUpdate = true;
    this.heads.geometry.attributes.color.needsUpdate = true;
  }

  async start(pid, rid, pumps) {
    this.pumps = pumps || [];
    await this.load(pid, rid);
    this._build();
    this.on = true;
    let last = performance.now();
    const loop = (now) => {
      if (!this.on) return;
      if (this.app.tab !== 'results') { this.stop(); return; }
      this.step((now - last) / 1000);
      last = now;
      this.app.view.render();
      this.raf = requestAnimationFrame(loop);
    };
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(loop);
  }

  restart(pid, rid, pumps) { if (this.on) return this.start(pid, rid, pumps); }

  stop() {
    this.on = false;
    cancelAnimationFrame(this.raf);
    this._disposeObj();
    this.app.view.render();
  }

  forget() { this.stop(); this.field = null; this.key = null; }

  _disposeObj() {
    if (!this.obj) return;
    this.root.remove(this.obj);
    this.obj.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    this.obj = null;
  }
}
