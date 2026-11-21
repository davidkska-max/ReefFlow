// Node check of the sculpt brushes in static/js/sculpt.js (run by tests/test_sculpt_brush.py).
// usage: node sculpt_brush_check.mjs <work dir with meta.json + field.bin> <repo dir>
// Prints one JSON object with the measurements; the Python test asserts on them.
import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';

const [dir, repo] = process.argv.slice(2);
const js = path.join(repo, 'reefflow/ui/static/js');
const three = pathToFileURL(path.join(repo, 'reefflow/ui/static/vendor/three/three.module.js')).href;
fs.writeFileSync(path.join(dir, 'viewport_stub.mjs'),
  `import * as THREE from '${three}'; export function rockMaterial(){ return new THREE.MeshStandardMaterial(); } export function disposeTree(){}`);
const src = fs.readFileSync(path.join(js, 'sculpt.js'), 'utf8')
  .replace("from 'three'", `from '${three}'`)
  .replace("from './util.js'", `from '${pathToFileURL(path.join(js, 'util.js')).href}'`)
  .replace("from './viewport.js'", "from './viewport_stub.mjs'");
fs.writeFileSync(path.join(dir, 'sculpt_node.mjs'), src);
globalThis.window = { addEventListener() {} };
const S = await import(pathToFileURL(path.join(dir, 'sculpt_node.mjs')).href);
const THREE = await import(three);
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json')));
const bytes = new Uint8Array(fs.readFileSync(path.join(dir, 'field.bin')));
const app = { view: { editorRoot: new THREE.Group(), renderer: { domElement: { addEventListener() {}, removeEventListener() {} } }, render() {},
  orbit: { target: V(0.3, 0.15, 0.15) }, camera: { position: V(0.3, -0.6, 0.15) } },
  designer: { group: {} }, tab: 'x', units: null, ed: { designer: {} } };
const make = (o) => { const sc = new S.Sculpt(app); sc.F = new S.Field(meta, bytes); sc.nc = [1, 1, 1]; Object.assign(sc, o); return sc; };
const FWD = V(0, 1, 0);                     // looking through the front glass
const ray = (x, z) => ({ origin: V(x, -0.5, z), direction: FWD.clone() });
const frontY = (F, x, z) => { const h = F.raycast(V(x, -0.5, z), FWD); return h && !h.sand ? h.point.y : null; };
const frame = (sc, r, dt = 1 / 60) => sc._applyStroke(sc.F.raycast(r.origin, r.direction), dt, r);
const out = { speed_full_m_s: null };
const CX = 0.3, CZ = 0.15;                  // centre of the box's front face

// 1. hold still: speed at the centre and the dome profile
{
  const sc = make({ tool: 'raise', radius: 0.05, strength: 0.5 });
  const y0 = frontY(sc.F, CX, CZ), yh0 = frontY(sc.F, CX + 0.025, CZ), ye0 = frontY(sc.F, CX + 0.06, CZ);
  sc._startStroke();
  for (let i = 0; i < 30; i++) frame(sc, ray(CX, CZ));        // 0.5 s
  out.hold_half_s_centre_mm = (y0 - frontY(sc.F, CX, CZ)) * 1000;
  out.hold_half_s_half_radius_mm = (yh0 - frontY(sc.F, CX + 0.025, CZ)) * 1000;
  out.hold_half_s_outside_mm = (ye0 - frontY(sc.F, CX + 0.06, CZ)) * 1000;
  sc._endStroke();
}
// 2. lower, same
{
  const sc = make({ tool: 'lower', radius: 0.05, strength: 0.5 });
  const y0 = frontY(sc.F, CX, CZ);
  sc._startStroke();
  for (let i = 0; i < 30; i++) frame(sc, ray(CX, CZ));
  out.lower_half_s_centre_mm = (frontY(sc.F, CX, CZ) - y0) * 1000;
  sc._endStroke();
}
// 3. passes add up: one scrub pass vs two (same path, separate strokes)
{
  const pass = (sc) => { sc._startStroke(); for (let i = 0; i <= 60; i++) frame(sc, ray(0.24 + 0.12 * i / 60, CZ)); sc._endStroke(); };
  const sc = make({ tool: 'raise', radius: 0.05, strength: 0.5 });
  const y0 = frontY(sc.F, CX, CZ);
  pass(sc); out.scrub_one_pass_mm = (y0 - frontY(sc.F, CX, CZ)) * 1000;
  pass(sc); out.scrub_two_passes_mm = (y0 - frontY(sc.F, CX, CZ)) * 1000;
  // undo both strokes restores the field exactly
  sc._swap(sc.undo, sc.redo); sc._swap(sc.undo, sc.redo);     // what Undo does, without the status bar
  const F0 = new S.Field(meta, bytes);
  let diff = 0; for (let i = 0; i < F0.d.length; i++) if (Math.abs(F0.d[i] - sc.F.d[i]) > 1e-4) diff++;
  out.undo_differing_nodes = diff;
}
// 4. safety: Raise never removes rock and leaves the gap at the front glass; Lower never adds
{
  const sc = make({ tool: 'raise', radius: 0.05, strength: 1 });
  const before = sc.F.d.slice();
  sc._startStroke();
  for (let i = 0; i < 300; i++) frame(sc, ray(CX, CZ));        // 5 s: runs into the front glass
  sc._endStroke();
  const F = sc.F;
  let removed = 0, inGap = 0;
  for (let k = F.ks; k < F.nz; k++) for (let j = 0; j < F.ny; j++) for (let i = 0; i < F.nx; i++) {
    const id = i + F.nx * (j + F.ny * k);
    if (F.d[id] < before[id] - 1e-4) removed++;
    if (F.o[1] + j * F.dx < 0.01 && F.d[id] >= 127.5) inGap++;
  }
  out.raise_removed_nodes = removed;
  out.raise_rock_nodes_within_10mm_of_front_glass = inGap;
  out.raise_front_after_5s_mm = frontY(F, CX, CZ) * 1000;
  const sl = make({ tool: 'lower', radius: 0.05, strength: 1 });
  const b2 = sl.F.d.slice();
  sl._startStroke();
  for (let i = 0; i < 120; i++) frame(sl, ray(CX, CZ));
  sl._endStroke();
  let added = 0; for (let i = 0; i < b2.length; i++) if (sl.F.d[i] > b2[i] + 1e-4) added++;
  out.lower_added_nodes = added;
}
// 5. raising on bare sand grows rock up out of it (looking down)
{
  const sc = make({ tool: 'raise', radius: 0.04, strength: 1 });
  const r = { origin: V(0.08, 0.15, 0.5), direction: V(0, 0, -1) };
  const h0 = sc.F.raycast(r.origin, r.direction);
  sc._startStroke();
  for (let i = 0; i < 60; i++) frame(sc, r);
  sc._endStroke();
  const h1 = sc.F.raycast(r.origin, r.direction);
  out.sand_start_is_sand = !!(h0 && h0.sand);
  out.sand_mound_mm = h1 && !h1.sand ? (h1.point.z - h0.point.z) * 1000 : 0;
}
// 6. Tunnel still bores straight through in one click
{
  const sc = make({ tool: 'drill', radius: 0.02, strength: 0.5, dirMode: 'fixed', az: 90, el: 0, bore: true });
  sc._startStroke();
  frame(sc, ray(CX, CZ));
  out.tunnel_through_one_click = sc.stroke.done;
  sc._endStroke();
  out.tunnel_open_line = frontY(sc.F, CX, CZ) === null;
}
console.log(JSON.stringify(out));
