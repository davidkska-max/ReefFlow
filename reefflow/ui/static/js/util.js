// Small shared helpers: HTTP API, unit conversion, DOM builders, status line.

// ------------------------------------------------------------------ API
export class ApiError extends Error {}

async function check(r) {
  if (r.ok) return r;
  let msg = `${r.status} ${r.statusText}`;
  try { const j = await r.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not JSON */ }
  throw new ApiError(msg);
}

export const api = {
  async get(url) { return (await check(await fetch(url))).json(); },
  async send(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
    return (await check(r)).json();
  },
  post(url, body) { return this.send('POST', url, body ?? {}); },
  put(url, body) { return this.send('PUT', url, body); },
  del(url) { return this.send('DELETE', url); },
  async form(url, formData) { return (await check(await fetch(url, { method: 'POST', body: formData }))).json(); },
  // binary responses: little-endian float32/uint32 body + JSON X-Meta header
  async binary(url, opts) {
    const r = await check(await fetch(url, opts));
    const meta = JSON.parse(r.headers.get('X-Meta') || '{}');
    return { meta, buf: await r.arrayBuffer() };
  },
  async mesh(url, body) {
    const opts = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined;
    const r = await check(await fetch(url, opts));
    return unpackMesh(await r.arrayBuffer());
  },
};

export function unpackMesh(buf) {
  const head = new Uint32Array(buf, 0, 2);
  const nv = head[0], nf = head[1];
  const pos = new Float32Array(buf, 8, nv * 3);
  const idx = new Uint32Array(buf, 8 + nv * 12, nf * 3);
  return { positions: pos, index: idx, nv, nf };
}

// ------------------------------------------------------------------ units
// The core works in SI. The UI shows and accepts the project's display units.
export class Units {
  constructor(lib, system) { this.table = lib.units; this.display = lib.display; this.system = system || 'US'; }
  unit(kind) { return this.display[this.system][kind] || ({ time: 's', angle: 'deg' })[kind] || ''; }
  factor(unit) { const u = this.table[unit]; return u ? u[1] : 1; }
  // SI -> display number
  to(si, kind) { if (si === null || si === undefined || Number.isNaN(si)) return null; const u = this.unit(kind); return u && this.table[u] ? si / this.factor(u) : si; }
  // display number -> SI
  from(v, kind) { const u = this.unit(kind); return u && this.table[u] ? v * this.factor(u) : v; }
  fmt(si, kind, digits = 3) { const v = this.to(si, kind); return v === null ? '–' : `${num(v, digits)} ${this.unit(kind)}`; }
  fmtv(si, kind, digits = 3) { const v = this.to(si, kind); return v === null ? '–' : num(v, digits); }
}

export function num(v, digits = 3) {
  // 'digits' significant figures; never exponent notation for ordinary engineering values
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e7 || a < 1e-4) return v.toExponential(Math.max(digits - 1, 1));
  if (a >= 10 ** digits) return v.toFixed(0);
  const dec = Math.max(0, digits - 1 - Math.floor(Math.log10(a)));
  return v.toFixed(Math.min(dec, 8));
}

const DECIMALS = { in: 3, mm: 1, cm: 2, m: 4, ft: 3, GPH: 0, 'L/h': 0, 'cm/s': 2, 'in/s': 2, deg: 1, s: 2, '%': 1, 'kg/L': 2 };
export function inputValue(v, unit, precision) {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  if (precision) return String(+v.toPrecision(precision));
  const d = DECIMALS[unit];
  return d === undefined ? String(+v.toPrecision(5)) : String(+v.toFixed(d));
}

export function stepFor(unit) {
  return ({ in: 0.125, mm: 1, m: 0.01, ft: 0.1, GPH: 50, 'L/h': 100, 'cm/s': 0.1, 'in/s': 0.1, s: 1, deg: 1 })[unit] || 'any';
}

// ------------------------------------------------------------------ DOM
export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'html') e.innerHTML = v;
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

// Numeric field bound to an SI value via getter/setter, shown in display units.
export function numRow(label, units, kind, get, set, opts = {}) {
  const u = kind ? units.unit(kind) : (opts.unit || '');
  const inp = el('input', { type: 'number', step: opts.step ?? stepFor(u), title: opts.title });
  const v0 = get();
  inp.value = v0 === null || v0 === undefined ? '' : inputValue(kind ? units.to(v0, kind) : v0, u, opts.precision);
  if (opts.min !== undefined) inp.min = opts.min;
  if (opts.max !== undefined) inp.max = opts.max;
  if (opts.placeholder) inp.placeholder = opts.placeholder;
  inp.addEventListener('change', () => {
    if (inp.value === '' && opts.allowEmpty) { inp.classList.remove('bad'); set(null); return; }
    const v = parseFloat(inp.value);
    if (!Number.isFinite(v) || (opts.min !== undefined && v < opts.min) || (opts.max !== undefined && v > opts.max)) { inp.classList.add('bad'); return; }
    inp.classList.remove('bad');
    set(kind ? units.from(v, kind) : v);
  });
  return el('div', { class: 'row', title: opts.title }, el('label', {}, label), inp, el('span', { class: 'unit' }, u));
}

export function vecRow(label, units, kind, get, set, opts = {}) {
  const u = kind ? units.unit(kind) : (opts.unit || '');
  const vals = get();
  const inputs = [0, 1, 2].slice(0, opts.n || 3).map(i => {
    const inp = el('input', { type: 'number', step: opts.step ?? stepFor(u), title: (opts.axes || 'xyz')[i] });
    inp.value = inputValue(kind ? units.to(vals[i], kind) : vals[i], u, opts.precision);
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (!Number.isFinite(v)) { inp.classList.add('bad'); return; }
      inp.classList.remove('bad');
      const cur = get().slice();
      cur[i] = kind ? units.from(v, kind) : v;
      set(cur);
    });
    return inp;
  });
  const row = el('div', { class: 'row vec', title: opts.title }, el('label', {}, label), ...inputs, el('span', { class: 'unit' }, u));
  if ((opts.n || 3) === 2) row.style.gridTemplateColumns = '1fr auto auto 34px';
  return row;
}

export function selectRow(label, options, value, onChange, opts = {}) {
  const s = el('select', { title: opts.title });
  for (const o of options) {
    const [v, t] = Array.isArray(o) ? o : [o, o];
    const op = el('option', { value: v }, t);
    if (v === value) op.selected = true;
    s.appendChild(op);
  }
  s.addEventListener('change', () => onChange(s.value));
  return el('div', { class: 'row', title: opts.title }, el('label', {}, label), s, el('span', { class: 'unit' }, opts.unit || ''));
}

export function textRow(label, value, onChange, opts = {}) {
  const i = el('input', { type: 'text', value: value ?? '', style: 'width: 170px', title: opts.title });
  i.addEventListener('change', () => onChange(i.value.trim()));
  return el('div', { class: 'row', title: opts.title }, el('label', {}, label), i, el('span'));
}

export function checkRow(label, value, onChange, opts = {}) {
  const c = el('input', { type: 'checkbox' });
  c.checked = !!value;
  if (opts.disabled) c.disabled = true;
  c.addEventListener('change', () => onChange(c.checked));
  return el('div', { class: 'row', title: opts.title }, el('label', {}, label), c, el('span'));
}

export function kvTable(rows) {
  return el('table', { class: 'kv' }, rows.filter(Boolean).map(([k, v, t]) => el('tr', { title: t }, el('td', {}, k), el('td', {}, v))));
}

// ------------------------------------------------------------------ status line, busy, modal
let statusTimer = null;
export function status(msg, kind = '') {
  const s = document.getElementById('status');
  s.textContent = msg;
  s.className = kind;
  clearTimeout(statusTimer);
  if (kind === 'ok') statusTimer = setTimeout(() => { s.textContent = 'Ready.'; s.className = ''; }, 6000);
}

let busyCount = 0;
export async function busy(text, fn) {
  const b = document.getElementById('busy');
  busyCount++;
  document.getElementById('busyText').textContent = text;
  b.hidden = false;
  try { return await fn(); }
  finally { if (--busyCount <= 0) { busyCount = 0; b.hidden = true; } }
}

export function modal(content, opts = {}) {
  const m = document.getElementById('modal');
  const body = document.getElementById('modalBody');
  clear(body);
  body.parentElement.classList.toggle('wide', !!opts.wide);
  body.appendChild(content);
  m.hidden = false;
  const close = () => { m.hidden = true; clear(body); };
  m.onclick = (e) => { if (e.target === m && !opts.sticky) close(); };
  return close;
}
export function closeModal() { document.getElementById('modal').hidden = true; }

export function uid(prefix = 'e') { return prefix + Math.random().toString(36).slice(2, 8); }
export const deg = Math.PI / 180;
