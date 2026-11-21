// Run tab: pre-run tank-model summary + estimate, start runs, live progress (SSE), cancel.
import { api, el, clear, numRow, selectRow, textRow, checkRow, kvTable, num, status, busy } from './util.js';

function fmtTime(s) {
  if (s === null || s === undefined) return '–';
  if (s < 90) return `${num(s, 2)} s`;
  if (s < 5400) return `${num(s / 60, 3)} min`;
  return `${num(s / 3600, 3)} h`;
}

export class RunPanel {
  constructor(app) {
    this.app = app;
    this.opts = null;
    this.est = null;
    this.streams = new Map();     // job id -> EventSource
    this.logs = new Map();        // job id -> lines
    this.jobs = [];
    this.openLog = null;
    this.timer = null;
  }

  resetFor() {
    const s = this.app.ed.simulation;
    this.opts = { preset: s.preset || 'preview', lattice: s.lattice || 'D3Q19', spinup: s.spinup, average: s.average, dx: null,
      lighting: !!this.app.ed.lighting.fixtures.length, label: '' };
    this.est = null;
  }

  async estimate() {
    const app = this.app;
    if (app.dirty) await app.save();
    try {
      this.est = await busy('building the tank model for the estimate', () =>
        api.post(`/api/projects/${app.pid}/estimate`, { preset: this.opts.preset, lattice: this.opts.lattice, dx: this.opts.dx }));
      status('estimate ready', 'ok');
    } catch (e) { this.est = { error: e.message }; status(e.message, 'err'); }
    this.render();
  }

  async start() {
    const app = this.app;
    if (app.dirty) await app.save();
    const o = this.opts;
    try {
      const r = await api.post(`/api/projects/${app.pid}/runs`, { preset: o.preset, lattice: o.lattice, dx: o.dx,
        spinup: o.spinup, average: o.average, lighting: o.lighting, label: o.label || undefined });
      status(`run queued: ${r.run_id}`, 'ok');
      this.openLog = r.job.id;
      await this.refreshJobs();
    } catch (e) { status(e.message, 'err'); }
  }

  async cancel(id) {
    try { await api.post(`/api/jobs/${id}/cancel`); status('cancel requested', 'ok'); } catch (e) { status(e.message, 'err'); }
    this.refreshJobs();
  }

  async refreshJobs() {
    try { this.jobs = (await api.get('/api/jobs')).jobs; } catch (e) { return; }
    for (const j of this.jobs) if (!['done', 'failed', 'cancelled', 'interrupted'].includes(j.status)) this.follow(j.id);
    if (this.app.tab === 'run') this.renderJobs();
    this.app.updateJobBadge(this.jobs);
  }

  follow(id) {
    if (this.streams.has(id)) return;
    const lines = this.logs.get(id) || [];
    this.logs.set(id, lines);
    const es = new EventSource(`/api/jobs/${id}/events?since=${lines.length}`);
    this.streams.set(id, es);
    es.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      lines.push(...d.lines);
      const j = this.jobs.find(x => x.id === id);
      if (j) Object.assign(j, { status: d.status, pct: d.pct, phase: d.phase, error: d.error, info: d.info || j.info, last_line: lines[lines.length - 1] || j.last_line });
      if (this.app.tab === 'run') this.renderJobs();
      this.app.updateJobBadge(this.jobs);
    };
    es.addEventListener('end', () => {
      es.close(); this.streams.delete(id);
      const j = this.jobs.find(x => x.id === id);
      if (j?.status === 'done') status(`run ${j.run_id} finished${j.kind === 'run' ? ' and analysed' : ''}: open the Results tab`, 'ok');
      else if (j?.status === 'failed') status(`run ${j.run_id} failed: ${j.error}`, 'err');
      this.refreshJobs();
      this.app.results?.refreshRuns();
    });
    es.onerror = () => { es.close(); this.streams.delete(id); setTimeout(() => this.refreshJobs(), 3000); };
  }

  async showLog(id) {
    this.openLog = this.openLog === id ? null : id;
    if (this.openLog && !this.streams.has(id)) {
      try { const j = await api.get(`/api/jobs/${id}`); this.logs.set(id, j.lines); if (j.traceback) this.logs.get(id).push('', j.traceback); } catch (e) { /* ignore */ }
    }
    this.renderJobs();
  }

  // ---------------------------------------------------------------- panel
  render() {
    const app = this.app, P = app.panel, U = app.units;
    if (!this.opts) this.resetFor();
    const o = this.opts;
    clear(P);
    P.append(el('h2', {}, 'Run a simulation'));
    const v = app.validation;
    if (v && !v.ok) P.append(el('div', { class: 'errbox' }, 'Fix these before running:\n• ' + v.errors.join('\n• ')));
    if (app.designer.stale()) P.append(el('div', { class: 'warnbox' }, 'The rockscape has unbuilt changes: open the Rockscape tab and press "Build rockscape".'));
    P.append(el('h3', {}, 'Settings'));
    P.append(selectRow('Resolution preset', Object.entries(app.lib.presets).map(([k, dx]) => [k, `${k} (${U.fmt(dx, 'length', 2)} cells)`]), o.preset, (val) => { o.preset = val; this.est = null; this.render(); },
      { title: 'Preview for quick checks, Standard for design decisions, Fine for final confirmation' }));
    P.append(numRow('Custom cell size (optional)', U, null, () => o.dx ? o.dx * 1000 : null, (val) => { o.dx = val ? val / 1000 : null; this.est = null; }, { unit: 'mm', allowEmpty: true, placeholder: 'preset', step: 1, min: 1.5, max: 50 }));
    P.append(selectRow('Lattice', [['D3Q19', 'D3Q19 (default)'], ['D3Q27', 'D3Q27']], o.lattice, (val) => { o.lattice = val; this.est = null; }));
    P.append(numRow('Spin-up time', U, null, () => o.spinup, (val) => { o.spinup = val; this.est = null; }, { unit: 's', step: 10, min: 0, title: 'simulated time before averaging starts' }));
    P.append(numRow('Averaging time', U, null, () => o.average, (val) => { o.average = val; this.est = null; }, { unit: 's', step: 10, min: 0.1 }));
    const hasLights = app.ed.lighting.fixtures.length > 0;
    P.append(checkRow('Compute PAR (lighting) too', o.lighting && hasLights, (val) => { o.lighting = val; }, { disabled: !hasLights || !app.lib.lighting_available,
      title: hasLights ? 'After the flow run, trace the lights on the same grid and include light in the coral score (Phase 4)' : 'Add light fixtures first' }));
    P.append(textRow('Label (optional)', o.label, (val) => { o.label = val; }, { title: 'Short name for this run, e.g. "PH-left aimed down"' }));
    P.append(el('div', { class: 'btns' },
      el('button', { onclick: () => this.estimate() }, 'Estimate'),
      el('button', { class: 'primary', disabled: !(v?.ok) || app.designer.stale(), onclick: () => this.start() }, 'Start run')));
    P.append(el('div', { class: 'hint' }, 'Runs go to a queue and use the GPU one at a time. After the flow run the analysis (dead zones, water age, sand shear, coral scores, report) runs automatically.'));

    if (this.est) this._estimate(P);
    P.append(el('h3', {}, 'Jobs'));
    this.jobsBox = el('div');
    P.append(this.jobsBox);
    this.renderJobs();
    this.refreshJobs();
  }

  _estimate(P) {
    const U = this.app.units, e = this.est;
    P.append(el('h3', {}, 'Pre-run summary'));
    if (e.error) { P.append(el('div', { class: 'errbox' }, e.error)); return; }
    const s = e.estimate, i = e.info;
    if (e.warning) P.append(el('div', { class: 'warnbox' }, e.warning));
    P.append(el('div', { class: 'kpi' },
      el('div', {}, el('b', {}, `${num(s.cells / 1e6, 3)} M`), el('span', {}, 'cells')),
      el('div', {}, el('b', {}, fmtTime(s.runtime_s_est)), el('span', {}, `est. runtime (${s.backend})`)),
      el('div', {}, el('b', {}, `${num(s.gpu_memory_GB, 2)} GB`), el('span', {}, 'GPU memory'))));
    P.append(kvTable([
      ['Cell size', U.fmt(e.dx_m, 'length', 3)],
      ['Grid (nodes)', i.shape.join(' × ')],
      ['Water nodes', `${(i.fluid_nodes / 1e6).toFixed(3)} M`],
      ['Water in model', `${U.fmt(i.water_volume_model_m3, 'volume', 4)}`],
      ['Rock in model', `${num(i.rock_volume_m3 * 1000, 3)} L`],
      ['Time step', `${num(s.dt_s * 1000, 3)} ms`],
      ['Steps', `${s.steps.toLocaleString()} (spin-up ${fmtTime(e.spinup_s)} + averaging ${fmtTime(e.average_s)})`],
      ['Host memory', `${num(s.host_memory_GB, 2)} GB`],
      ['Free GPU memory now', s.free_vram_GB !== null ? `${num(s.free_vram_GB, 2)} GB` : 'n/a'],
      ['Throughput assumed', `${num(s.mlups_assumed, 4)} MLUPS`, 'measured on the RTX 3070 (D-119), derated 10 %'],
      ['Lattice speed of sound', `${num(i.lattice.sound_speed_ms, 3)} m/s`, 'artificial; real seawater ~1500 m/s'],
      ['Peak speed for dt', `${num(i.lattice.u_peak_est_ms, 3)} m/s`]]));
    if (i.pumps?.length) {
      const t = el('table', { class: 'grid' }, el('tr', {}, el('th', {}, 'Pump'), el('th', {}, 'Delivered'), el('th', {}, 'Outlet'), el('th', {}, 'Modelled as'), el('th', {}, 'U_eq')));
      for (const p of i.pumps) t.append(el('tr', { title: p.flow_basis }, el('td', {}, p.name), el('td', {}, U.fmt(p.delivered_flow_m3s, 'flow', 3)),
        el('td', {}, p.outlet), el('td', {}, `${p.outlet_model} (${p.source_nodes} nodes)`), el('td', {}, `${num(p.equivalent_velocity_ms, 3)} m/s`)));
      P.append(t);
    }
    if (e.notes?.length) {
      P.append(el('details', { open: true }, el('summary', {}, `Model notes (${e.notes.length})`),
        el('ul', { style: 'margin:4px 0;padding-left:18px;font-size:12px' }, e.notes.map(n => el('li', {}, n)))));
      P.append(el('div', { class: 'hint' }, 'Equivalent outlets keep each pump\'s momentum flux J; the far field of a jet depends on J only, the first ~5 outlet diameters are approximate.'));
    }
  }

  renderJobs() {
    const box = this.jobsBox;
    if (!box || !box.isConnected) return;
    clear(box);
    const mine = this.jobs.filter(j => j.project === this.app.pid);
    if (!mine.length) { box.append(el('div', { class: 'hint' }, 'No jobs since the server started. Finished runs are listed in the Results tab.')); }
    for (const j of mine) {
      const live = ['queued', 'running'].includes(j.status);
      const d = el('div', { class: 'job' + (j.status === 'running' ? ' running' : '') },
        el('div', { class: 'head' }, el('b', {}, j.label || j.run_id),
          el('span', { class: 'status-' + j.status }, `${j.status}${j.status === 'running' ? ' · ' + j.phase : ''}`)),
        el('div', { class: 'progress' }, el('div', { style: `width:${j.status === 'done' ? 100 : j.pct}%` })),
        live && j.info && j.info.t_sim_s !== undefined ? el('div', { class: 'last' },
          `simulated ${num(j.info.t_sim_s, 3)} / ${num(j.info.t_total_s, 3)} s (${j.info.phase || ''}) · max |u| ${num(j.info.u_max_ms, 3)} m/s · ETA ${fmtTime(j.info.eta_s)}`) : null,
        el('div', { class: 'last', title: j.last_line }, `${num(j.pct, 3)} %  ${j.last_line || ''}`),
        j.error ? el('div', { class: 'errbox' }, j.error) : null,
        el('div', { class: 'btns' },
          live ? el('button', { class: 'small danger', onclick: () => this.cancel(j.id) }, j.status === 'queued' ? 'Remove from queue' : 'Cancel') : null,
          el('button', { class: 'small', onclick: () => this.showLog(j.id) }, this.openLog === j.id ? 'Hide log' : 'Log'),
          j.status === 'done' ? el('button', { class: 'small primary', onclick: () => this.app.openResults(j.project, j.run_id) }, 'Open results') : null));
      if (this.openLog === j.id) {
        const pre = el('pre', { class: 'log' }, (this.logs.get(j.id) || []).join('\n') || '(no output yet)');
        d.append(pre);
        requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
      }
      box.append(d);
    }
  }
}
