"""Background simulation jobs: one GPU job at a time, the rest queued.

Each job runs in its own *process* (multiprocessing 'spawn'): Taichi/CUDA state is
isolated from the web server, a crash cannot take the server down, and cancel is a
clean process termination (Taichi kernels cannot be interrupted from Python). The
worker streams human-readable progress lines back through a queue; the manager
derives a percentage from them and keeps the log in memory (for the browser) and
in the run directory (``ui_log.txt``, ``ui_job.json``).

Job kinds
    run       build the tank model, run the CFD (reefflow.run.run), optionally trace
              lighting (Phase 4) on the CFD grid, then analyze (reefflow.analysis.analyze)
    analyze   (re-)analyze an existing run directory
    carve     photo reconstruction: visual-hull carving of a photo session (reefflow.ui.photos)

Cancel is cooperative first: the worker gets a stop event that ``run.run`` polls through
``should_stop`` (clean ``RunCancelled``); if the process has not exited after a grace
period (e.g. while Taichi compiles or the analysis runs) it is terminated.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import queue as _queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

TERMINAL = ("done", "failed", "cancelled", "interrupted")
_PCT = re.compile(r"\(\s*([0-9.]+)\s*%\)")
_ANALYZE_STEPS = ["metrics", "water age", "dead zones", "wall shear", "coral scores", "figures", "done"]


@dataclass
class Job:
    id: str
    kind: str
    project: str
    run_id: str
    run_dir: Path
    spec: dict
    label: str = ""
    status: str = "queued"
    phase: str = "queued"
    pct: float = 0.0
    lines: list = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False
    status_info: dict = field(default_factory=dict)

    def public(self, with_lines: bool = False) -> dict:
        d = {"id": self.id, "kind": self.kind, "project": self.project, "run_id": self.run_id, "label": self.label,
             "status": self.status, "phase": self.phase, "pct": round(self.pct, 1), "error": self.error,
             "created": self.created, "started": self.started, "finished": self.finished,
             "n_lines": len(self.lines), "last_line": self.lines[-1] if self.lines else "",
             "preset": self.spec.get("preset"), "lattice": self.spec.get("lattice"), "dx_m": self.spec.get("dx"),
             "info": self.status_info}
        if with_lines:
            d["lines"] = list(self.lines)
            d["traceback"] = self.traceback
        return d


def _phase_range(spec: dict) -> dict:
    light = bool(spec.get("lighting"))
    if spec.get("kind") == "carve":
        return {"load": (0, 5), "carve": (5, 85), "finish": (85, 100)}
    if spec.get("kind") == "analyze":
        return {"build": (0, 0), "run": (0, 0), "light": (0, 0), "analyze": (0, 100)}
    return {"build": (0, 3), "run": (3, 86 if light else 92), "light": (86, 92) if light else (92, 92),
            "analyze": (92, 100) if spec.get("analyze", True) else (100, 100)}


class JobManager:
    def __init__(self, start_worker: bool = True, prefix: str = "j", cancel_grace_s: float = 20.0):
        self.prefix = prefix
        self.cancel_grace_s = cancel_grace_s
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self._ids = itertools.count(1)
        self._proc = None
        self._stop = False
        self._thread = None
        if start_worker:
            self._thread = threading.Thread(target=self._loop, name="reefflow-jobs", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ API
    def submit(self, kind: str, project: str, run_id: str, run_dir: Path, spec: dict, label: str = "") -> Job:
        with self.cv:
            jid = f"{self.prefix}{next(self._ids)}-{int(time.time()) % 100000}"
            spec = {**spec, "kind": kind, "run_dir": str(run_dir)}
            job = Job(jid, kind, project, run_id, Path(run_dir), spec, label=label or run_id)
            self.jobs[jid] = job
            self.order.append(jid)
            self._persist(job)
            self.cv.notify_all()
            return job

    def get(self, jid: str) -> Job:
        with self.lock:
            if jid not in self.jobs:
                raise KeyError(f"no job {jid!r}")
            return self.jobs[jid]

    def list(self) -> list:
        with self.lock:
            return [self.jobs[j].public() for j in reversed(self.order)]

    def active_for_run(self, project: str, run_id: str) -> Job | None:
        with self.lock:
            for j in self.jobs.values():
                if j.project == project and j.run_id == run_id and j.status not in TERMINAL:
                    return j
        return None

    def cancel(self, jid: str) -> Job:
        with self.cv:
            job = self.get(jid)
            if job.status == "queued":
                job.status = job.phase = "cancelled"
                job.finished = time.time()
                job.lines.append("[ui] cancelled before it started")
                self._persist(job)
            elif job.status == "running":
                job.cancel_requested = True
                job.lines.append("[ui] cancel requested: stopping the worker process")
            self.cv.notify_all()
            return job

    def shutdown(self):
        with self.cv:
            self._stop = True
            for j in self.jobs.values():
                if j.status == "running":
                    j.cancel_requested = True
            self.cv.notify_all()
        p = self._proc
        if p is not None and p.is_alive():
            p.terminate()
            p.join(5)

    def run_now(self, job: Job):
        """Run one job synchronously in this thread (tests)."""
        self._execute(job)

    # ------------------------------------------------------------------ worker loop
    def _next(self) -> Job | None:
        for jid in self.order:
            j = self.jobs[jid]
            if j.status == "queued":
                return j
        return None

    def _loop(self):
        while True:
            with self.cv:
                while not self._stop and self._next() is None:
                    self.cv.wait(1.0)
                if self._stop:
                    return
                job = self._next()
            try:
                self._execute(job)
            except Exception as e:                        # noqa: BLE001 (manager must survive anything)
                with self.lock:
                    job.status = "failed"
                    job.error = f"job manager error: {e}"
                    job.traceback = traceback.format_exc()
                    job.finished = time.time()
                    self._persist(job)

    def _execute(self, job: Job):
        with self.lock:
            if job.status != "queued":
                return
            job.status, job.phase, job.started = "running", "starting", time.time()
            self._persist(job)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        stop = ctx.Event()
        p = ctx.Process(target=worker_main, args=(job.spec, q, stop), name=f"reefflow-{job.id}", daemon=True)
        p.start()
        self._proc = p
        ranges = _phase_range(job.spec)
        got_end = False
        cancel_t0 = time.monotonic()
        log = job.run_dir / "ui_log.txt"
        with open(log, "a", encoding="utf-8") as fh:
            while True:
                if job.cancel_requested and p.is_alive():
                    if not stop.is_set():
                        stop.set()                      # cooperative: run.run raises RunCancelled
                        cancel_t0 = time.monotonic()
                    elif time.monotonic() - cancel_t0 > self.cancel_grace_s:
                        p.terminate()                   # not responding (compiling / analysing): kill
                        p.join(10)
                        with self.lock:
                            job.status = job.phase = "cancelled"
                            job.lines.append("[ui] cancelled by the user (worker terminated)")
                        break
                try:
                    msg = q.get(timeout=0.25)
                except _queue.Empty:
                    if not p.is_alive():
                        # drain anything left, then stop
                        try:
                            while True:
                                self._handle(job, q.get_nowait(), ranges, fh)
                        except _queue.Empty:
                            pass
                        if not got_end and job.status == "running":
                            with self.lock:
                                job.status = "failed"
                                job.error = job.error or f"worker process exited unexpectedly (code {p.exitcode})"
                        break
                    continue
                got_end = self._handle(job, msg, ranges, fh) or got_end
                if got_end:
                    break
        p.join(30)
        if p.is_alive():
            p.terminate()
        self._proc = None
        with self.lock:
            job.finished = time.time()
            if job.status == "running":
                job.status = "done"
            if job.status == "done":
                job.pct, job.phase = 100.0, "done"
            self._persist(job)

    def _handle(self, job: Job, msg, ranges, fh) -> bool:
        kind = msg[0]
        with self.lock:
            if kind == "line":
                line = str(msg[1])
                job.lines.append(line)
                if len(job.lines) > 5000:
                    del job.lines[:1000]
                fh.write(line + "\n")
                fh.flush()
                lo, hi = ranges.get(job.phase, (job.pct, job.pct))
                if job.phase == "run":
                    m = _PCT.search(line)
                    if m:
                        job.pct = lo + (hi - lo) * float(m.group(1)) / 100.0
                elif job.phase == "analyze" and line.startswith("[analyze]"):
                    for i, s in enumerate(_ANALYZE_STEPS):
                        if s in line:
                            job.pct = max(job.pct, lo + (hi - lo) * i / len(_ANALYZE_STEPS))
            elif kind == "status":
                st = msg[1]
                lo, hi = ranges.get(job.phase, (job.pct, job.pct))
                if st.get("fraction") is not None:
                    job.pct = max(job.pct, lo + (hi - lo) * float(st["fraction"]))
                job.status_info = st
            elif kind == "cancelled":
                job.status = job.phase = "cancelled"
                job.lines.append(f"[ui] {msg[1]}")
                fh.write(f"[ui] {msg[1]}\n")
                job.finished = time.time()
                self._persist(job)
                return True
            elif kind == "phase":
                job.phase = msg[1]
                job.pct = max(job.pct, ranges.get(msg[1], (job.pct, job.pct))[0])
                self._persist(job)
            elif kind == "done":
                job.status = "done"
                job.pct, job.phase, job.finished = 100.0, "done", time.time()
                self._persist(job)
                return True
            elif kind == "error":
                job.status = "failed"
                job.error = str(msg[1])
                job.traceback = msg[2] if len(msg) > 2 else None
                job.lines.append(f"[error] {msg[1]}")
                fh.write(f"[error] {msg[1]}\n{job.traceback or ''}\n")
                job.finished = time.time()
                self._persist(job)
                return True
        return False

    def _persist(self, job: Job):
        try:
            rec = {"job_id": job.id, "kind": job.kind, "label": job.label, "status": job.status, "created": job.created,
                   "started": job.started, "finished": job.finished, "error": job.error,
                   "preset": job.spec.get("preset"), "lattice": job.spec.get("lattice"), "dx_m": job.spec.get("dx"),
                   "spinup_s": job.spec.get("spinup"), "average_s": job.spec.get("average"),
                   "lighting": job.spec.get("lighting")}
            old = job.run_dir / "ui_job.json"
            if old.is_file() and job.kind == "analyze":
                prev = json.loads(old.read_text(encoding="utf-8"))
                rec = {**prev, "status": "analysed" if job.status == "done" else prev.get("status"),
                       "analyze_status": job.status, "analyze_error": job.error,
                       "error": job.error if job.status == "failed" else prev.get("error")}
            elif job.status == "done" and (job.run_dir / "analysis.json").is_file():
                rec["status"] = "analysed"
            elif job.status == "done":
                rec["status"] = "finished"
            (job.run_dir / "ui_job.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        except Exception:
            pass


def mark_interrupted(run_dirs):
    """At server start: runs whose record says queued/running were cut off by a restart."""
    for d in run_dirs:
        f = Path(d) / "ui_job.json"
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("status") in ("queued", "running"):
            rec["status"] = "interrupted"
            rec["error"] = "the server stopped while this run was queued or running"
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------
# worker process
# ----------------------------------------------------------------------
def worker_main(spec: dict, q, stop=None):
    """Entry point of the job process. Never raises: errors go back as ('error', msg, tb)."""
    def say(s):
        q.put(("line", str(s)))

    last = [0.0]

    def on_status(st):                                 # throttled structured progress
        now = time.monotonic()
        if now - last[0] > 0.5 or st.get("fraction", 0) >= 1:
            last[0] = now
            q.put(("status", {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in st.items()}))

    def should_stop():
        return stop is not None and stop.is_set()

    try:
        import reefflow  # noqa: F401  (disables Taichi's online version check before Taichi loads)
        from reefflow import backend
        from reefflow import project as PR
        from reefflow import tank as TK
        from reefflow.analysis.analyze import analyze
        from . import lighting_adapter as LA
        run_dir = Path(spec["run_dir"])
        if spec.get("kind") == "carve":
            from .photos import carve_job
            ok = carve_job(Path(spec["session_dir"]), progress=say, phase=lambda p: q.put(("phase", p)),
                           status=on_status, should_stop=should_stop)
            q.put(("done", {}) if ok else ("cancelled", "carving cancelled"))
            return
        if spec.get("kind") == "analyze":
            q.put(("phase", "analyze"))
            par = LA.load_par(run_dir)
            analyze(run_dir, par=par["meter"] if par else None, progress=say)
            q.put(("done", {}))
            return
        q.put(("phase", "build"))
        P = PR.load(spec["project_file"])             # project errors surface before the GPU is touched
        arch = backend.init(spec.get("arch") or "auto")
        say(f"[ui] compute backend: {arch}")
        over = {"preset": spec.get("preset"), "lattice": spec.get("lattice")}
        over["dx"] = f"{float(spec['dx']) * 1000:.6g} mm" if spec.get("dx") else None
        if spec.get("spinup") is not None:
            over["spinup"] = f"{float(spec['spinup']):.6g} s"
        if spec.get("average") is not None:
            over["average"] = f"{float(spec['average']):.6g} s"
        P = P.with_overrides(**over)
        say(f"[ui] building the tank model at dx = {P.dx() * 1e3:g} mm")
        model = TK.build(P, P.dx())
        for n in model.notes:
            say(f"[model] {n}")
        q.put(("phase", "run"))
        from reefflow.run import RunCancelled, run
        try:
            run(model, run_dir, progress=say, on_status=on_status, should_stop=should_stop)
        except RunCancelled as e:
            q.put(("cancelled", str(e)))
            return
        par = None
        if spec.get("lighting"):
            q.put(("phase", "light"))
            try:
                par = LA.compute_par(P, model, run_dir, progress=say, rays=spec.get("rays"))
            except Exception as e:                        # noqa: BLE001 (flow results stay valid)
                say(f"[light] lighting failed, continuing with flow only: {type(e).__name__}: {e}")
                par = None
        if spec.get("analyze", True):
            q.put(("phase", "analyze"))
            del model
            analyze(run_dir, par=par, progress=say)
        q.put(("done", {}))
    except BaseException as e:                            # noqa: BLE001
        q.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))
