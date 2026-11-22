"""Scratch workspace: create the UI demo project, run a standard simulation with lighting, report."""
import http.client
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WS = Path(sys.argv[1])
WS.mkdir(parents=True, exist_ok=True)
s = socket.socket(); s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]; s.close()
log = open(WS / "server.log", "w")
srv = subprocess.Popen([str(REPO / ".venv/Scripts/python.exe"), "-m", "reefflow.ui", "--workspace", str(WS), "--port", str(PORT),
                        "--no-browser"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)


def req(method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    if r.status >= 400:
        raise RuntimeError(f"{method} {path}: {r.status} {data[:400]!r}")
    return json.loads(data) if data else None


try:
    for _ in range(300):
        try:
            req("GET", "/api/health"); break
        except Exception:
            time.sleep(0.2)
    pid = req("POST", "/api/projects", {"name": "Demo 75-gallon reef", "template": "demo"})["id"]
    print("project", pid, flush=True)
    job = req("POST", f"/api/projects/{pid}/runs", {"preset": "standard", "lattice": "D3Q19", "lighting": True, "label": "demo"})
    jid = job.get("id") or job.get("job", {}).get("id")
    print("job", jid, flush=True)
    t0 = time.time()
    while True:
        j = req("GET", f"/api/jobs/{jid}")
        j = j.get("job", j)
        st = j.get("status")
        if st in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(20)
        print(f"{(time.time() - t0) / 60:5.1f} min  {st}  {j.get('pct', 0):.0f} %  {str(j.get('last_line', ''))[:100]}", flush=True)
    print("final", st, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    runs = req("GET", f"/api/projects/{pid}/runs")["runs"]
    rid = runs[0].get("id") or runs[0].get("rid")
    summ = req("GET", f"/api/runs/{pid}/{rid}")
    print(json.dumps({"run": rid, "status": runs[0].get("status"), "key_metrics": summ.get("key_metrics")}, indent=1)[:3000], flush=True)
finally:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
