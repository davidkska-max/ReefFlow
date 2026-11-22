"""Record a captioned walkthrough of ReefFlow: headless Chrome screencast -> timed JPEG frames -> MP4.

usage: video.py STEPS.json OUT.mp4 WORKSPACE
steps: {"eval": js} | {"wait": s} | {"note": text} (printed)
The page gets helpers (see HELPERS): __cap(text, step), __title(main, sub), __untitle(), __move(sel|{x,y}), __click(sel),
__orbit(seconds, degrees) - all return promises."""
import base64, http.client, json, os, shutil, socket, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pylib"))
import websocket  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W, H = 1920, 1080

HELPERS = r"""
(() => {
  const css = `
  #vcap { position: fixed; left: 50%; bottom: 46px; transform: translateX(-50%); z-index: 9999; display: flex; align-items: center; gap: 14px;
    background: rgba(6, 22, 32, .86); color: #fff; padding: 14px 26px 14px 16px; border-radius: 14px; font: 600 25px/1.3 "Segoe UI", system-ui;
    box-shadow: 0 14px 40px rgba(0,0,0,.35); border: 1px solid rgba(255,255,255,.12); backdrop-filter: blur(8px); max-width: 78vw;
    transition: opacity .35s, transform .35s; }
  #vcap.off { opacity: 0; transform: translateX(-50%) translateY(14px); }
  #vcap .st { flex: 0 0 auto; background: #19b6d2; color: #04202c; font-weight: 800; border-radius: 9px; padding: 3px 11px; font-size: 21px; }
  #vcap .st:empty { display: none; }
  #vcur { position: fixed; z-index: 10000; width: 26px; height: 26px; left: 0; top: 0; pointer-events: none;
    transition: transform .9s cubic-bezier(.4,.1,.2,1); transform: translate(960px, 540px); }
  #vcur svg { width: 26px; height: 26px; filter: drop-shadow(0 2px 3px rgba(0,0,0,.5)); }
  #vcur .ring { position: absolute; left: -11px; top: -11px; width: 22px; height: 22px; border-radius: 50%; border: 3px solid #19b6d2; opacity: 0; }
  #vcur .ring.go { animation: vring .55s ease-out; }
  @keyframes vring { from { opacity: .95; transform: scale(.4); } to { opacity: 0; transform: scale(2.2); } }
  #vtitle { position: fixed; inset: 0; z-index: 9998; display: grid; place-items: center; color: #e7f4f8; transition: opacity .6s;
    background: radial-gradient(120% 140% at 0% 0%, rgba(54,150,190,.6), rgba(54,150,190,0) 60%), linear-gradient(135deg, #0c2a3b, #081f2e); }
  #vtitle.off { opacity: 0; pointer-events: none; }
  #vtitle .b { text-align: center; }
  #vtitle svg { width: 120px; height: 120px; filter: drop-shadow(0 6px 18px rgba(0,0,0,.4)); }
  #vtitle h1 { font: 800 76px/1.1 "Segoe UI", system-ui; margin: 18px 0 8px; color: #fff; letter-spacing: -1px; }
  #vtitle h1 span { color: #6fd6ea; }
  #vtitle p { font: 500 30px/1.4 "Segoe UI", system-ui; color: #b9d5df; margin: 0; }`;
  const st = document.createElement('style'); st.textContent = css; document.head.append(st);
  const cap = document.createElement('div'); cap.id = 'vcap'; cap.className = 'off'; cap.innerHTML = '<span class="st"></span><span class="tx"></span>';
  const cur = document.createElement('div'); cur.id = 'vcur';
  cur.innerHTML = '<svg viewBox="0 0 24 24"><path d="M3 2l17 9.5-7.4 1.6L9.3 21z" fill="#fff" stroke="#0c2a3b" stroke-width="1.6" stroke-linejoin="round"/></svg><div class="ring"></div>';
  const logo = document.querySelector('#topbar .brand svg').outerHTML;
  const ttl = document.createElement('div'); ttl.id = 'vtitle'; ttl.className = 'off'; ttl.innerHTML = `<div class="b">${logo}<h1></h1><p></p></div>`;
  document.body.append(cap, cur, ttl);
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  let pos = [960, 540];
  window.__cap = async (text, step = '') => {
    if (!text) { cap.classList.add('off'); await sleep(350); return; }
    if (!cap.classList.contains('off')) { cap.classList.add('off'); await sleep(350); }
    cap.querySelector('.st').textContent = step; cap.querySelector('.tx').textContent = text;
    cap.classList.remove('off'); await sleep(400);
  };
  window.__title = async (main, sub) => { ttl.querySelector('h1').innerHTML = main; ttl.querySelector('p').textContent = sub || ''; ttl.classList.remove('off'); await sleep(700); };
  window.__untitle = async () => { ttl.classList.add('off'); await sleep(700); };
  const target = (t) => {
    if (typeof t === 'object' && !(t instanceof Element)) return [t.x, t.y];
    const e = typeof t === 'string' ? document.querySelector(t) : t;
    if (!e) throw new Error('no element ' + t);
    e.scrollIntoView({ block: 'nearest' });
    const r = e.getBoundingClientRect(); return [r.left + Math.min(r.width / 2, 60), r.top + r.height / 2];
  };
  window.__move = async (t) => { pos = target(t); cur.style.transform = `translate(${pos[0]}px, ${pos[1]}px)`; await sleep(950); };
  window.__click = async (t) => {
    await window.__move(t);
    const ring = cur.querySelector('.ring'); ring.classList.remove('go'); void ring.offsetWidth; ring.classList.add('go');
    await sleep(250);
    const e = typeof t === 'string' ? document.querySelector(t) : (t instanceof Element ? t : null);
    if (e) e.click();
    await sleep(350);
  };
  // pick an option in a <select>: point at it, then change the value
  window.__pick = async (sel, value) => {
    const e = typeof sel === 'string' ? document.querySelector(sel) : sel;
    await window.__click(e);
    e.value = value; e.dispatchEvent(new Event('change')); await sleep(500);
  };
  window.__orbit = async (seconds, degrees) => {
    const v = window.reefflow.view, c = v.camera, t = v.orbit.target;
    const r = Math.hypot(c.position.x - t.x, c.position.y - t.y), a0 = Math.atan2(c.position.y - t.y, c.position.x - t.x);
    const t0 = performance.now();
    await new Promise(res => { const f = (now) => { const k = Math.min((now - t0) / (seconds * 1000), 1); const e = 0.5 - 0.5 * Math.cos(Math.PI * k);
      const a = a0 + e * degrees * Math.PI / 180; c.position.x = t.x + r * Math.cos(a); c.position.y = t.y + r * Math.sin(a); c.lookAt(t); v.orbit.update(); v.render();
      if (k < 1) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  };
  window.__find = (selector, text) => [...document.querySelectorAll(selector)].find(e => e.textContent.trim().startsWith(text));
  return 'helpers';
})()
"""


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def main():
    steps = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]); ws_dir = sys.argv[3]
    frames_dir = Path(tempfile.mkdtemp(prefix="rf_frames_"))
    port = free_port()
    srv = subprocess.Popen([str(REPO / ".venv/Scripts/python.exe"), "-m", "reefflow.ui", "--workspace", ws_dir, "--port", str(port),
                            "--no-browser"], cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chrome = None
    frames = []
    try:
        for _ in range(200):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=1); c.request("GET", "/api/health"); c.getresponse().read(); break
            except Exception:
                time.sleep(0.2)
        dport = free_port()
        prof = tempfile.mkdtemp(prefix="rf_chrome_")
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={dport}", f"--user-data-dir={prof}",
                                   f"--window-size={W},{H}", "--hide-scrollbars", "--force-device-scale-factor=1",
                                   "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tgt = None
        for _ in range(100):
            try:
                c = http.client.HTTPConnection("127.0.0.1", dport, timeout=1); c.request("GET", "/json")
                tg = [t for t in json.loads(c.getresponse().read()) if t["type"] == "page"]
                if tg: tgt = tg[0]; break
            except Exception:
                time.sleep(0.2)
        ws = websocket.create_connection(tgt["webSocketDebuggerUrl"], timeout=600, max_size=None)
        mid = [0]

        def handle(m):
            if m.get("method") == "Page.screencastFrame":
                p = m["params"]
                f = frames_dir / f"f{len(frames):06d}.jpg"
                f.write_bytes(base64.b64decode(p["data"]))
                frames.append((p["metadata"]["timestamp"], f))
                mid[0] += 1
                ws.send(json.dumps({"id": mid[0], "method": "Page.screencastFrameAck", "params": {"sessionId": p["sessionId"]}}))
            elif m.get("method") == "Runtime.exceptionThrown":
                print("[page exc]", m["params"]["exceptionDetails"].get("exception", {}).get("description", "")[:300])

        def send(method, **params):
            mid[0] += 1; my = mid[0]
            ws.send(json.dumps({"id": my, "method": method, "params": params}))
            while True:
                m = json.loads(ws.recv())
                if m.get("id") == my:
                    return m
                handle(m)

        send("Runtime.enable"); send("Page.enable")
        send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        send("Page.navigate", url=f"http://127.0.0.1:{port}/")
        time.sleep(3)
        print(send("Runtime.evaluate", expression=HELPERS, returnByValue=True)["result"].get("result", {}).get("value"))
        send("Page.startScreencast", format="jpeg", quality=88, maxWidth=W, maxHeight=H, everyNthFrame=1)
        for st in steps:
            if "eval" in st:
                r = send("Runtime.evaluate", expression=st["eval"], awaitPromise=True, returnByValue=True)
                res = r.get("result", {})
                if "exceptionDetails" in res:
                    print("[eval EXC]", res["exceptionDetails"].get("exception", {}).get("description", res["exceptionDetails"])[:400])
                elif st.get("show"):
                    print("[eval]", json.dumps(res.get("result", {}).get("value"))[:400])
            elif "wait" in st:
                t_end = time.time() + st["wait"]
                while time.time() < t_end:
                    send("Runtime.evaluate", expression="1")
                    time.sleep(0.03)
            elif "note" in st:
                print("[note]", st["note"], f"{len(frames)} frames")
        send("Page.stopScreencast")
        ws.close()
    finally:
        if chrome:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(chrome.pid)], capture_output=True)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
    # frames -> concat list with real durations -> 30 fps H.264
    lst = frames_dir / "list.txt"
    with open(lst, "w", encoding="utf-8") as fh:
        for i, (t, f) in enumerate(frames):
            dur = (frames[i + 1][0] - t) if i + 1 < len(frames) else 1.5
            fh.write(f"file '{f.as_posix()}'\nduration {max(dur, 0.001):.4f}\n")
        fh.write(f"file '{frames[-1][1].as_posix()}'\n")
    total = frames[-1][0] - frames[0][0]
    print(f"{len(frames)} frames over {total:.1f} s")
    ff = shutil.which("ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf",
                    f"fps=30,scale={W}:{H}:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
    shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
