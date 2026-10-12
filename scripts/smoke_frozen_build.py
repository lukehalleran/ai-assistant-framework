#!/usr/bin/env python3
"""Build (optional) and smoke-test the frozen Daemon executable, one command.

    python scripts/smoke_frozen_build.py                 # smoke dist/Daemon/Daemon
    python scripts/smoke_frozen_build.py --build         # SPA + models + freeze, then smoke
    python scripts/smoke_frozen_build.py --dist PATH --work DIR

What it proves (the P3 procedure of 2026-10-10, made repeatable):

  Smoke A  cold first run: the legacy Gradio wizard answers, the user-data dir
           is created under the fake HOME, SIGINT exits 0 within 120 s and the
           shutdown phases run.
  Smoke B  returning user (synthetic pre-seed, fake API key): FastAPI answers,
           ``/health`` 200, the SPA root div is served, ``/api/session`` is 401
           without the launch token and 200 with it, a POST ``/api/chat`` stream
           ends in a ``complete`` event, SIGINT exits 0, shutdown phases run.
  Strict write assertion (per smoke): everything newer than a stamp file lies
           under ``home/.daemon`` (plus ``home/.cache``, ``home/.nv``,
           ``run.log``, the stamp itself). A frozen app that scatters stores
           into its launch cwd (the F4 class, BC-83) FAILS here.
  Isolation: ``~/.daemon`` (the owner's real data dir) is unchanged -- mtime and
           an md5 over its (path, mtime, size) listing; file CONTENTS are never
           read.

Safety (this script never touches the owner's Daemon): each smoke runs the
binary under ``env -i`` semantics (a from-scratch environment: HOME is a fresh
mkdtemp, no DISPLAY, no inherited PATH/PYTHONPATH/keys), on ports picked at
runtime that are never 8000 or 7860, and refuses to start if a target port is
already bound. Only the processes this script spawned are ever signalled. No
real API key is used, so no paid call is possible.

Exit status is non-zero when any step FAILs. ``report.md`` is written to the
work dir.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path; the
# build guard below imports utils.daemon_guard (same pattern as
# stage_frozen_models.py -- P3 smoke F3).
sys.path.insert(0, str(REPO_ROOT))
from utils.daemon_guard import daemon_running  # noqa: E402

FORBIDDEN_PORTS = frozenset({8000, 7860})
LOOPBACK = "127.0.0.1"
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

MIN_FREE_BYTES = 30 * 1024 ** 3
STARTUP_TIMEOUT_S = 180
EXIT_TIMEOUT_S = 120
CHAT_TIMEOUT_S = 150

# Everything the frozen app may legitimately write, relative to the smoke's
# temp dir. "" and "home" are the temp dir / fake HOME themselves (their mtime
# moves when a child is created).
ALLOWED_EXACT = frozenset({"", "home", "run.log", ".stamp"})
ALLOWED_PREFIXES = ("home/.daemon", "home/.cache", "home/.nv")

# Files that must be inside the bundle (data files PyInstaller cannot warn about).
REQUIRED_BUNDLE_FILES = ("rfc3987_syntax/syntax_rfc3987.lark",)

# Synthetic returning-user seed. Nothing here is real data.
SEED_PROFILE_JSON = (
    '{"user_id":"default","created_at":"2026-10-10T00:00:00",'
    '"updated_at":"2026-10-10T00:00:00",'
    '"identity":{"name":"Smoke Tester","pronouns":"they/them"},'
    '"quick_profile":{"name":"Smoke Tester"},"categories":{},"raw_log":[]}\n'
)
SEED_ENV = "OPENROUTER_API_KEY=sk-or-v1-fake-smoke\nDAEMON_MODE=user\n"


def out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without launching anything)
# ---------------------------------------------------------------------------

def port_in_use(port: int, host: str = LOOPBACK) -> bool:
    """True if something already accepts connections on (host, port)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def port_bindable(port: int, host: str = LOOPBACK) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def pick_free_ports(count: int, host: str = LOOPBACK) -> List[int]:
    """`count` distinct bindable ports, never 8000 or 7860."""
    ports: List[int] = []
    attempts = 0
    while len(ports) < count:
        attempts += 1
        if attempts > 200:
            raise RuntimeError("could not find free ports")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
        if port in FORBIDDEN_PORTS or port in ports:
            continue
        if port_in_use(port, host):
            continue
        ports.append(port)
    return ports


def assert_ports_safe(ports: Iterable[int], host: str = LOOPBACK) -> None:
    """Refuse forbidden or already-occupied ports (raises SystemExit)."""
    for port in ports:
        if port in FORBIDDEN_PORTS:
            raise SystemExit(f"refusing to use forbidden port {port} (8000/7860 belong to the live Daemon)")
        if port_in_use(port, host):
            raise SystemExit(f"refusing to run: port {port} is already in use")


def build_env(home: Path, api_port: int, gradio_port: int) -> Dict[str, str]:
    """A from-scratch environment (env -i): no DISPLAY, HOME is the temp home,
    nothing inherited from the caller (keys, PYTHONPATH, DAEMON_* ...)."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "DAEMON_API_HOST": LOOPBACK,
        "DAEMON_API_PORT": str(api_port),
        "GRADIO_SERVER_NAME": LOOPBACK,
        # launch_gui reads GRADIO_PORT (default 7860); GRADIO_SERVER_PORT is
        # set too so neither name can fall back to the live Daemon's port.
        "GRADIO_SERVER_PORT": str(gradio_port),
        "GRADIO_PORT": str(gradio_port),
    }


def seed_returning_user(home: Path) -> List[Path]:
    """Write the synthetic pre-seed that makes check_first_run() False
    (identity.name set) and gives the app a fake key. Returns the files made."""
    data_dir = home / ".daemon"
    data_dir.mkdir(parents=True, exist_ok=True)
    profile = data_dir / "user_profile.json"
    env_file = data_dir / ".env"
    profile.write_text(SEED_PROFILE_JSON, encoding="utf-8")
    env_file.write_text(SEED_ENV, encoding="utf-8")
    return [profile, env_file]


def is_allowed_write(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if rel == ".":
        rel = ""
    if ".." in rel.split("/"):
        return False
    if rel in ALLOWED_EXACT:
        return True
    return any(rel == p or rel.startswith(p + "/") for p in ALLOWED_PREFIXES)


def strict_write_violations(rel_paths: Iterable[str]) -> List[str]:
    """Paths (relative to the smoke temp dir) that are not allowed writes."""
    return sorted(p for p in rel_paths if not is_allowed_write(p))


def find_newer(root: Path, stamp: Path) -> List[str]:
    """`find root -newer stamp` as relative paths ('' is root itself)."""
    ref = stamp.stat().st_mtime_ns
    found: List[str] = []
    if root.stat().st_mtime_ns > ref:
        found.append("")
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            try:
                if p.lstat().st_mtime_ns > ref:
                    found.append(str(p.relative_to(root)))
            except OSError:  # degrades: file vanished mid-walk; cannot be a stray write we can name
                continue
    return sorted(found)


def snapshot_dir(path: Path) -> Dict[str, object]:
    """mtime + md5 over the sorted (path, mtime_ns, size) listing. File contents
    are never read."""
    if not path.exists():
        return {"exists": False, "mtime": None, "digest": None, "files": 0}
    h = hashlib.md5()
    count = 0
    entries: List[str] = []
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:  # degrades: file vanished mid-walk; the next snapshot would differ anyway
                continue
            entries.append(f"{p} {st.st_mtime_ns} {st.st_size}")
            count += 1
    for line in sorted(entries):
        h.update(line.encode("utf-8", "replace"))
        h.update(b"\n")
    return {"exists": True, "mtime": path.stat().st_mtime_ns, "digest": h.hexdigest(), "files": count}


def shutdown_phases(log_text: str) -> Tuple[int, int]:
    """(highest phase reached, declared total) from '[Shutdown] n/N' lines."""
    reached, total = 0, 0
    for m in re.finditer(r"\[Shutdown\] (\d+)/(\d+)", log_text):
        reached = max(reached, int(m.group(1)))
        total = max(total, int(m.group(2)))
    return reached, total


def extract_launch_token(index_html: str) -> Optional[str]:
    m = re.search(r'name="daemon-launch-token"\s+content="([^"]+)"', index_html)
    return m.group(1) if m else None


Result = Tuple[str, str, str]  # (step, status, detail)


def exit_code(results: Iterable[Result]) -> int:
    return 1 if any(status == FAIL for _s, status, _d in results) else 0


def render_table(results: Iterable[Result]) -> str:
    rows = list(results)
    w1 = max([len("Step")] + [len(r[0]) for r in rows])
    lines = [f"{'Step'.ljust(w1)}  Status  Detail", f"{'-' * w1}  ------  ------"]
    for step, status, detail in rows:
        lines.append(f"{step.ljust(w1)}  {status.ljust(6)}  {detail}")
    verdict = "FAIL" if exit_code(rows) else "PASS"
    lines.append("")
    lines.append(f"OVERALL: {verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP (stdlib)
# ---------------------------------------------------------------------------

def http_request(port: int, path: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                 body: Optional[bytes] = None, timeout: float = 10.0) -> Tuple[int, str]:
    conn = http.client.HTTPConnection(LOOPBACK, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", "replace")
    finally:
        conn.close()


def http_status(port: int, path: str, **kw) -> int:
    try:
        return http_request(port, path, **kw)[0]
    except (OSError, http.client.HTTPException):  # degrades: not up yet / refused -> status 0
        return 0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def run_logged(cmd: List[str], log: Path, cwd: Path, env: Optional[Dict[str, str]] = None) -> int:
    with open(log, "ab") as fh:
        fh.write(("$ " + " ".join(cmd) + "\n").encode())
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def freeze_command(repo: Path) -> List[str]:
    """The capped, niced PyInstaller invocation (the owner is using the box)."""
    py = [sys.executable, "-s"]
    inner = ["env", "-u", "PYTHONPATH", "HF_HUB_OFFLINE=1", *py,
             "-m", "PyInstaller", "daemon.spec", "--clean", "--noconfirm"]
    if shutil.which("systemd-run"):
        return ["nice", "-n", "10", "systemd-run", "--user", "--scope", "-q",
                "-p", "MemoryMax=6G", "-p", "MemorySwapMax=1G",
                f"--working-directory={repo}", *inner]
    return ["nice", "-n", "10", *inner]


def do_build(repo: Path, work: Path) -> List[Result]:
    results: List[Result] = []
    # Same rule as stage_frozen_models.py --apply: a 6 GB freeze must not
    # compete with a Daemon running from THIS checkout (cwd-keyed guard, so a
    # build in a clone beside the live Daemon is allowed).
    if daemon_running():
        return [("build: refuse while Daemon runs from this checkout", FAIL,
                 "stop it, or build from a clone")]
    log = work / "build.log"
    # Inherit the caller's environment (npm needs PATH/HOME) minus PYTHONPATH,
    # which in a dev shell can point imports at a different checkout.
    web_env = dict(os.environ)
    web_env.pop("PYTHONPATH", None)
    py = [sys.executable, "-s"]
    steps = [
        ("build: SPA (npm ci + build)", ["bash", "-c", "npm ci --prefer-offline --no-audit --no-fund && npm run build"],
         repo / "web", web_env),
        ("build: stage models (dry run)", [*py, "scripts/stage_frozen_models.py"], repo, web_env),
        ("build: stage models (--apply)", [*py, "scripts/stage_frozen_models.py", "--apply"], repo, web_env),
        ("build: freeze (PyInstaller)", freeze_command(repo), repo, web_env),
    ]
    for name, cmd, cwd, env in steps:
        t0 = time.time()
        out(f"[build] {name} ...")
        rc = run_logged(cmd, log, cwd, env)
        dt = time.time() - t0
        results.append((name, PASS if rc == 0 else FAIL, f"exit {rc}, {dt:.0f}s (log: {log})"))
        if rc != 0:
            break
    return results


def check_bundle(dist: Path) -> List[Result]:
    internal = dist.parent / "_internal"
    missing = [rel for rel in REQUIRED_BUNDLE_FILES if not (internal / rel).is_file()]
    return [("bundle: required data files", FAIL if missing else PASS,
             ("missing " + ", ".join(missing)) if missing else f"{len(REQUIRED_BUNDLE_FILES)} present")]


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class Smoke:
    """One smoke run in its own temp dir. Owns exactly one child process."""

    def __init__(self, tag: str, dist: Path, work: Path, returning: bool, children: List[subprocess.Popen]):
        self.tag = tag
        self.dist = dist
        self.returning = returning
        self.children = children
        self.tmp = Path(tempfile.mkdtemp(prefix=f"smoke{tag}.", dir=work))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.stamp = self.tmp / ".stamp"
        self.results: List[Result] = []
        self.proc: Optional[subprocess.Popen] = None
        self.api_port = 0
        self.gradio_port = 0

    def rec(self, step: str, ok: bool, detail: str = "") -> bool:
        self.results.append((f"{self.tag}: {step}", PASS if ok else FAIL, detail))
        return ok

    def log_text(self) -> str:
        try:
            return (self.tmp / "run.log").read_text(encoding="utf-8", errors="replace")
        except OSError:  # degrades: no log yet -> empty text, the checks that need it FAIL
            return ""

    def launch(self) -> None:
        self.api_port, self.gradio_port = pick_free_ports(2)
        assert_ports_safe([self.api_port, self.gradio_port])
        if self.returning:
            seed_returning_user(self.home)
        self.stamp.touch()
        time.sleep(1.1)  # stamp must be strictly older than anything the app writes
        env = build_env(self.home, self.api_port, self.gradio_port)
        with open(self.tmp / "run.log", "wb") as log:
            self.proc = subprocess.Popen([str(self.dist)], cwd=self.tmp, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(self.proc)
        self.t_launch = time.time()

    def wait_up(self, ports: Tuple[int, ...]) -> Optional[Tuple[int, float]]:
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return None
            for port in ports:
                if http_status(port, "/", timeout=2.0) != 0:
                    return port, time.time() - self.t_launch
            time.sleep(1.0)
        return None

    def stop(self) -> None:
        """SIGINT our own child, expect a clean exit within EXIT_TIMEOUT_S."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            code = None if proc is None else proc.returncode
            self.rec("clean exit", False, f"process already gone (exit {code}) before SIGINT")
            return
        t0 = time.time()
        proc.send_signal(signal.SIGINT)
        try:
            code = proc.wait(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()  # our own child only
            proc.wait()
            self.rec("clean exit", False, f"still running {EXIT_TIMEOUT_S}s after SIGINT; killed")
            return
        self.rec("clean exit", code == 0, f"exit {code} in {time.time() - t0:.0f}s")
        reached, total = shutdown_phases(self.log_text())
        self.rec("shutdown phases", total > 0 and reached == total, f"{reached}/{total} reached")

    def strict_write(self) -> None:
        bad = strict_write_violations(find_newer(self.tmp, self.stamp))
        self.rec("strict write assertion", not bad,
                 "nothing outside home/.daemon" if not bad else "stray: " + ", ".join(bad[:8]))

    def run(self) -> List[Result]:
        try:
            self.launch()
            if self.returning:
                self._check_returning()
            else:
                self._check_cold()
        except SystemExit as exc:
            self.rec("launch safety", False, str(exc))
            return self.results
        except Exception as exc:  # degrades: surfaced as a FAIL row; the finally in main() still reaps the child
            self.rec("smoke harness", False, f"{type(exc).__name__}: {exc}")
        self.stop()
        self.strict_write()
        tail = self.log_text()
        self.rec("no startup failure in log", "Startup failed" not in tail and "FATAL" not in tail)
        return self.results

    # -- Smoke A -----------------------------------------------------------
    def _check_cold(self) -> None:
        up = self.wait_up((self.gradio_port, self.api_port))
        if up is None:
            self.rec("wizard answers", False, "no server within %ds or process died; see %s" % (STARTUP_TIMEOUT_S, self.tmp / "run.log"))
            return
        port, dt = up
        self.rec("wizard answers", port == self.gradio_port and http_status(port, "/") == 200,
                 f"Gradio / -> 200 on :{port} after {dt:.0f}s" if port == self.gradio_port
                 else f"answered on :{port}, expected the wizard on :{self.gradio_port}")
        data_dir = self.home / ".daemon"
        made = [n for n in ("corpus_v4.json", "user_profile.json") if (data_dir / n).exists()]
        self.rec("user-data dir created", len(made) == 2, f"{data_dir.name}: {', '.join(made) or 'missing'}")

    # -- Smoke B -----------------------------------------------------------
    def _check_returning(self) -> None:
        up = self.wait_up((self.api_port, self.gradio_port))
        if up is None:
            self.rec("API answers", False, "no server within %ds or process died; see %s" % (STARTUP_TIMEOUT_S, self.tmp / "run.log"))
            return
        port, dt = up
        if not self.rec("API answers", port == self.api_port, f"/ on :{port} after {dt:.0f}s"
                        if port == self.api_port else f"wizard appeared on :{port} (seed not recognised as returning user)"):
            return
        self.rec("/health 200", http_status(port, "/health") == 200)
        status, index = http_request(port, "/")
        self.rec("SPA root div served", status == 200 and '<div id="root">' in index, f"/ -> {status}")
        token = extract_launch_token(index)
        self.rec("launch token in index", token is not None)
        self.rec("/api/session 401 without token", http_status(port, "/api/session") == 401)
        if token:
            hdr = {"X-Daemon-Launch-Token": token}
            self.rec("/api/session 200 with token", http_status(port, "/api/session", headers=hdr) == 200)
            self._chat(port, token)
        self.rec("/admin hidden when packaged", http_status(port, "/admin/") == 404, "404 by design (admin_packaged)")

    def _chat(self, port: int, token: str) -> None:
        hdr = {"X-Daemon-Launch-Token": token, "Origin": f"http://{LOOPBACK}:{port}",
               "Content-Type": "application/json"}
        body = b'{"text":"[test] hello, this is a smoke test [/test]"}'
        try:
            status, text = http_request(port, "/api/chat", method="POST", headers=hdr, body=body, timeout=CHAT_TIMEOUT_S)
        except (OSError, http.client.HTTPException) as exc:
            self.rec("POST /api/chat completes", False, f"{type(exc).__name__}: {exc}")
            return
        events = re.findall(r"^event: (\w+)", text, flags=re.M)
        ok = status == 200 and bool(events) and events[-1] == "complete"
        self.rec("POST /api/chat completes", ok,
                 f"{len(events)} events, last={events[-1] if events else None} (fake key: an auth-error reply is expected)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def kill_children(children: List[subprocess.Popen]) -> None:
    """Reap ONLY processes this script spawned."""
    for proc in children:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError):  # degrades: escalate to kill on our own child
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):  # degrades: already gone
                    pass


def write_report(work: Path, results: List[Result], extra: List[str]) -> Path:
    path = work / "report.md"
    table = render_table(results)
    lines = [f"# Frozen build smoke report ({datetime.now().isoformat(timespec='seconds')})", "",
             "```", table, "```", ""]
    lines += extra
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", default=str(REPO_ROOT / "dist" / "Daemon" / "Daemon"),
                    help="frozen executable (default: <repo>/dist/Daemon/Daemon)")
    ap.add_argument("--build", action="store_true", help="run SPA build, model staging and the freeze first")
    ap.add_argument("--work", default=None,
                    help="work dir (default: ~/daemon_exec/smoke_runs/<timestamp>)")
    ap.add_argument("--guard-dir", default=str(Path.home() / ".daemon"),
                    help="the owner's real data dir whose immutability is verified (default: ~/.daemon)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(args.work) if args.work else Path.home() / "daemon_exec" / "smoke_runs" / stamp
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < MIN_FREE_BYTES:
        raise SystemExit("refusing to run: less than 30 GB free")
    guard_dir = Path(args.guard_dir)
    dist = Path(args.dist).resolve()

    results: List[Result] = []
    extra: List[str] = []
    children: List[subprocess.Popen] = []
    before = snapshot_dir(guard_dir)
    try:
        if args.build:
            results += do_build(REPO_ROOT, work)
            if exit_code(results):
                return _finish(work, results, extra)
        if not dist.is_file():
            results.append(("preflight: binary exists", FAIL, f"{dist} not found (build first, or pass --dist)"))
            return _finish(work, results, extra)
        results.append(("preflight: binary exists", PASS, str(dist)))
        results += check_bundle(dist)
        for tag, returning in (("A", False), ("B", True)):
            smoke = Smoke(tag, dist, work, returning, children)
            out(f"[smoke {tag}] {'returning user' if returning else 'cold first run'} in {smoke.tmp}")
            results += smoke.run()
            extra.append(f"- smoke {tag}: dir `{smoke.tmp}`, api :{smoke.api_port}, gradio :{smoke.gradio_port}")
    finally:
        kill_children(children)
    after = snapshot_dir(guard_dir)
    same = before == after
    results.append(("isolation: ~/.daemon unchanged", PASS if same else FAIL,
                    f"{before['files']} files, mtime+md5 {'identical' if same else 'CHANGED'}"
                    if before["exists"] else "directory absent before and after" if not after["exists"]
                    else "directory CREATED during the run"))
    left = [p.pid for p in children if p.poll() is None]
    results.append(("isolation: no processes left", PASS if not left else FAIL, f"{len(left)} still running"))
    return _finish(work, results, extra)


def _finish(work: Path, results: List[Result], extra: List[str]) -> int:
    out("")
    out(render_table(results))
    report = write_report(work, results, extra)
    out(f"report: {report}")
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
