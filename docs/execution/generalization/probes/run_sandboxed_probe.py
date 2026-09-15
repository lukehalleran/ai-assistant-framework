"""Run a probe script with the adaptive exemplar store sandboxed to a temp file (fresh-user
state, nothing read from or written to data/) and tone exemplar learning disabled."""
import os, runpy, sys, tempfile
assert os.environ.get("TONE_EXEMPLAR_LEARNING") == "0", "set TONE_EXEMPLAR_LEARNING=0 in the environment"
sys.path.insert(0, "/home/lukeh/daemon_exec/generalization")
import utils.adaptive_exemplars as ae
tmpdir = tempfile.mkdtemp(prefix="probe_store_")
ae._STORE_PATH = os.path.join(tmpdir, "adaptive_exemplars.json")
print(f"[sandbox] adaptive store -> {ae._STORE_PATH}; TONE_EXEMPLAR_LEARNING=0")
runpy.run_path(sys.argv[1], run_name="__main__")
print(f"[sandbox] store file exists after run: {os.path.exists(ae._STORE_PATH)}")
