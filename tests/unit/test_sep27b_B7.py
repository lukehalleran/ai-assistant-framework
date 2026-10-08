"""
Static contract test for the Docker image (2026-09-27, batch B7, class: BC-71
documentation/build drift, BC-82 validated-only-under-dev-config).

FOLLOWUPS.md: "Docker NOT ready (pre-FastAPI-migration image): `api/` not
copied, no SPA build, port 7860 not 8000, only MiniLM pre-downloaded (needs
bge-small, ms-marco cross-encoder, gpt2)." -- the Dockerfile/docker-compose.yml
predated the 2026-07-14 FastAPI migration and were never updated: the default
`gui` mode (main.py) imports `api.app.create_app()`, which the image never
copied; no React/Vite SPA was ever built into it; everything was wired to the
legacy Gradio port 7860 instead of the FastAPI default (config/app_config.py
API_HOST/API_PORT); and only one of the four models the runtime actually
loads offline (HF_HUB_OFFLINE=1) was pre-downloaded in the builder stage.

This is a STATIC contract test: it reads Dockerfile/docker-compose.yml as
text and never invokes docker/podman (a real build needs several GB of
network -- torch/transformers wheels, npm packages, four HF model downloads
-- well beyond "the base image", and 10+ minutes; see the batch handoff for
the separate, explicitly-scoped partial `podman build --target python-base`
smoke run). Per CLAUDE.md's "Validation must call the deployed function":
every fact this test checks the Dockerfile against (the API port, the SPA
directory, and each of the four model-name literals) is read from the actual
module that uses it at runtime, never re-typed as an independent guess that
could quietly drift out of sync with the code again.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE_PATH.read_text(encoding="utf-8")


class TestDockerfileCopiesApiPackage:
    """The pre-migration image's COPY list omitted `api/` entirely, so the
    default `gui` mode's `from api.app import create_app` 404'd/crashed at
    import time before ever binding a socket."""

    def test_api_package_copied_into_runtime_image(self, dockerfile_text):
        assert re.search(r"COPY\s+(?:--\S+\s+)*api/\s+/app/api/", dockerfile_text), (
            "Dockerfile must COPY the api/ package (main.py's default 'gui' "
            "mode imports api.app.create_app())"
        )


class TestDockerfileBuildsTheSpa:
    """No SPA build existed at all -- the React UI was absent and the app
    fell back to 'API + /admin only'. FRONTEND_DIST_DIR is read from the
    live config, not assumed, so a future rename of that setting fails this
    test instead of silently going stale."""

    def test_frontend_builder_stage_present(self, dockerfile_text):
        assert re.search(
            r"FROM\s+node[:\w.\-]*\s+AS\s+frontend-builder", dockerfile_text, re.IGNORECASE
        ), "expected a Node build stage for the web/ React/Vite SPA"
        assert "npm ci" in dockerfile_text
        assert "npm run build" in dockerfile_text

    def test_built_spa_lands_where_the_deployed_app_serves_it_from(self, dockerfile_text):
        from config.app_config import FRONTEND_DIST_DIR

        # The deployed contract (api/app.py::mount_admin_and_frontend): the
        # SPA is served from FRONTEND_DIST_DIR, resolved relative to WORKDIR.
        rel_dir = FRONTEND_DIST_DIR.strip("./")
        assert rel_dir == "web/dist", (
            f"this test assumes FRONTEND_DIST_DIR is a 'web/dist'-shaped "
            f"relative path; got {FRONTEND_DIST_DIR!r} -- update the "
            f"Dockerfile COPY target to match before touching this assert"
        )
        pattern = rf"COPY\s+--from=frontend-builder[^\n]*\b{re.escape(rel_dir)}\b"
        assert re.search(pattern, dockerfile_text), (
            f"Dockerfile must copy the built SPA into /app/{rel_dir} -- the "
            f"exact path api/app.py resolves FRONTEND_DIST_DIR against"
        )


class TestDockerfilePortMatchesDeployedApi:
    """Everything was wired to the legacy Gradio port (7860); the default
    'gui' mode has served FastAPI + the SPA on API_PORT since the 2026-07-14
    migration. Port is read from config/app_config.py, never hardcoded."""

    def test_exposes_the_deployed_api_port(self, dockerfile_text):
        from config.app_config import API_PORT

        assert f"EXPOSE {API_PORT}" in dockerfile_text

    def test_healthcheck_targets_the_deployed_api_port(self, dockerfile_text):
        from config.app_config import API_PORT

        m = re.search(r"HEALTHCHECK.*?CMD\s+curl[^\n]*", dockerfile_text, re.DOTALL)
        assert m, "expected a HEALTHCHECK CMD curl line"
        assert f"localhost:{API_PORT}/health" in m.group(0)

    def test_healthcheck_is_not_pinned_to_the_legacy_gradio_port(self, dockerfile_text):
        # 7860 may still appear elsewhere (first-run wizard / --legacy-gui
        # fallback), but the HEALTHCHECK line itself must target the
        # deployed API port, not the retired hardcoded 7860.
        m = re.search(r"HEALTHCHECK.*?CMD\s+curl[^\n]*", dockerfile_text, re.DOTALL)
        assert m
        assert "localhost:7860" not in m.group(0)


class TestDockerComposePortMatchesDeployedApi:
    def test_primary_port_mapping_uses_deployed_api_port(self, compose_text):
        from config.app_config import API_PORT

        assert re.search(rf'"{API_PORT}:{API_PORT}"', compose_text), (
            f"docker-compose.yml must map {API_PORT}:{API_PORT} as the "
            f"primary port (was 7860:7860 for the legacy Gradio image)"
        )

    def test_healthcheck_uses_deployed_api_port(self, compose_text):
        from config.app_config import API_PORT

        assert f"http://localhost:{API_PORT}/health" in compose_text


class TestDockerfilePredownloadsEveryOfflineModel:
    """Only all-MiniLM-L6-v2 was pre-downloaded; HF_HUB_OFFLINE=1 at runtime
    means any model NOT baked in at build time can never load. Every model
    name below is read out of the module that actually loads it at runtime
    -- never an independent guess of "what the model probably is"."""

    def test_predownloads_the_tone_topic_shared_embedder(self, dockerfile_text):
        src = (REPO_ROOT / "models" / "model_manager.py").read_text(encoding="utf-8")
        m = re.search(r'SentenceTransformer\("([^"]+)"\)', src)
        assert m, "could not find ModelManager's cached-embedder model literal"
        model_name = m.group(1)
        assert model_name in dockerfile_text

    def test_predownloads_the_chroma_store_embedder(self, dockerfile_text):
        src = (REPO_ROOT / "memory" / "storage" / "multi_collection_chroma_store.py").read_text(
            encoding="utf-8"
        )
        m = re.search(r'os\.getenv\("CHROMA_ST_MODEL",\s*"([^"]+)"\)', src)
        assert m, "could not find the Chroma store's default embedder model literal"
        model_name = m.group(1)
        assert model_name in dockerfile_text, (
            f"Dockerfile must pre-download {model_name!r} -- the store's own "
            f"embedder AND the memory gate's scoring model since 2026-07-02"
        )

    def test_predownloads_the_gate_cross_encoder(self, dockerfile_text):
        src = (REPO_ROOT / "processing" / "gate_system.py").read_text(encoding="utf-8")
        m = re.search(r'get_cross_encoder\("([^"]+)"\)', src)
        assert m, "could not find the gate's cross-encoder model literal"
        model_name = m.group(1)
        assert model_name in dockerfile_text

    def test_predownloads_the_tokenizer_fallback(self, dockerfile_text):
        src = (REPO_ROOT / "models" / "tokenizer_manager.py").read_text(encoding="utf-8")
        assert 'AutoTokenizer.from_pretrained("gpt2")' in src, (
            "tokenizer_manager.py's fallback literal moved/changed -- update "
            "this test before trusting the Dockerfile assertion below"
        )
        assert "gpt2" in dockerfile_text


class TestDockerComposeDataPathsAgreeWithConfig:
    """docker-entrypoint.sh's own fallback default (chroma_db_v4_v2) no
    longer matches config.yaml's real default (chroma_db_v4) -- pin both
    paths explicitly in the compose environment so the entrypoint's
    pre-check directory and the app's actual store directory are the same
    one. docker-entrypoint.sh itself is out of this batch's file list, so
    the fix here is to make the explicit override win, not to edit its
    hardcoded fallback."""

    def test_chroma_path_pinned_to_the_configured_default(self, compose_text):
        import yaml

        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        configured_path = cfg["memory"]["chroma_path"].lstrip("./")
        assert re.search(rf"CHROMA_PATH:\s*/app/{re.escape(configured_path)}\b", compose_text), (
            f"docker-compose.yml's CHROMA_PATH override must match "
            f"config.yaml's memory.chroma_path ({configured_path!r})"
        )


# ---------------------------------------------------------------------------
# 2026-10-08 batch 4 (class: BC-71): docker-entrypoint.sh itself. It is a
# shell script, so these are string assertions on the file text (not a run).
# ---------------------------------------------------------------------------

ENTRYPOINT_PATH = REPO_ROOT / "docker-entrypoint.sh"


class TestDockerEntrypointAgreesWithConfig:
    @pytest.fixture(scope="class")
    def entrypoint_text(self) -> str:
        return ENTRYPOINT_PATH.read_text(encoding="utf-8")

    def test_chroma_fallback_default_matches_config(self, entrypoint_text):
        import yaml

        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        configured_path = cfg["memory"]["chroma_path"].lstrip("./")
        assert f'CHROMA_PATH="${{CHROMA_PATH:-/app/{configured_path}}}"' in entrypoint_text
        assert "chroma_db_v4_v2" not in entrypoint_text

    def test_banner_names_the_api_server_variables(self, entrypoint_text):
        from config.app_config import API_PORT

        assert "${DAEMON_API_HOST" in entrypoint_text
        assert "${DAEMON_API_PORT:-%d}" % API_PORT in entrypoint_text
        health = [l for l in entrypoint_text.splitlines() if l.startswith('echo "Health check')]
        assert health and "DAEMON_API_PORT" in health[0] and "GRADIO_PORT" not in health[0]
