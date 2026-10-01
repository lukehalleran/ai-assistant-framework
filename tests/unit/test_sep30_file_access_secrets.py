"""2026-09-30 (LC3): credential files are unreachable through FileAccessManager.

class: BC-58 — .env / OAuth token files sit inside an approved folder and used
to pass the folder + extension checks.  Fakes only, on a tmp_path tree.
"""

import asyncio

import pytest

from core.file_access_manager import FileAccessManager

ENV_MARK = "ZQXENVONLYMARK"
TOKEN_MARK = "ZQXTOKENONLYMARK"
NOTES_MARK = "ZQXNOTESMARK"


@pytest.fixture
def mgr(tmp_path_factory):
    # fixed root name: pytest's tmp_path embeds the test name ("secrets" would
    # itself match the protected-name patterns)
    tmp_path = tmp_path_factory.mktemp("root")
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={ENV_MARK}\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".env.local").write_text(f"K={ENV_MARK}\n")
    (tmp_path / "sub" / "code.py").write_text("x = 1\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "google_token.json").write_text(f'{{"refresh_token": "{TOKEN_MARK}"}}\n')
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.local.yaml").write_text(f"k: {ENV_MARK}\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(f"url = {ENV_MARK}\n")
    (tmp_path / "notes.md").write_text(f"# {NOTES_MARK}\n")
    (tmp_path / "link.md").symlink_to(tmp_path / ".env")
    return FileAccessManager(
        approved_folders=[str(tmp_path)],
        allowed_extensions=[".py", ".md", ".json", ".yaml", ".yml", ".txt"],
    )


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("rel", [
    (".env",), ("sub", ".env.local"), ("data", "google_token.json"),
    ("config", "config.local.yaml"), (".git", "config"),
])
def test_secret_read_denied(mgr, rel):
    tmp_path = mgr.approved_folders[0]
    r = run(mgr.read_file(str(tmp_path.joinpath(*rel))))
    assert r["success"] is False
    assert "content" not in r
    assert ENV_MARK not in str(r) and TOKEN_MARK not in str(r)


def test_normal_files_still_readable(mgr):
    tmp_path = mgr.approved_folders[0]
    assert run(mgr.read_file(str(tmp_path / "notes.md")))["success"] is True
    assert run(mgr.read_file(str(tmp_path / "sub" / "code.py")))["success"] is True


def test_symlink_to_secret_denied(mgr):
    tmp_path = mgr.approved_folders[0]
    r = run(mgr.read_file(str(tmp_path / "link.md")))
    assert r["success"] is False and ENV_MARK not in str(r)


def test_grep_never_returns_secret_content(mgr):
    for mark in (ENV_MARK, TOKEN_MARK):
        r = run(mgr.grep_files(mark))
        assert r["success"] is True
        assert not any(mark in m for m in r["matches"]), r["matches"]
    # data/ is excluded already; also grep the token dir directly
    r = run(mgr.grep_files(TOKEN_MARK, folder=str(mgr.approved_folders[0] / "data")))
    assert not any(TOKEN_MARK in m for m in r.get("matches", []))


def test_grep_normal_hit_keeps_format(mgr):
    r = run(mgr.grep_files(NOTES_MARK, context_lines=0))
    hits = [m for m in r["matches"] if NOTES_MARK in m]
    assert hits and "notes.md:1:" in hits[0] and "\0" not in hits[0]


def test_list_hides_secrets(mgr):
    tmp_path = mgr.approved_folders[0]
    for recursive in (False, True):
        r = run(mgr.list_directory(str(tmp_path), recursive=recursive))
        names = [e["path"] for e in r["entries"]]
        assert "notes.md" in names
        assert not any(n.endswith(".env") or ".git" in n or "link.md" == n for n in names)
        assert not any("google_token" in n or "config.local" in n for n in names)
