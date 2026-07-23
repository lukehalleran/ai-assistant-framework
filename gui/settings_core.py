"""
Module Contract — gui/settings_core.py

Purpose:
    THE runtime-settings logic, extracted from closures in gui/tabs/settings.py
    (2026-07-14) so the Gradio Settings tab and the FastAPI settings routes
    (api/routes/settings.py) call the SAME deployed functions — no forked
    apply logic between the two UIs.

Each apply_* function:
    - mutates the live orchestrator (config dict / model_manager / consolidator)
      exactly as the original Gradio closure did,
    - persists to config/config.yaml via save_settings(),
    - returns {"ok": bool, "persisted": bool, "message": str}.
      ok=False → validation error, nothing changed.
      ok=True, persisted=False → runtime applied but the YAML write failed.

Inputs:  orchestrator + primitive values.
Outputs: result dicts; get_settings_snapshot() for populating either UI.
Side effects: orchestrator mutation, config/config.yaml writes,
    SUMMARY_EVERY_N env var (summary cadence only — historical behavior).
"""

import os
from pathlib import Path
from typing import Callable, Optional

from utils.logging_utils import get_logger

logger = get_logger("settings_core")


# ---- persistence -----------------------------------------------------------

def load_settings() -> dict:
    """Load persisted settings from config/config.yaml (best-effort)."""
    try:
        import yaml
        cfg_path = Path("config") / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except (IOError, OSError, ImportError):
        pass
    return {}


def save_settings(updater: Callable[[dict], None]):
    """Read-modify-write config/config.yaml. Returns (ok, error_or_None)."""
    try:
        import yaml
        cfg_path = Path("config") / "config.yaml"
        data = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        updater(data)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return True, None
    except Exception as e:
        return False, str(e)


def _result(ok: bool, persisted: bool, message: str) -> dict:
    return {"ok": ok, "persisted": persisted, "message": message}


# ---- snapshot (populates both the Gradio tab defaults and GET /api/settings)

def model_choices(orchestrator) -> list:
    try:
        mm = orchestrator.model_manager
        api_aliases = list(getattr(mm, "api_models", {}).keys())
        local_models = list(getattr(mm, "models", {}).keys())
        return sorted(set(api_aliases + local_models)) or [
            mm.get_active_model_name() or "gpt-4-turbo"
        ]
    except (AttributeError, TypeError):
        return []


def get_settings_snapshot(orchestrator) -> dict:
    """Current values for every settings section (file values, live fallbacks)."""
    settings = load_settings()
    feat = settings.get("features", {}) or {}
    ws = settings.get("web_search", {}) or {}
    models_cfg = settings.get("models", {}) or {}
    syn = settings.get("synthesis_pooled", {}) or {}
    props = settings.get("code_proposals", {}) or {}
    mm = getattr(orchestrator, "model_manager", None)

    try:
        summary_n = int(orchestrator.memory_system.consolidator.consolidation_threshold)
    except (AttributeError, TypeError, ValueError):
        summary_n = 10

    gens = list(feat.get("best_of_generator_models", []) or [])
    return {
        "streaming": {
            "disable_best_of": not bool(feat.get("enable_best_of", True)),
            "disable_query_rewrite": not bool(feat.get("enable_query_rewrite", True)),
            "disable_llm_summaries": bool(feat.get("disable_llm_summaries", False)),
            "best_of_latency_budget_s": float(feat.get("best_of_latency_budget_s", 0) or 0),
        },
        "web_search": {
            "enabled": bool(ws.get("enabled", True)),
            "daily_credit_limit": int(ws.get("daily_credit_limit", 100)),
        },
        "duel": {
            "enabled": bool(feat.get("best_of_duel_mode", False)),
            "model_1": gens[0] if len(gens) > 0 else None,
            "model_2": gens[1] if len(gens) > 1 else None,
        },
        "tokens": {
            "best_of_max_tokens": int(feat.get("best_of_max_tokens", 128)),
            "judge_max_tokens": int(feat.get("best_of_selector_max_tokens", 64)),
            "streaming_max_tokens": int(
                models_cfg.get("default_max_tokens",
                               getattr(mm, "default_max_tokens", 2048) or 2048)
            ),
        },
        "temperature": float(
            models_cfg.get("default_temperature",
                           getattr(mm, "default_temperature", 0.7) or 0.7)
        ),
        "summary_every_n": summary_n,
        "synthesis": {
            "enabled": bool(syn.get("enabled", False)),
            "candidates_per_session": int(syn.get("candidates_per_session", 8)),
        },
        "proposals": {
            "enabled": bool(props.get("enabled", True)),
            "max_per_session": int(props.get("max_per_session", 5)),
        },
        "model_choices": model_choices(orchestrator),
    }


# ---- apply functions (one per Settings section) ----------------------------

def apply_streaming(orchestrator, *, disable_best_of: bool, disable_query_rewrite: bool,
                    disable_llm_summaries: bool, best_of_latency_budget_s: float,
                    save: Optional[Callable] = None) -> dict:
    save = save or save_settings
    try:
        cfg = getattr(orchestrator, "config", {}) or {}
        feats = cfg.setdefault("features", {}) if isinstance(cfg, dict) else {}
        feats["enable_best_of"] = not bool(disable_best_of)
        feats["enable_query_rewrite"] = not bool(disable_query_rewrite)
        feats["disable_llm_summaries"] = bool(disable_llm_summaries)
        feats["best_of_latency_budget_s"] = float(best_of_latency_budget_s)
        try:
            pb = getattr(orchestrator, "prompt_builder", None)
            if pb is not None:
                setattr(pb, "force_llm_summaries", not disable_llm_summaries)
        except (AttributeError, TypeError):
            pass
        ok, err = save(lambda d: d.setdefault("features", {}).update({
            "enable_best_of": not bool(disable_best_of),
            "enable_query_rewrite": not bool(disable_query_rewrite),
            "disable_llm_summaries": bool(disable_llm_summaries),
            "best_of_latency_budget_s": float(best_of_latency_budget_s),
        }))
        if not ok:
            return _result(True, False, f"Applied runtime settings. Persist failed: {err}")
        return _result(True, True, "Streaming settings updated (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_web_search(orchestrator, *, enabled: bool, daily_credit_limit: int,
                     save: Optional[Callable] = None) -> dict:
    save = save or save_settings
    try:
        cfg = getattr(orchestrator, "config", {}) or {}
        ws_cfg = cfg.setdefault("web_search", {}) if isinstance(cfg, dict) else {}
        ws_cfg["enabled"] = bool(enabled)
        ws_cfg["daily_credit_limit"] = int(daily_credit_limit)
        try:
            import config.app_config as app_cfg
            app_cfg.WEB_SEARCH_ENABLED = bool(enabled)
            app_cfg.WEB_SEARCH_DAILY_CREDIT_LIMIT = int(daily_credit_limit)
        except (ImportError, AttributeError):
            pass
        ok, err = save(lambda d: d.setdefault("web_search", {}).update({
            "enabled": bool(enabled),
            "daily_credit_limit": int(daily_credit_limit),
        }))
        if not ok:
            return _result(True, False, f"Applied runtime settings. Persist failed: {err}")
        return _result(True, True,
                       f"Web search settings updated: enabled={enabled}, daily_limit={daily_credit_limit}")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_duel(orchestrator, *, enabled: bool, model_1: str, model_2: str,
               save: Optional[Callable] = None) -> dict:
    save = save or save_settings
    try:
        m1 = (model_1 or "").strip()
        m2 = (model_2 or "").strip()
        if enabled and (not m1 or not m2):
            return _result(False, False, "Select both Model 1 and Model 2.")
        if enabled and m1 == m2:
            return _result(False, False, "Pick two different models for duel mode.")
        cfg = getattr(orchestrator, "config", {}) or {}
        feats = cfg.setdefault("features", {}) if isinstance(cfg, dict) else {}
        feats["best_of_duel_mode"] = bool(enabled)
        feats["best_of_generator_models"] = (
            [m1, m2] if m1 and m2 else feats.get("best_of_generator_models", [])
        )

        def _updater(d):
            f = d.setdefault("features", {})
            f["best_of_duel_mode"] = bool(enabled)
            if m1 and m2:
                f["best_of_generator_models"] = [m1, m2]

        ok, err = save(_updater)
        if not ok:
            return _result(True, False, f"Applied runtime duel settings. Persist failed: {err}")
        return _result(True, True,
                       f"Duel mode={'ON' if enabled else 'OFF'} | Model 1={m1 or '-'} "
                       f"Model 2={m2 or '-'} (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_tokens(orchestrator, *, best_of_max_tokens: int, judge_max_tokens: int,
                 streaming_max_tokens: int, save: Optional[Callable] = None) -> dict:
    save = save or save_settings
    try:
        gen_max = int(best_of_max_tokens)
        judge_max = int(judge_max_tokens)
        stream_max = int(streaming_max_tokens)
        cfg = getattr(orchestrator, "config", {}) or {}
        feats = cfg.setdefault("features", {}) if isinstance(cfg, dict) else {}
        models_cfg = cfg.setdefault("models", {}) if isinstance(cfg, dict) else {}
        feats["best_of_max_tokens"] = gen_max
        feats["best_of_selector_max_tokens"] = judge_max
        models_cfg["default_max_tokens"] = stream_max
        try:
            mm = getattr(orchestrator, "model_manager", None)
            if mm is not None:
                setattr(mm, "default_max_tokens", stream_max)
        except (AttributeError, TypeError, ValueError):
            pass

        def _updater(d):
            f = d.setdefault("features", {})
            f["best_of_max_tokens"] = gen_max
            f["best_of_selector_max_tokens"] = judge_max
            m = d.setdefault("models", {})
            m["default_max_tokens"] = stream_max

        ok, err = save(_updater)
        if not ok:
            return _result(True, False, f"Applied runtime tokens. Persist failed: {err}")
        return _result(True, True,
                       f"Applied: generators={gen_max} judge={judge_max} "
                       f"streaming={stream_max} (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_temperature(orchestrator, *, temperature: float,
                      save: Optional[Callable] = None) -> dict:
    save = save or save_settings
    try:
        t = float(temperature)
        try:
            mm = getattr(orchestrator, "model_manager", None)
            if mm is not None:
                setattr(mm, "default_temperature", t)
        except (AttributeError, TypeError):
            pass
        ok, err = save(lambda d: d.setdefault("models", {}).update(
            {"default_temperature": t}))
        if not ok:
            return _result(True, False,
                           f"Applied runtime temperature={t:.2f}. Persist failed: {err}")
        return _result(True, True, f"Model temperature set to {t:.2f} (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_summary_cadence(orchestrator, *, every_n: int,
                          save: Optional[Callable] = None) -> dict:
    save = save or save_settings
    try:
        n = int(every_n)
        try:
            mc = getattr(orchestrator, "memory_system", None)
            if mc and getattr(mc, "consolidator", None):
                mc.consolidator.consolidation_threshold = n
        except (AttributeError, TypeError):
            pass
        try:
            pb = getattr(orchestrator, "prompt_builder", None)
            if pb and getattr(pb, "consolidator", None):
                pb.consolidator.consolidation_threshold = n
        except (AttributeError, TypeError):
            pass
        try:
            os.environ["SUMMARY_EVERY_N"] = str(n)
        except (OSError, TypeError):
            pass
        ok, err = save(lambda d: d.setdefault("memory", {}).update(
            {"summary_interval": n}))
        if not ok:
            return _result(True, False, f"Applied (runtime). Persist failed: {err}")
        return _result(True, True, f"Summary cadence updated: every {n} exchanges (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_synthesis(orchestrator, *, enabled: bool, candidates_per_session: int,
                    save: Optional[Callable] = None) -> dict:
    """Synthesis dreaming (shutdown LLM step). ON enables the pooled discovery
    generator (the sole live one — retired tiers stay retired); OFF also forces
    the legacy generator flag off so `_run_synthesis_dreaming`'s
    (GENERATOR or POOLED) gate can't resurrect dreaming from a stale yaml."""
    save = save or save_settings
    try:
        n = int(candidates_per_session)
        if not 1 <= n <= 20:
            return _result(False, False, "Candidates per session must be 1-20.")
        cfg = getattr(orchestrator, "config", {}) or {}
        if isinstance(cfg, dict):
            cfg.setdefault("synthesis_pooled", {}).update(
                {"enabled": bool(enabled), "candidates_per_session": n})
        try:
            import config.app_config as app_cfg
            app_cfg.SYNTHESIS_POOLED_ENABLED = bool(enabled)
            app_cfg.SYNTHESIS_POOLED_CANDIDATES_PER_SESSION = n
            if not enabled:
                app_cfg.SYNTHESIS_GENERATOR_ENABLED = False
        except (ImportError, AttributeError):
            pass

        def _updater(d):
            d.setdefault("synthesis_pooled", {}).update(
                {"enabled": bool(enabled), "candidates_per_session": n})
            if not enabled:
                d.setdefault("synthesis_generator", {})["enabled"] = False

        ok, err = save(_updater)
        if not ok:
            return _result(True, False, f"Applied runtime synthesis settings. Persist failed: {err}")
        return _result(True, True,
                       f"Synthesis dreaming {'ON' if enabled else 'OFF'}, "
                       f"{n} candidates per shutdown (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")


def apply_proposals(orchestrator, *, enabled: bool, max_per_session: int,
                    save: Optional[Callable] = None) -> dict:
    """Goal-directed code proposals (shutdown LLM step)."""
    save = save or save_settings
    try:
        n = int(max_per_session)
        if not 1 <= n <= 10:
            return _result(False, False, "Max proposals per session must be 1-10.")
        cfg = getattr(orchestrator, "config", {}) or {}
        if isinstance(cfg, dict):
            cfg.setdefault("code_proposals", {}).update(
                {"enabled": bool(enabled), "max_per_session": n})
        try:
            import config.app_config as app_cfg
            app_cfg.CODE_PROPOSALS_ENABLED = bool(enabled)
            app_cfg.CODE_PROPOSALS_MAX_PER_SESSION = n
        except (ImportError, AttributeError):
            pass
        ok, err = save(lambda d: d.setdefault("code_proposals", {}).update(
            {"enabled": bool(enabled), "max_per_session": n}))
        if not ok:
            return _result(True, False, f"Applied runtime proposal settings. Persist failed: {err}")
        return _result(True, True,
                       f"Code proposals {'ON' if enabled else 'OFF'}, "
                       f"max {n} per shutdown (persisted).")
    except Exception as e:
        return _result(False, False, f"Failed to apply: {e}")
