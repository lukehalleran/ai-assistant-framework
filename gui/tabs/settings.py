"""
gui/tabs/settings.py — Runtime settings tab for streaming, web search, duel mode, tokens, temperature.

Extracted from gui/launch.py to reduce file size.
"""
import logging
import gradio as gr


def build_settings_tab(orchestrator, _load_settings, _save_settings):
    """Build the Settings tab. Must be called inside a gr.Blocks/Tabs context."""

    with gr.TabItem("Settings"):
        gr.Markdown("### ⚙️ Runtime Settings")

        # --- Faster Streaming options ---
        gr.Markdown("### ⚡ Faster Streaming (reduce first-token latency)")
        try:
            _settings = _load_settings()
            _feat = (_settings.get('features', {}) or {})
            _disable_bestof_default = not bool(_feat.get('enable_best_of', True))
            _disable_rewrite_default = not bool(_feat.get('enable_query_rewrite', True))
            _disable_summaries_default = bool(_feat.get('disable_llm_summaries', False))
        except (AttributeError, TypeError, KeyError):
            _disable_bestof_default = False
            _disable_rewrite_default = False
            _disable_summaries_default = False

        with gr.Row():
            disable_bestof = gr.Checkbox(label="Disable Best-of (no multi-sample reranking)", value=_disable_bestof_default)
            disable_rewrite = gr.Checkbox(label="Disable Query Rewrite (skip pre-call)", value=_disable_rewrite_default)
            disable_summaries = gr.Checkbox(label="Disable LLM Summaries (skip pre-call)", value=_disable_summaries_default)
        try:
            _bo_budget_default = float((_settings.get('features', {}) or {}).get('best_of_latency_budget_s', 0))
        except (ValueError, TypeError, KeyError):
            _bo_budget_default = 0.0
        bestof_budget = gr.Slider(label="Best‑of Latency Budget (s)", minimum=0.0, maximum=120.0, step=0.5, value=_bo_budget_default)
        apply_fast_btn = gr.Button("Apply Streaming Settings")
        fast_status = gr.Markdown(visible=True)

        def _apply_fast(disable_bo: bool, disable_rw: bool, disable_sums: bool, bo_budget: float):
            try:
                cfg = getattr(orchestrator, 'config', {}) or {}
                feats = cfg.setdefault('features', {}) if isinstance(cfg, dict) else {}
                feats['enable_best_of'] = not bool(disable_bo)
                feats['enable_query_rewrite'] = not bool(disable_rw)
                feats['disable_llm_summaries'] = bool(disable_sums)
                feats['best_of_latency_budget_s'] = float(bo_budget)
                try:
                    pb = getattr(orchestrator, 'prompt_builder', None)
                    if pb is not None:
                        setattr(pb, 'force_llm_summaries', False if disable_sums else True)
                except (AttributeError, TypeError):
                    pass
                ok, err = _save_settings(lambda d: d.setdefault('features', {}).update({
                    'enable_best_of': not bool(disable_bo),
                    'enable_query_rewrite': not bool(disable_rw),
                    'disable_llm_summaries': bool(disable_sums),
                    'best_of_latency_budget_s': float(bo_budget),
                }))
                if not ok:
                    return f"Applied runtime settings. Persist failed: {err}"
                return "Streaming settings updated (persisted)."
            except Exception as e:
                return f"Failed to apply: {e}"

        apply_fast_btn.click(_apply_fast, inputs=[disable_bestof, disable_rewrite, disable_summaries, bestof_budget], outputs=[fast_status])

        # --- Web Search Settings ---
        gr.Markdown("### 🔍 Web Search (Tavily API)")
        try:
            _ws_settings = _load_settings()
            _ws_cfg = (_ws_settings.get('web_search', {}) or {})
            _ws_enabled_default = bool(_ws_cfg.get('enabled', True))
            _ws_daily_limit_default = int(_ws_cfg.get('daily_credit_limit', 100))
        except (AttributeError, TypeError, KeyError, ValueError):
            _ws_enabled_default = True
            _ws_daily_limit_default = 100

        with gr.Row():
            web_search_enabled = gr.Checkbox(
                label="Enable Web Search",
                value=_ws_enabled_default,
                info="Search the web for queries requiring current information"
            )
            web_search_daily_limit = gr.Slider(
                label="Daily Credit Limit",
                minimum=10,
                maximum=500,
                step=10,
                value=_ws_daily_limit_default,
                info="Maximum credits to use per day (Tavily free tier: ~33/day)"
            )

        apply_ws_btn = gr.Button("Apply Web Search Settings")
        ws_status = gr.Markdown(visible=True)

        def _apply_web_search(enabled: bool, daily_limit: int):
            try:
                cfg = getattr(orchestrator, 'config', {}) or {}
                ws_cfg = cfg.setdefault('web_search', {}) if isinstance(cfg, dict) else {}
                ws_cfg['enabled'] = bool(enabled)
                ws_cfg['daily_credit_limit'] = int(daily_limit)
                try:
                    import config.app_config as app_cfg
                    app_cfg.WEB_SEARCH_ENABLED = bool(enabled)
                    app_cfg.WEB_SEARCH_DAILY_CREDIT_LIMIT = int(daily_limit)
                except (ImportError, AttributeError):
                    pass
                ok, err = _save_settings(lambda d: d.setdefault('web_search', {}).update({
                    'enabled': bool(enabled),
                    'daily_credit_limit': int(daily_limit),
                }))
                if not ok:
                    return f"Applied runtime settings. Persist failed: {err}"
                return f"Web search settings updated: enabled={enabled}, daily_limit={daily_limit}"
            except Exception as e:
                return f"Failed to apply: {e}"

        apply_ws_btn.click(
            _apply_web_search,
            inputs=[web_search_enabled, web_search_daily_limit],
            outputs=[ws_status]
        )

        # --- Best-of / Duel Mode ---
        gr.Markdown("### 🥊 Best‑of / Duel Mode")
        try:
            _settings = _load_settings()
            _feat = (_settings.get('features', {}) or {})
            _duel_enabled_default = bool(_feat.get('best_of_duel_mode', False))
            _gens_default = list(_feat.get('best_of_generator_models', []))
        except (AttributeError, TypeError, KeyError):
            _duel_enabled_default = False
            _gens_default = []

        try:
            _mm = orchestrator.model_manager
            _api_aliases = list(getattr(_mm, 'api_models', {}).keys())
            _local_models = list(getattr(_mm, 'models', {}).keys())
            _all_model_choices = sorted(set(_api_aliases + _local_models)) or [_mm.get_active_model_name() or 'gpt-4-turbo']
        except (AttributeError, TypeError):
            _all_model_choices = ['claude-opus-4.8', 'gpt-5.1', 'gpt-5', 'gpt-4-turbo', 'claude-opus-4.5', 'claude-opus', 'sonnet-4.6', 'sonnet-4.5', 'gpt-4o', 'gpt-4o-mini']

        _m1_value = _gens_default[0] if len(_gens_default) > 0 else (_all_model_choices[0] if _all_model_choices else None)
        _m2_value = _gens_default[1] if len(_gens_default) > 1 else (next((m for m in _all_model_choices if m != _m1_value), _m1_value))

        with gr.Row():
            duel_enable = gr.Checkbox(label="Enable Duel Mode (two models + judge)", value=_duel_enabled_default)
        with gr.Row():
            duel_model_1 = gr.Dropdown(label="Model 1", choices=_all_model_choices, value=_m1_value, interactive=True)
            duel_model_2 = gr.Dropdown(label="Model 2", choices=_all_model_choices, value=_m2_value, interactive=True)
        apply_duel_btn = gr.Button("Apply Duel Settings", variant="primary")
        duel_status = gr.Markdown(visible=True)

        def _apply_duel_settings(enable: bool, m1: str, m2: str):
            try:
                m1 = (m1 or '').strip()
                m2 = (m2 or '').strip()
                if enable and (not m1 or not m2):
                    return "Select both Model 1 and Model 2."
                if enable and m1 == m2:
                    return "Pick two different models for duel mode."
                cfg = getattr(orchestrator, 'config', {}) or {}
                feats = cfg.setdefault('features', {}) if isinstance(cfg, dict) else {}
                feats['best_of_duel_mode'] = bool(enable)
                feats['best_of_generator_models'] = [m1, m2] if m1 and m2 else feats.get('best_of_generator_models', [])

                def _updater(d):
                    f = d.setdefault('features', {})
                    f['best_of_duel_mode'] = bool(enable)
                    if m1 and m2:
                        f['best_of_generator_models'] = [m1, m2]
                ok, err = _save_settings(_updater)
                if not ok:
                    return f"Applied runtime duel settings. Persist failed: {err}"
                return f"Duel mode={'ON' if enable else 'OFF'} | Model 1={m1 or '-'} Model 2={m2 or '-'} (persisted)."
            except Exception as e:
                return f"Failed to apply: {e}"

        apply_duel_btn.click(_apply_duel_settings, inputs=[duel_enable, duel_model_1, duel_model_2], outputs=[duel_status])

        # --- Max Tokens Controls ---
        gr.Markdown("### ✂️ Max Tokens (length/speed)")
        try:
            _settings = _load_settings()
            _feat = (_settings.get('features', {}) or {})
            _models_cfg = (_settings.get('models', {}) or {})
            _gen_maxtok_default = int(_feat.get('best_of_max_tokens', 128))
            _judge_maxtok_default = int(_feat.get('best_of_selector_max_tokens', 64))
            _stream_maxtok_default = int(_models_cfg.get('default_max_tokens', getattr(orchestrator.model_manager, 'default_max_tokens', 2048)))
        except (AttributeError, TypeError, KeyError, ValueError):
            _gen_maxtok_default = 128
            _judge_maxtok_default = 64
            _stream_maxtok_default = getattr(orchestrator.model_manager, 'default_max_tokens', 2048)

        with gr.Row():
            gen_max_tok = gr.Slider(label="Duel/Best‑of Max Tokens (per answer)", minimum=16, maximum=10000, step=16, value=_gen_maxtok_default)
            judge_max_tok = gr.Slider(label="Judge Max Tokens", minimum=16, maximum=2000, step=8, value=_judge_maxtok_default)
        stream_max_tok = gr.Slider(label="Streaming Max Tokens (final answer)", minimum=256, maximum=10000, step=64, value=_stream_maxtok_default)
        apply_tok_btn = gr.Button("Apply Token Settings", variant="primary")
        tok_status = gr.Markdown(visible=True)

        def _apply_tokens(gen_max: int, judge_max: int, stream_max: int):
            try:
                gen_max = int(gen_max); judge_max = int(judge_max); stream_max = int(stream_max)
                cfg = getattr(orchestrator, 'config', {}) or {}
                feats = cfg.setdefault('features', {}) if isinstance(cfg, dict) else {}
                models_cfg = cfg.setdefault('models', {}) if isinstance(cfg, dict) else {}
                feats['best_of_max_tokens'] = gen_max
                feats['best_of_selector_max_tokens'] = judge_max
                models_cfg['default_max_tokens'] = stream_max
                try:
                    mm = getattr(orchestrator, 'model_manager', None)
                    if mm is not None:
                        setattr(mm, 'default_max_tokens', int(stream_max))
                except (AttributeError, TypeError, ValueError):
                    pass
                def _updater(d):
                    f = d.setdefault('features', {})
                    f['best_of_max_tokens'] = int(gen_max)
                    f['best_of_selector_max_tokens'] = int(judge_max)
                    m = d.setdefault('models', {})
                    m['default_max_tokens'] = int(stream_max)
                ok, err = _save_settings(_updater)
                if not ok:
                    return f"Applied runtime tokens. Persist failed: {err}"
                return f"Applied: generators={gen_max} judge={judge_max} streaming={stream_max} (persisted)."
            except Exception as e:
                return f"Failed to apply: {e}"

        apply_tok_btn.click(_apply_tokens, inputs=[gen_max_tok, judge_max_tok, stream_max_tok], outputs=[tok_status])

        # --- Model Temperature ---
        gr.Markdown("### 🌡️ Model Temperature")
        try:
            _settings = _load_settings()
            _models = (_settings.get('models', {}) or {})
            _temp_default = float(_models.get('default_temperature', getattr(orchestrator.model_manager, 'default_temperature', 0.7)))
        except (AttributeError, TypeError, KeyError, ValueError):
            _temp_default = getattr(orchestrator.model_manager, 'default_temperature', 0.7)

        with gr.Row():
            model_temp = gr.Slider(
                label="Model Temperature",
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                value=_temp_default,
            )
            apply_temp_btn = gr.Button("Apply Temperature", variant="primary")
        temp_status = gr.Markdown(visible=True)

        def _apply_temperature(t: float):
            try:
                t = float(t)
                try:
                    mm = getattr(orchestrator, 'model_manager', None)
                    if mm is not None:
                        setattr(mm, 'default_temperature', t)
                except (AttributeError, TypeError):
                    pass
                ok, err = _save_settings(lambda d: d.setdefault('models', {}).update({'default_temperature': float(t)}))
                if not ok:
                    return f"Applied runtime temperature={t:.2f}. Persist failed: {err}"
                return f"Model temperature set to {t:.2f} (persisted)."
            except Exception as e:
                return f"Failed to apply: {e}"

        apply_temp_btn.click(_apply_temperature, inputs=[model_temp], outputs=[temp_status])

        # --- Summary cadence ---
        try:
            current_n = int(getattr(getattr(orchestrator, 'memory_system', None), 'consolidator', None).consolidation_threshold)
        except (AttributeError, TypeError, ValueError):
            current_n = 10

        with gr.Row():
            summary_n = gr.Slider(
                label="Summary Every N Exchanges (applies at shutdown)",
                minimum=1,
                maximum=200,
                step=1,
                value=current_n,
            )
            apply_btn = gr.Button("Apply", variant="primary")
        status_md = gr.Markdown(visible=True)

        def _apply_summary_n(n: int):
            import os
            from pathlib import Path
            import yaml
            try:
                n = int(n)
                try:
                    mc = getattr(orchestrator, 'memory_system', None)
                    if mc and getattr(mc, 'consolidator', None):
                        mc.consolidator.consolidation_threshold = n
                except (AttributeError, TypeError):
                    pass
                try:
                    pb = getattr(orchestrator, 'prompt_builder', None)
                    if pb and getattr(pb, 'consolidator', None):
                        pb.consolidator.consolidation_threshold = n
                except (AttributeError, TypeError):
                    pass
                try:
                    os.environ['SUMMARY_EVERY_N'] = str(n)
                except (OSError, TypeError):
                    pass
                try:
                    cfg_path = Path('config') / 'config.yaml'
                    data = {}
                    if cfg_path.exists():
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f) or {}
                    mem = data.setdefault('memory', {})
                    mem['summary_interval'] = int(n)
                    with open(cfg_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(data, f, sort_keys=False)
                except Exception as _e:
                    return f"Applied (runtime). Persist failed: {_e}"
                return f"Summary cadence updated: every {n} exchanges (persisted)."
            except Exception as e:
                return f"Failed to apply: {e}"

        apply_btn.click(_apply_summary_n, inputs=[summary_n], outputs=[status_md])
