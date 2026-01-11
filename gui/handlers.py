"""
# gui/handlers.py

Module Contract
- Purpose: Orchestrates a single chat submission in the GUI: preprocesses files, routes to raw/enhanced/agentic flows, streams the response to the UI, and persists interaction + debug trace.
- Inputs:
  - handle_submit(user_text, files, history, use_raw_gpt, orchestrator, system_prompt=?, force_summarize=?, include_summaries=?, personality=?)
- Outputs:
  - Yields streaming dicts {role, content, debug?, is_progress?} as Gradio updates.
- Behavior:
  - RAW mode: send directly through orchestrator.process_user_query(use_raw_mode=True)
  - AGENTIC: If agentic_search.enabled and query triggers search, route through AgenticSearchController
  - ENHANCED: orchestrator.prepare_prompt → response_generator.generate_streaming_response → store interaction
- Side effects:
  - Writes to conversation logger; stores to memory_system; updates debug_state for Debug Trace tab.
"""
import logging
from utils.logging_utils import log_and_time
from utils.conversation_logger import get_conversation_logger
from utils.file_processor import FileProcessor
import json
from config.app_config import load_system_prompt, ENABLE_BEST_OF, BEST_OF_N, BEST_OF_TEMPS, BEST_OF_MAX_TOKENS, BEST_OF_MIN_QUESTION, BEST_OF_MODEL, BEST_OF_MIN_TOKENS
from utils.query_checker import analyze_query
DEFAULT_SYSTEM_PROMPT = load_system_prompt()
logger = logging.getLogger("gradio_gui")

# Initialize FileProcessor for secure file handling
file_processor = FileProcessor()


def smart_join(prev: str, new: str) -> str:
    """
    Inserts a space between tokens unless the new chunk begins with punctuation or whitespace.
    Prevents jammed-together words while respecting formatting.
    """
    if not prev:
        return new
    if prev.endswith((' ', '\n')) or new.startswith((' ', '\n', '.', ',', '?', '!', "'", '"', ")", "’", "”")):
        return prev + new
    else:
        return prev + ' ' + new


@log_and_time("Handle Submit")
async def handle_submit(
    user_text,
    files,
    history,
    use_raw_gpt,
    orchestrator,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    force_summarize=False,
    include_summaries=True,
    personality=None
):
    logger.debug(f"[Handle Submit] ENTRY - raw_mode={use_raw_gpt}")
    logger.debug(f"[Handle Submit] Received user_text: {user_text}")

    # Update activity timestamp for idle monitor
    try:
        import main
        if hasattr(main, 'update_activity_timestamp'):
            main.update_activity_timestamp()
    except Exception:
        pass  # Silently ignore if main module not available

    # Get conversation logger
    conversation_logger = get_conversation_logger()

    if not user_text.strip():
        yield {"role": "assistant", "content": "⚠️ Empty input received."}
        return

    # Process files using security-hardened FileProcessor
    # This supports .txt, .csv, .py, and .docx files with security checks
    file_names = [file.name for file in files] if files else []
    merged_input = await file_processor.process_files(user_text, files or [])

    # RAW MODE: go straight through orchestrator (personality hook is handled inside process_user_query)
    if use_raw_gpt:
        logger.info("[Handle Submit] RAW MODE ENABLED – skipping memory and prompt building.")

        response_text, debug_info = await orchestrator.process_user_query(
            user_input=merged_input,
            files=None,
            use_raw_mode=True,
            personality=personality
        )

        # Log the raw mode conversation
        conversation_logger.log_interaction(
            user_input=user_text,  # Log original input without file content for clarity
            assistant_response=response_text,
            metadata={
                'mode': 'raw',
                'files': file_names if file_names else None,
                'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
            }
        )

        # Emit final chunk including debug record for UI tracing
        # Compute token counts for the raw prompt (no system prompt)
        try:
            model_for_tokens = getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)()
            tm = getattr(orchestrator, 'tokenizer_manager', None)
            prompt_tokens = int(tm.count_tokens(merged_input, model_for_tokens)) if tm else None
        except Exception:
            prompt_tokens = None

        debug_record = {
            'mode': 'raw',
            'query': user_text,
            'prompt': merged_input,
            'system_prompt': None,
            'response': response_text,
            'model': getattr(orchestrator.model_manager, 'get_active_model_name', lambda: None)(),
            'prompt_tokens': prompt_tokens,
            'system_tokens': 0,
            'total_tokens': (prompt_tokens or 0),
            'citations': [],  # Raw mode bypasses memory
            'citations_enabled': False,
        }
        yield {"role": "assistant", "content": response_text, "debug": debug_record}
        return

    # ENHANCED MODE: build prompt first via orchestrator.prepare_prompt (do NOT pass personality here)
    logger.debug("[Handle Submit] About to call prepare_prompt")
    full_prompt, system_prompt = await orchestrator.prepare_prompt(
        user_input=user_text,
        files=files,
        use_raw_mode=False  # enhanced mode
    )
    logger.debug(f"[Handle Submit] prepare_prompt done, prompt length={len(full_prompt)}")

    logger.debug(f"[Handle Submit] Final prompt being passed to model:\n{full_prompt}")

    # Check if agentic search should be used
    _cfg = getattr(orchestrator, 'config', {}) or {}
    agentic_cfg = _cfg.get('agentic_search', {}) if isinstance(_cfg, dict) else {}
    agentic_enabled = bool(agentic_cfg.get('enabled', False))
    logger.debug(f"[Handle Submit] Agentic pre-check: enabled={agentic_enabled}")

    if agentic_enabled:
        # Quick filter: skip agentic for obvious non-search queries
        _lower = user_text.lower().strip()
        _words = _lower.split()
        _skip_patterns = [
            len(_words) < 5 and not any(w in _lower for w in ['search', 'news', 'latest', 'current', 'today', 'recent', '2026', '2025']),
            _lower.startswith(('nice', 'thanks', 'thank you', 'cool', 'great', 'awesome', 'got it', 'ok', 'okay', 'yeah', 'yes', 'no', 'nope')),
            'working' in _lower and 'search' in _lower,  # meta-comments about search
            all(w in ['yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope', 'thanks', 'thank', 'you'] for w in _words),
        ]
        if any(_skip_patterns):
            logger.debug("[Handle Submit] Agentic skipped - casual/short message")
            should_use_agentic = False
            search_terms = []
        else:
            # Check if web search should be triggered for this query using LLM-first trigger
            try:
                from utils.web_search_trigger import analyze_for_web_search_llm
                trigger_decision = await analyze_for_web_search_llm(
                    query=user_text,
                    model_manager=orchestrator.model_manager
                )
                # trigger_decision is a WebSearchDecision object with attributes
                should_use_agentic = getattr(trigger_decision, 'should_search', False)
                search_terms = getattr(trigger_decision, 'search_terms', []) or []
                logger.debug(f"[Handle Submit] Agentic trigger: should_search={should_use_agentic}, terms={search_terms}")
            except Exception as e:
                logger.warning(f"[Handle Submit] Agentic trigger check failed: {e}")
                import traceback
                traceback.print_exc()
                should_use_agentic = False
                search_terms = []

        if should_use_agentic:
            logger.warning("[Handle Submit] AGENTIC SEARCH MODE - routing through agentic controller")
            try:
                from core.agentic import AgenticSearchController, ProgressEvent

                # Get the agentic controller from orchestrator
                agentic_controller = orchestrator.agentic_controller
                model_name = orchestrator.model_manager.get_active_model_name()

                # Get initial search terms from the trigger decision we already have
                initial_terms = search_terms if search_terms else []
                logger.debug(f"[Handle Submit] Agentic initial terms: {initial_terms}")

                # Run agentic search loop
                agentic_response = ""
                logger.debug("[Handle Submit] Starting agentic loop")
                async for item in agentic_controller.run_agentic_search(
                    query=user_text,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    initial_search_terms=initial_terms,
                ):
                    if isinstance(item, ProgressEvent):
                        # Yield progress events as status messages
                        status_icon = {
                            "searching": "🔍",
                            "found_results": "📄",
                            "synthesizing": "✨",
                            "done": "✅",
                            "error": "❌",
                        }.get(item.event_type, "•")
                        status_msg = f"{status_icon} {item.message}"
                        logger.debug(f"[Handle Submit] Agentic progress: {item.event_type}")
                        yield {"role": "assistant", "content": status_msg, "is_progress": True}
                    else:
                        # Response chunk - accumulate and stream
                        agentic_response += item
                        yield {"role": "assistant", "content": agentic_response}

                # Final output from agentic search - strip thinking blocks
                final_output = agentic_response
                thinking_part, final_answer = orchestrator._parse_thinking_block(final_output)
                display_output = final_answer if final_answer else final_output
                logger.debug(f"[Handle Submit] Agentic loop done, response_len={len(final_output)}, display_len={len(display_output)}")

                # Token counts for debug
                try:
                    tm = getattr(orchestrator, 'tokenizer_manager', None)
                    prompt_tokens = int(tm.count_tokens(full_prompt, model_name)) if tm else None
                    system_tokens = int(tm.count_tokens(system_prompt or '', model_name)) if tm else 0
                    total_tokens = (prompt_tokens or 0) + (system_tokens or 0)
                except Exception:
                    prompt_tokens = None
                    system_tokens = None
                    total_tokens = None

                debug_record = {
                    'mode': 'agentic-search',
                    'query': user_text,
                    'prompt': full_prompt,
                    'system_prompt': system_prompt,
                    'response': final_output,
                    'model': model_name,
                    'prompt_tokens': prompt_tokens,
                    'system_tokens': system_tokens,
                    'total_tokens': total_tokens,
                    'citations': [],
                    'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                }
                # Yield the clean response first (without debug, to ensure it displays)
                logger.debug(f"[Handle Submit] Agentic yielding final response: {display_output[:100]}...")
                yield {"role": "assistant", "content": display_output}

                # Then yield with debug record for the debug trace
                yield {"role": "assistant", "content": display_output, "debug": debug_record}
                logger.debug("[Handle Submit] Agentic final response yielded")

                # Store interaction in memory
                try:
                    # Sanitize response before storage
                    try:
                        from core.prompt import _truncate_at_spurious_turns
                        final_output_sanitized = _truncate_at_spurious_turns(final_output)
                    except Exception as e:
                        logger.warning(f"[Handle Submit] Failed to sanitize agentic response: {e}")
                        final_output_sanitized = final_output

                    tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
                    memory_id = await orchestrator.memory_system.store_interaction(
                        query=merged_input,
                        response=final_output_sanitized,
                        tags=tags
                    )
                    logger.info(f"[Handle Submit] Agentic interaction stored with ID: {memory_id}")

                    # Log conversation
                    conversation_logger.log_interaction(
                        user_input=user_text,
                        assistant_response=final_output,
                        metadata={
                            'mode': 'agentic-search',
                            'files': file_names if file_names else None,
                            'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
                            'topic': getattr(orchestrator, 'current_topic', None),
                            'db_id': memory_id
                        }
                    )
                except Exception as e:
                    logger.error(f"[Handle Submit] Failed to store agentic interaction: {e}")

                return  # Exit after agentic search completes

            except Exception as e:
                logger.error(f"[Handle Submit] Agentic search failed, falling back to standard: {e}")
                import traceback
                logger.debug(f"[Agentic] Exception traceback:\n{traceback.format_exc()}")

    final_output = ""
    display_output = ""
    debug_emitted = False
    try:
        logger.debug(
            "[🔍 FINAL MESSAGE PAYLOAD TO OPENAI]:\n" +
            json.dumps(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': full_prompt}],
                indent=2
            )
        )

        # Respect runtime settings via orchestrator.config if available
        _cfg = getattr(orchestrator, 'config', {}) or {}
        _features = _cfg.get('features', {}) if isinstance(_cfg, dict) else {}
        _enable_bestof = bool(_features.get('enable_best_of', ENABLE_BEST_OF))
        # Force streaming in GUI unless explicitly disabled via env FORCE_STREAMING=0
        try:
            import os as _os
            if str(_os.getenv('FORCE_STREAMING', '1')).strip().lower() in {'1','true','yes','on'}:
                _enable_bestof = False
        except Exception:
            pass

        # Policy: always stream, except DUEL mode. Single-model best-of is disabled for streaming UX.
        use_bestof = False
        try:
            # Detect duel-mode configuration from runtime features
            GEN_MODELS = list(_features.get('best_of_generator_models', []))
            SEL_MODELS = list(_features.get('best_of_selector_models', []))
            _DUEL_MODE = bool(_features.get('best_of_duel_mode', False))
            duel_possible = bool(_DUEL_MODE and len(GEN_MODELS) >= 2 and len(SEL_MODELS) >= 1)
            use_bestof = duel_possible  # only enable best-of path when duel is configured
            logger.info(f"[GUI] Duel check: duel_mode={_DUEL_MODE}, gen={len(GEN_MODELS)}, sel={len(SEL_MODELS)}, use_bestof(duel-only)={use_bestof}")
        except Exception as e:
            use_bestof = False
            logger.info(f"[GUI] Duel check failed: {e}; default to streaming")

        model_name = orchestrator.model_manager.get_active_model_name()
        # Token counts for debug trace
        try:
            tm = getattr(orchestrator, 'tokenizer_manager', None)
            prompt_tokens = int(tm.count_tokens(full_prompt, model_name)) if tm else None
            system_tokens = int(tm.count_tokens(system_prompt or '', model_name)) if tm else 0
            total_tokens = (prompt_tokens or 0) + (system_tokens or 0)
        except Exception:
            prompt_tokens = None
            system_tokens = None
            total_tokens = None
        bestof_model = BEST_OF_MODEL or model_name

        if use_bestof:
            # Best-of with optional latency budget and streaming fallback
            # If BEST_OF_TEMPS not explicitly configured, anchor around current runtime temperature
            try:
                _explicit_temps = tuple(BEST_OF_TEMPS) if isinstance(BEST_OF_TEMPS, (list, tuple)) else None
            except Exception:
                _explicit_temps = None

            if _explicit_temps and len(_explicit_temps) > 0:
                _temps_to_use = _explicit_temps
            else:
                # Derive two temps around current default_temperature to reflect the Settings slider
                try:
                    _t = float(getattr(orchestrator.model_manager, 'default_temperature', 0.7))
                except Exception:
                    _t = 0.7
                # Slight spread while staying in [0.0, 1.5]
                _low = max(0.0, min(1.5, round(0.6 * _t, 2)))
                _hi = max(0.0, min(1.5, round(_t, 2)))
                _temps_to_use = (_low, _hi)

            # Read latency budget: runtime features override, else app_config default
            try:
                from config.app_config import BEST_OF_LATENCY_BUDGET_S as _DEF_BUDGET
            except Exception:
                _DEF_BUDGET = 0.0
            try:
                _budget = float(_features.get('best_of_latency_budget_s', _DEF_BUDGET))
            except Exception:
                _budget = float(_DEF_BUDGET) if _DEF_BUDGET else 0.0

            # Check for duel mode configuration (two generators + one judge)
            try:
                from config.app_config import (
                    BEST_OF_GENERATOR_MODELS as DEF_GEN_MODELS,
                    BEST_OF_SELECTOR_MODELS as DEF_SEL_MODELS,
                    BEST_OF_SELECTOR_MAX_TOKENS as DEF_SEL_MAXTOK,
                    BEST_OF_DUEL_MODE as DEF_DUEL_MODE,
                )
            except Exception:
                DEF_GEN_MODELS = []
                DEF_SEL_MODELS = []
                DEF_SEL_MAXTOK = 64
                DEF_DUEL_MODE = False

            GEN_MODELS = list(_features.get('best_of_generator_models', DEF_GEN_MODELS))
            SEL_MODELS = list(_features.get('best_of_selector_models', DEF_SEL_MODELS))
            SEL_MAXTOK = int(_features.get('best_of_selector_max_tokens', DEF_SEL_MAXTOK))
            _DUEL_MODE = bool(_features.get('best_of_duel_mode', DEF_DUEL_MODE))
            use_duel = bool(_DUEL_MODE and len(GEN_MODELS) == 2 and len(SEL_MODELS) >= 1)

            import asyncio as _a
            try:
                if _budget and _budget > 0:
                    # With latency budget
                    if use_duel:
                        # DUEL MODE: Two models (Claude Opus + GPT-5) with judge (GPT-4o-mini)
                        m1, m2 = list(GEN_MODELS)[:2]
                        judge = list(SEL_MODELS)[0]
                        logger.info(f"[GUI] DUEL MODE: {m1} vs {m2}, judge={judge}")
                        best_task = _a.create_task(
                            orchestrator.response_generator.generate_duel_and_judge(
                                prompt=full_prompt,
                                model_a=m1,
                                model_b=m2,
                                judge_model=judge,
                                system_prompt=system_prompt,
                                question_text=user_text,
                                context_hint=full_prompt,
                                max_tokens=BEST_OF_MAX_TOKENS,
                                temperature_a=(_temps_to_use[0] if len(_temps_to_use) > 0 else None),
                                temperature_b=(_temps_to_use[1] if len(_temps_to_use) > 1 else None),
                                judge_max_tokens=int(SEL_MAXTOK),
                            )
                        )
                    else:
                        # Single-model best-of
                        logger.info(f"[GUI] SINGLE-MODEL BEST-OF: {bestof_model}")
                        best_task = _a.create_task(
                            orchestrator.response_generator.generate_best_of(
                                prompt=full_prompt,
                                model_name=bestof_model,
                                system_prompt=system_prompt,
                                question_text=user_text,
                                context_hint=full_prompt,
                                n=BEST_OF_N,
                                temps=_temps_to_use,
                                max_tokens=BEST_OF_MAX_TOKENS,
                            )
                        )
                    best = await _a.wait_for(best_task, timeout=float(_budget))
                    logger.info(f"[GUI] Got best result: type={type(best)}, is_dict={isinstance(best, dict)}, keys={best.keys() if isinstance(best, dict) else 'N/A'}")
                    # Handle dict return from duel mode (with thinking blocks) or string from best-of
                    if isinstance(best, dict) and 'answer' in best:
                        final_output = best['answer']
                        display_output = final_output
                        # Yield thinking blocks for duel mode
                        thinking_data = {
                            'thinking_a': best.get('thinking_a', ''),
                            'thinking_b': best.get('thinking_b', ''),
                            'model_a': best.get('model_a', ''),
                            'model_b': best.get('model_b', ''),
                            'winner': best.get('winner', ''),
                            'scores': best.get('scores', {})
                        }
                        logger.info(f"[GUI] YIELDING THINKING DATA (with budget): models={thinking_data.get('model_a')}/{thinking_data.get('model_b')}, winner={thinking_data.get('winner')}")
                        yield {"role": "assistant", "content": "", "thinking": thinking_data}
                    else:
                        final_output = best
                        # Parse thinking block - only show final answer to user
                        _, final_answer = orchestrator._parse_thinking_block(final_output)
                        display_output = final_answer if final_answer else final_output
                    # Create debug record for best-of with latency budget
                    try:
                        tm = getattr(orchestrator, 'tokenizer_manager', None)
                        model_for_tokens = bestof_model
                        prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                        system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                        total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
                    except Exception:
                        prompt_tokens2 = None
                        system_tokens2 = None
                        total_tokens2 = None
                    # Extract citations if enabled
                    citations = []
                    debug_output = final_output
                    if getattr(orchestrator, 'enable_citations', False):
                        try:
                            memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                            if memory_id_map:
                                debug_output, citations = orchestrator._extract_citations(final_output, memory_id_map)
                        except Exception as e:
                            logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

                    debug_record = {
                        'mode': 'best-of-duel' if use_duel else 'best-of',
                        'query': user_text,
                        'prompt': full_prompt,
                        'system_prompt': system_prompt,
                        'response': debug_output,
                        'model': bestof_model,
                        'prompt_tokens': prompt_tokens2,
                        'system_tokens': system_tokens2,
                        'total_tokens': total_tokens2,
                        'citations': citations,
                        'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                    }
                    yield {"role": "assistant", "content": display_output, "debug": debug_record}
                    debug_emitted = True
                else:
                    # No budget: run to completion (non-streaming)
                    if use_duel:
                        # DUEL MODE: Two models (Claude Opus + GPT-5) with judge (GPT-4o-mini)
                        m1, m2 = list(GEN_MODELS)[:2]
                        judge = list(SEL_MODELS)[0]
                        logger.info(f"[GUI] DUEL MODE: {m1} vs {m2}, judge={judge}")
                        best = await orchestrator.response_generator.generate_duel_and_judge(
                            prompt=full_prompt,
                            model_a=m1,
                            model_b=m2,
                            judge_model=judge,
                            system_prompt=system_prompt,
                            question_text=user_text,
                            context_hint=full_prompt,
                            max_tokens=BEST_OF_MAX_TOKENS,
                            temperature_a=(_temps_to_use[0] if len(_temps_to_use) > 0 else None),
                            temperature_b=(_temps_to_use[1] if len(_temps_to_use) > 1 else None),
                            judge_max_tokens=int(SEL_MAXTOK),
                        )
                    else:
                        # Single-model best-of
                        logger.info(f"[GUI] SINGLE-MODEL BEST-OF: {bestof_model}")
                        best = await orchestrator.response_generator.generate_best_of(
                            prompt=full_prompt,
                            model_name=bestof_model,
                            system_prompt=system_prompt,
                            question_text=user_text,
                            context_hint=full_prompt,
                            n=BEST_OF_N,
                            temps=_temps_to_use,
                            max_tokens=BEST_OF_MAX_TOKENS,
                        )
                    # Handle dict return from duel mode (with thinking blocks) or string from best-of
                    if isinstance(best, dict) and 'answer' in best:
                        final_output = best['answer']
                        display_output = final_output
                        # Yield thinking blocks for duel mode
                        thinking_data = {
                            'thinking_a': best.get('thinking_a', ''),
                            'thinking_b': best.get('thinking_b', ''),
                            'model_a': best.get('model_a', ''),
                            'model_b': best.get('model_b', ''),
                            'winner': best.get('winner', ''),
                            'scores': best.get('scores', {})
                        }
                        logger.info(f"[GUI] YIELDING THINKING DATA (no budget): models={thinking_data.get('model_a')}/{thinking_data.get('model_b')}, winner={thinking_data.get('winner')}")
                        yield {"role": "assistant", "content": "", "thinking": thinking_data}
                    else:
                        final_output = best
                        # Parse thinking block - only show final answer to user
                        _, final_answer = orchestrator._parse_thinking_block(final_output)
                        display_output = final_answer if final_answer else final_output
                    # Create debug record for best-of without latency budget
                    try:
                        tm = getattr(orchestrator, 'tokenizer_manager', None)
                        model_for_tokens = bestof_model
                        prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                        system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                        total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
                    except Exception:
                        prompt_tokens2 = None
                        system_tokens2 = None
                        total_tokens2 = None
                    # Extract citations if enabled
                    citations = []
                    debug_output = final_output
                    if getattr(orchestrator, 'enable_citations', False):
                        try:
                            memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                            if memory_id_map:
                                debug_output, citations = orchestrator._extract_citations(final_output, memory_id_map)
                        except Exception as e:
                            logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

                    debug_record = {
                        'mode': 'best-of-duel' if use_duel else 'best-of',
                        'query': user_text,
                        'prompt': full_prompt,
                        'system_prompt': system_prompt,
                        'response': debug_output,
                        'model': bestof_model,
                        'prompt_tokens': prompt_tokens2,
                        'system_tokens': system_tokens2,
                        'total_tokens': total_tokens2,
                        'citations': citations,
                        'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                    }
                    yield {"role": "assistant", "content": display_output, "debug": debug_record}
                    debug_emitted = True
            except Exception as e:
                # Timeout or error: fall back to streaming
                logger.warning(f"[GUI] Best-of/duel mode failed, falling back to streaming: {e}")
                import traceback
                logger.debug(f"[GUI] Exception traceback:\n{traceback.format_exc()}")
                try:
                    if 'best_task' in locals():
                        best_task.cancel()
                except Exception:
                    pass
                thinking_started = False
                thinking_complete = False
                async for chunk in orchestrator.response_generator.generate_streaming_response(
                    prompt=full_prompt,
                    model_name=model_name,
                    system_prompt=system_prompt
                ):
                    final_output = smart_join(final_output, chunk)
                    # Parse in real-time to separate thinking from answer
                    thinking_part, final_answer = orchestrator._parse_thinking_block(final_output)

                    # If we have thinking content and haven't shown the answer yet
                    if thinking_part and not final_answer:
                        # Stream thinking block with special marker
                        thinking_started = True
                        display_output = f"💭 **Thinking...**\n\n{thinking_part}"
                        yield {"role": "assistant", "content": display_output, "is_thinking": True}
                    elif thinking_part and final_answer and not thinking_complete:
                        # Thinking is complete, answer is starting - switch to answer
                        thinking_complete = True
                        # Remove XML-like wrappers (e.g., <result> … </result>) for GUI display
                        try:
                            import re
                            m = re.match(r"^\s*<\s*result[^>]*>([\s\S]*?)<\s*/\s*result\s*>\s*$", final_answer or "", flags=re.IGNORECASE)
                            display_output = (m.group(1).strip() if m else final_answer)
                        except Exception:
                            display_output = final_answer
                        yield {"role": "assistant", "content": display_output, "is_thinking": False}
                    elif final_answer:
                        # Continue streaming the answer
                        try:
                            import re
                            m = re.match(r"^\s*<\s*result[^>]*>([\s\S]*?)<\s*/\s*result\s*>\s*$", final_answer or "", flags=re.IGNORECASE)
                            display_output = (m.group(1).strip() if m else final_answer)
                        except Exception:
                            display_output = final_answer
                        yield {"role": "assistant", "content": display_output}
                    else:
                        # No thinking block detected, stream normally
                        try:
                            import re
                            m = re.match(r"^\s*<\s*result[^>]*>([\s\S]*?)<\s*/\s*result\s*>\s*$", (final_output or ""), flags=re.IGNORECASE)
                            display_output = (m.group(1).strip() if m else final_output)
                        except Exception:
                            display_output = final_output
                        yield {"role": "assistant", "content": display_output}
                # After fallback streaming completes, emit debug record
                thinking_part_fb, final_answer_fb = orchestrator._parse_thinking_block(final_output)
                if thinking_part_fb:
                    final_output = final_answer_fb if final_answer_fb else final_output
                try:
                    tm = getattr(orchestrator, 'tokenizer_manager', None)
                    model_for_tokens = model_name
                    prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                    system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                    total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
                except Exception:
                    prompt_tokens2 = None
                    system_tokens2 = None
                    total_tokens2 = None
                # Ensure we have content to log even if no chunk set display_output yet
                _resp_for_debug = display_output or final_output

                # Truncate at spurious turn markers (training data leakage) in fallback
                try:
                    from core.prompt import _truncate_at_spurious_turns
                    _resp_for_debug = _truncate_at_spurious_turns(_resp_for_debug)
                except Exception as e:
                    logger.warning(f"[HANDLE_SUBMIT] Failed to truncate spurious turns in fallback: {e}")

                # Extract citations if enabled
                citations = []
                if getattr(orchestrator, 'enable_citations', False):
                    try:
                        memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                        if memory_id_map:
                            _resp_for_debug, citations = orchestrator._extract_citations(_resp_for_debug, memory_id_map)
                    except Exception as e:
                        logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

                debug_record = {
                    'mode': 'fallback-streaming',
                    'query': user_text,
                    'prompt': full_prompt,
                    'system_prompt': system_prompt,
                    'response': _resp_for_debug,
                    'model': model_name,
                    'prompt_tokens': prompt_tokens2,
                    'system_tokens': system_tokens2,
                    'total_tokens': total_tokens2,
                    'citations': citations,
                    'citations_enabled': getattr(orchestrator, 'enable_citations', False),
                }
                yield {"role": "assistant", "content": _resp_for_debug, "debug": debug_record}
                debug_emitted = True
        else:
            # Default streaming path with thinking block visibility
            thinking_started = False
            thinking_complete = False
            async for chunk in orchestrator.response_generator.generate_streaming_response(
                prompt=full_prompt,
                model_name=model_name,
                system_prompt=system_prompt
            ):
                logger.warning(f"[GUI HANDLER DEBUG] Received chunk: {chunk!r}")
                final_output = smart_join(final_output, chunk)
                # Parse in real-time to separate thinking from answer
                thinking_part, final_answer = orchestrator._parse_thinking_block(final_output)

                # If we have thinking content and haven't shown the answer yet
                if thinking_part and not final_answer:
                    # Stream thinking block with special marker
                    thinking_started = True
                    display_output = f"💭 **Thinking...**\n\n{thinking_part}"
                    yield {"role": "assistant", "content": display_output, "is_thinking": True}
                elif thinking_part and final_answer and not thinking_complete:
                    # Thinking is complete, answer is starting - switch to answer
                    thinking_complete = True
                    display_output = final_answer
                    yield {"role": "assistant", "content": display_output, "is_thinking": False}
                elif final_answer:
                    # Continue streaming the answer
                    try:
                        import re
                        # Strip ONLY outer wrapper tags at start/end (not tags mentioned in content)
                        # Use non-greedy match and ensure we capture everything between outer tags
                        m = re.match(r"^\s*<\s*(result|reply|response|answer)\s*>\s*([\s\S]*?)\s*<\s*/\s*\1\s*>\s*$", final_answer or "", flags=re.IGNORECASE)
                        display_output = (m.group(2).strip() if m else final_answer)
                    except Exception:
                        display_output = final_answer
                    yield {"role": "assistant", "content": display_output}
                else:
                    # No thinking block detected, stream normally
                    try:
                        import re
                        # Strip ONLY outer wrapper tags at start/end (not tags mentioned in content)
                        m = re.match(r"^\s*<\s*(result|reply|response|answer)\s*>\s*([\s\S]*?)\s*<\s*/\s*\1\s*>\s*$", (final_output or ""), flags=re.IGNORECASE)
                        display_output = (m.group(2).strip() if m else final_output)
                    except Exception:
                        display_output = final_output
                    yield {"role": "assistant", "content": display_output}

            # After streaming completes, parse thinking block for logging and storage
            thinking_part_stream, final_answer_stream = orchestrator._parse_thinking_block(final_output)
            if thinking_part_stream:
                logger.debug(f"[HANDLE_SUBMIT][THINKING BLOCK FROM STREAM]\n{thinking_part_stream}")
                # Update final_output to only include the final answer for storage
                final_output = final_answer_stream if final_answer_stream else final_output

            # After streaming completes, emit a final debug record so the Debug Trace tab is populated
            try:
                tm = getattr(orchestrator, 'tokenizer_manager', None)
                model_for_tokens = model_name
                prompt_tokens2 = int(tm.count_tokens(full_prompt, model_for_tokens)) if tm else None
                system_tokens2 = int(tm.count_tokens(system_prompt or '', model_for_tokens)) if tm else 0
                total_tokens2 = (prompt_tokens2 or 0) + (system_tokens2 or 0)
            except Exception:
                prompt_tokens2 = None
                system_tokens2 = None
                total_tokens2 = None

            # Ensure we have content to log even if no chunk set display_output yet
            _resp_for_debug = display_output or final_output

            # Truncate at spurious turn markers (training data leakage) for display
            try:
                from core.prompt import _truncate_at_spurious_turns
                _resp_for_debug = _truncate_at_spurious_turns(_resp_for_debug)
            except Exception as e:
                logger.warning(f"[HANDLE_SUBMIT] Failed to truncate spurious turns in display: {e}")

            # Extract citations if enabled
            citations = []
            if getattr(orchestrator, 'enable_citations', False):
                try:
                    # Get memory_id_map from orchestrator if available
                    memory_id_map = getattr(orchestrator, '_current_memory_id_map', {})
                    if memory_id_map:
                        _resp_for_debug, citations = orchestrator._extract_citations(_resp_for_debug, memory_id_map)
                except Exception as e:
                    logger.warning(f"[CITATIONS] Failed to extract citations: {e}")

            debug_record = {
                'mode': 'enhanced',
                'query': user_text,
                'prompt': full_prompt,
                'system_prompt': system_prompt,
                'response': _resp_for_debug,
                'model': model_name,
                'prompt_tokens': prompt_tokens2,
                'system_tokens': system_tokens2,
                'total_tokens': total_tokens2,
                'citations': citations,
                'citations_enabled': getattr(orchestrator, 'enable_citations', False),
            }
            yield {"role": "assistant", "content": _resp_for_debug, "debug": debug_record}
            debug_emitted = True

    except Exception as e:
        logger.error(f"[HANDLE_SUBMIT] Streaming error: {e}")
        error_message = f"⚠️ Streaming error: {str(e)}"

        # Log error conversation
        conversation_logger.log_interaction(
            user_input=user_text,
            assistant_response=error_message,
            metadata={
                'error': str(e),
                'mode': 'enhanced',
                'files': file_names if file_names else None
            }
        )

        yield {"role": "assistant", "content": error_message}

    finally:
        # Persist interaction and debug after streaming, but do not emit additional
        # assistant content here (avoid overwriting the last streamed UI state).
        if final_output and len(user_text.strip()) > 0:
            # Store in memory system FIRST to get the db_id
            memory_id = None
            try:
                logger.info("[HANDLE_SUBMIT] Storing interaction in memory...")
                tags = [f"topic:{getattr(orchestrator, 'current_topic', 'general') or 'general'}", "topic:general"]
                # Ensure corpus capacity is generous during testing (override at runtime)
                try:
                    cm = getattr(getattr(orchestrator, "memory_system", None), "corpus_manager", None)
                    if cm and hasattr(cm, "max_entries"):
                        # Default test cap to 5000 if not set via env
                        import os as _os
                        cm.max_entries = int(_os.getenv("CORPUS_MAX_ENTRIES", "5000"))
                except Exception:
                    pass

                # Parse thinking block from final_output before storing
                # Only store the final answer, not the thinking process
                thinking_part, final_answer = orchestrator._parse_thinking_block(final_output)
                if thinking_part:
                    logger.debug(f"[HANDLE_SUBMIT][THINKING BLOCK]\n{thinking_part}")
                response_to_store = final_answer if final_answer else final_output

                # Truncate at spurious turn markers (training data leakage)
                try:
                    from core.prompt import _truncate_at_spurious_turns
                    response_to_store = _truncate_at_spurious_turns(response_to_store)
                except Exception as e:
                    logger.warning(f"[HANDLE_SUBMIT] Failed to truncate spurious turns: {e}")

                # Sanitize any echoed prompt headers before storing
                try:
                    import re
                    header_patterns = [
                        r"^\s*\[TIME CONTEXT\]",
                        r"^\s*\[RECENT CONVERSATION[^\]]*\]",
                        r"^\s*\[RELEVANT INFORMATION\]",
                        r"^\s*\[RELEVANT MEMORIES\]",
                        r"^\s*\[FACTS[ ^\]]*\]",
                        r"^\s*\[RECENT FACTS\]",
                        r"^\s*\[CURRENT MESSAGE FACTS\]",
                        r"^\s*\[DIRECTIVES\]",
                        r"^\s*\[CURRENT USER QUERY[ ^\]]*\]",
                        r"^\s*\[USER INPUT\]",
                        r"^\s*\[BACKGROUND KNOWLEDGE\]",
                        r"^\s*\[CONVERSATION SUMMARIES[ ^\]]*\]",
                        r"^\s*\[RECENT REFLECTIONS[ ^\]]*\]",
                        r"^\s*\[SESSION REFLECTIONS[ ^\]]*\]",
                    ]
                    header_re = re.compile("(" + ")|(".join(header_patterns) + ")", re.IGNORECASE)
                    lines = []
                    skip = False
                    for line in (response_to_store.splitlines() if response_to_store else []):
                        if header_re.search(line):
                            skip = True
                            continue
                        if skip:
                            if not line.strip():
                                skip = False
                            continue
                        lines.append(line)
                    response_to_store = "\n".join(lines).strip()
                except Exception:
                    pass

                memory_id = await orchestrator.memory_system.store_interaction(
                    query=merged_input,
                    response=response_to_store,
                    tags=tags
                )
                logger.info(f"[HANDLE_SUBMIT] Interaction successfully stored with ID: {memory_id}")

                # No mid-session consolidation: summaries are generated at shutdown
            except Exception as e:
                logger.error(f"[HANDLE_SUBMIT] Failed to store interaction: {e}")

            # Log conversation with db_id (after storing to get the ID)
            conversation_logger.log_interaction(
                user_input=user_text,  # Log original input for clarity
                assistant_response=final_output,
                metadata={
                    'mode': 'enhanced',
                    'files': file_names if file_names else None,
                    'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
                    'topic': getattr(orchestrator, 'current_topic', None),
                    'db_id': memory_id  # Add database ID for traceability
                }
            )

            # Do not yield another assistant message here; the UI already
            # received the final content during streaming. If needed, a debug
            # record is captured in-stream above.
