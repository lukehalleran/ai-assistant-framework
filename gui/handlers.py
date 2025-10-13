"""
# gui/handlers.py

Module Contract
- Purpose: Orchestrates a single chat submission in the GUI: preprocesses files, routes to raw/enhanced flows, streams the response to the UI, and persists interaction + debug trace.
- Inputs:
  - handle_submit(user_text, files, history, use_raw_gpt, orchestrator, system_prompt=?, force_summarize=?, include_summaries=?, personality=?)
- Outputs:
  - Yields streaming dicts {role, content, debug?} as Gradio updates.
- Behavior:
  - RAW mode: send directly through orchestrator.process_user_query(use_raw_mode=True)
  - ENHANCED: orchestrator.prepare_prompt → response_generator.generate_streaming_response → store interaction
- Side effects:
  - Writes to conversation logger; stores to memory_system; updates debug_state for Debug Trace tab.
"""
import pandas as pd
import logging
from utils.logging_utils import log_and_time
from utils.conversation_logger import get_conversation_logger
import json
from config.app_config import load_system_prompt, ENABLE_BEST_OF, BEST_OF_N, BEST_OF_TEMPS, BEST_OF_MAX_TOKENS, BEST_OF_MIN_QUESTION, BEST_OF_MODEL, BEST_OF_MIN_TOKENS
from utils.query_checker import analyze_query
DEFAULT_SYSTEM_PROMPT = load_system_prompt()
logger = logging.getLogger("gradio_gui")


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
    logger.debug(f"[Handle Submit] Received user_text: {user_text}")

    # Get conversation logger
    conversation_logger = get_conversation_logger()

    if not user_text.strip():
        yield {"role": "assistant", "content": "⚠️ Empty input received."}
        return

    # Process files into the 'file_data' list
    file_data = []
    file_names = []  # Track file names for metadata
    if files:
        for file in files:
            file_names.append(file.name)
            try:
                if file.name.endswith(".txt"):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        file_data.append(f.read())
                elif file.name.endswith(".csv"):
                    df = pd.read_csv(file.name)
                    file_data.append(df.to_string())
                elif file.name.endswith(".py"):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        file_data.append(f.read())
                else:
                    file_data.append(f"[Unsupported file type: {file.name}]")
            except Exception as e:
                file_data.append(f"[Error reading {file.name}: {str(e)}]")

    # Merge file content into user_input
    merged_input = user_text + "\n\n" + "\n\n".join(file_data)

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
        }
        yield {"role": "assistant", "content": response_text, "debug": debug_record}
        return

    # ENHANCED MODE: build prompt first via orchestrator.prepare_prompt (do NOT pass personality here)
    full_prompt, system_prompt = await orchestrator.prepare_prompt(
        user_input=user_text,
        files=files,
        use_raw_mode=False  # enhanced mode
    )

    logger.debug(f"[Handle Submit] Final prompt being passed to model:\n{full_prompt}")

    final_output = ""
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

        use_bestof = False
        try:
            qinfo = analyze_query(user_text)
            use_bestof = bool(
                _enable_bestof and (
                    (qinfo.is_question and qinfo.token_count >= BEST_OF_MIN_TOKENS)
                    or (not BEST_OF_MIN_QUESTION)
                )
            )
            logger.info(f"[GUI] Best-of check: enabled={_enable_bestof}, is_question={qinfo.is_question}, tokens={qinfo.token_count}, min_question={BEST_OF_MIN_QUESTION}, use_bestof={use_bestof}")
        except Exception as e:
            use_bestof = bool(_enable_bestof)
            logger.info(f"[GUI] Best-of check failed: {e}, fallback use_bestof={use_bestof}")

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
                    debug_record = {
                        'mode': 'best-of-duel' if use_duel else 'best-of',
                        'query': user_text,
                        'prompt': full_prompt,
                        'system_prompt': system_prompt,
                        'response': final_output,
                        'model': bestof_model,
                        'prompt_tokens': prompt_tokens2,
                        'system_tokens': system_tokens2,
                        'total_tokens': total_tokens2,
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
                    debug_record = {
                        'mode': 'best-of-duel' if use_duel else 'best-of',
                        'query': user_text,
                        'prompt': full_prompt,
                        'system_prompt': system_prompt,
                        'response': final_output,
                        'model': bestof_model,
                        'prompt_tokens': prompt_tokens2,
                        'system_tokens': system_tokens2,
                        'total_tokens': total_tokens2,
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
                async for chunk in orchestrator.response_generator.generate_streaming_response(
                    prompt=full_prompt,
                    model_name=model_name,
                    system_prompt=system_prompt
                ):
                    final_output = smart_join(final_output, chunk)
                    # Parse in real-time to hide thinking block during streaming
                    thinking_part, final_answer = orchestrator._parse_thinking_block(final_output)
                    display_output = final_answer if final_answer else final_output
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
                debug_record = {
                    'mode': 'fallback-streaming',
                    'query': user_text,
                    'prompt': full_prompt,
                    'system_prompt': system_prompt,
                    'response': final_output,
                    'model': model_name,
                    'prompt_tokens': prompt_tokens2,
                    'system_tokens': system_tokens2,
                    'total_tokens': total_tokens2,
                }
                yield {"role": "assistant", "content": display_output, "debug": debug_record}
                debug_emitted = True
        else:
            # Default streaming path
            async for chunk in orchestrator.response_generator.generate_streaming_response(
                prompt=full_prompt,
                model_name=model_name,
                system_prompt=system_prompt
            ):
                final_output = smart_join(final_output, chunk)
                # Parse in real-time to hide thinking block during streaming
                thinking_part, final_answer = orchestrator._parse_thinking_block(final_output)
                display_output = final_answer if final_answer else final_output
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

            debug_record = {
                'mode': 'enhanced',
                'query': user_text,
                'prompt': full_prompt,
                'system_prompt': system_prompt,
                'response': final_output,
                'model': model_name,
                'prompt_tokens': prompt_tokens2,
                'system_tokens': system_tokens2,
                'total_tokens': total_tokens2,
            }
            yield {"role": "assistant", "content": final_output, "debug": debug_record}
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
            # Log successful conversation
            conversation_logger.log_interaction(
                user_input=user_text,  # Log original input for clarity
                assistant_response=final_output,
                metadata={
                    'mode': 'enhanced',
                    'files': file_names if file_names else None,
                    'personality': personality or getattr(getattr(orchestrator, "personality_manager", None), "current_personality", None),
                    'topic': getattr(orchestrator, 'current_topic', None)
                }
            )

            # Store in memory system (existing code)
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

                await orchestrator.memory_system.store_interaction(
                    query=merged_input,
                    response=response_to_store,
                    tags=tags
                )
                logger.info("[HANDLE_SUBMIT] Interaction successfully stored.")

                # No mid-session consolidation: summaries are generated at shutdown
            except Exception as e:
                logger.error(f"[HANDLE_SUBMIT] Failed to store interaction: {e}")

            # Do not yield another assistant message here; the UI already
            # received the final content during streaming. If needed, a debug
            # record is captured in-stream above.
