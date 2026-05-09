"""
gui/tabs/synthesis.py — Synthesis audit queue tab for blind review and grading.

Extracted from gui/launch.py to reduce file size.
"""
import gradio as gr


def build_synthesis_tab(orchestrator, _show_dev_tabs):
    """Build the Synthesis Audit tab. Must be called inside a gr.Blocks/Tabs context."""

    with gr.TabItem("Synthesis", visible=_show_dev_tabs):
        gr.Markdown("### Synthesis Audit Queue")
        gr.Markdown(
            "Review accepted insights (FP detection) and composite-rejected "
            "insights (FN detection). Generator labels hidden for blind review."
        )

        # --- Audit stats dashboard ---
        synth_stats_md = gr.Markdown(value="*Loading audit stats...*")

        # --- Queue filter ---
        with gr.Row():
            synth_queue_filter = gr.Radio(
                choices=["Accepted (ungraded)", "Composite rejects (ungraded)", "Graded history"],
                value="Accepted (ungraded)",
                label="View",
            )
            synth_refresh_btn = gr.Button("Refresh", variant="secondary")

        # --- Queue display ---
        synth_queue_html = gr.HTML(value="<p><em>Click Refresh to load queue.</em></p>")

        # --- Grading controls ---
        with gr.Row():
            synth_selector = gr.Dropdown(
                choices=[], label="Select insight to grade",
                interactive=True, allow_custom_value=False,
            )
        synth_id_map = gr.State({})  # label -> doc_id

        gr.Markdown(
            "**Step 1 — Binary screening** (answer honestly, don't overthink):"
        )
        with gr.Row():
            synth_q_thinking = gr.Radio(
                choices=["Yes", "No"],
                label="Does this make me think about something differently?",
                interactive=True,
            )
            synth_q_mechanism = gr.Radio(
                choices=["Yes", "No", "Unsure"],
                label="Is the mechanism it describes real, as far as I know?",
                interactive=True,
            )
            synth_q_heard = gr.Radio(
                choices=["Yes", "No"],
                label="Have I heard this connection before?",
                interactive=True,
            )

        gr.Markdown(
            "**Step 2 — Gut feel** (first instinct, don't agonize):"
        )
        with gr.Row():
            synth_grade_slider = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="1=Nonsense  2=Surface metaphor  3=True but obvious  4=Real insight  5=Breakthrough",
                interactive=True,
            )
            synth_grade_btn = gr.Button("Submit Grade", variant="primary")

        synth_notes_input = gr.Textbox(
            label="Notes (optional)",
            placeholder="Anything that stood out...",
            lines=1,
        )
        synth_grade_status = gr.Markdown(value="")

        # --- Handlers ---
        def _get_synth_memory():
            from memory.synthesis_memory import SynthesisMemory
            mc = getattr(orchestrator, "memory_system", None)
            store = getattr(mc, "chroma_store", None) if mc else None
            if not store:
                return None
            return SynthesisMemory(store)

        def _load_synth_stats() -> str:
            sm = _get_synth_memory()
            if not sm:
                return "*Synthesis memory not available.*"
            stats = sm.get_audit_stats()
            halt_icon = "HALTED" if stats["auto_halt"] else "OK"
            lines = [
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total graded | {stats['total_graded']} |",
                f"| Grade 4-5 (valid) | {stats['valid_count']} |",
                f"| Grade 1-3 (reject) | {stats['invalid_count']} |",
                f"| **FP rate** (grade 1-3 in accepted) | **{stats['fp_rate']:.1%}** |",
                f"| Avg grade | {stats.get('avg_grade', 'N/A')} |",
                f"| Ungraded accepted | {stats['ungraded_accepted']} |",
                f"| Ungraded rejects | {stats['ungraded_rejected']} |",
                f"| **Pipeline status** | **{halt_icon}** (halt at FP>{stats['fp_halt_threshold']:.0%}, min {stats['min_graded_for_halt']} graded) |",
            ]
            return "\n".join(lines)

        def _render_synth_card(doc_id: str, result, show_grade: bool = False) -> str:
            """Render one synthesis result as an HTML card. No generator label (blind)."""
            status = result.status.value if result.status else "unknown"
            coherence = result.coherence_level.name if result.coherence_level else "N/A"
            composite = f"{result.composite_score:.3f}" if result.composite_score else "N/A"

            status_colors = {
                "accepted": "#10b981", "converging": "#6366f1",
                "rejected": "#ef4444", "pending": "#6b7280",
            }
            color = status_colors.get(status, "#6b7280")

            grade_html = ""
            if show_grade and result.human_grade:
                grade_labels = {
                    "1": "Nonsense", "2": "Surface metaphor",
                    "3": "True/obvious", "4": "Real insight", "5": "Breakthrough",
                }
                try:
                    gn = int(result.human_grade)
                    gc = "#ef4444" if gn <= 2 else "#f59e0b" if gn == 3 else "#10b981"
                except (ValueError, TypeError):
                    gn = 0
                    gc = "#6b7280"
                label = grade_labels.get(result.human_grade, result.human_grade)
                grade_html = (
                    f' <span style="background:{gc};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.8em;">{result.human_grade}: {label}</span>'
                )
                # Binary screening answers
                binary_parts = []
                if result.changes_thinking is not None:
                    binary_parts.append(f"Thinking: {'Y' if result.changes_thinking else 'N'}")
                if result.mechanism_real:
                    binary_parts.append(f"Real: {result.mechanism_real}")
                if result.heard_before is not None:
                    binary_parts.append(f"Heard: {'Y' if result.heard_before else 'N'}")
                if binary_parts:
                    grade_html += f' <span style="font-size:0.8em;color:#aaa;">({" | ".join(binary_parts)})</span>'
                if result.grade_notes:
                    grade_html += f' <em style="font-size:0.85em;">— {result.grade_notes}</em>'

            claim = result.candidate.connection_claim if result.candidate else ""
            concept_a = result.candidate.concept_a if result.candidate else ""
            concept_b = result.candidate.concept_b if result.candidate else ""

            rejection_info = ""
            if result.rejection_stage:
                rejection_info = (
                    f"<p style='color:#ef4444;font-size:0.85em;'>"
                    f"Rejected at: {result.rejection_stage} — {result.rejection_reason}</p>"
                )

            return (
                f'<details style="margin:8px 0;padding:8px;border:1px solid #444;border-radius:6px;">'
                f'<summary style="cursor:pointer;font-weight:bold;">'
                f'{concept_a} &harr; {concept_b} '
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:0.8em;">{status}</span>'
                f'{grade_html}'
                f'</summary>'
                f'<p style="margin:8px 0;">{claim}</p>'
                f'<p style="font-size:0.85em;color:#aaa;">'
                f'Coherence: {coherence} | Composite: {composite} | '
                f'Novelty: {result.novelty_score_external:.3f} | '
                f'Co-occur: {result.cooccurrence_similarity:.3f}</p>'
                f'{rejection_info}'
                f'<p style="font-size:0.8em;color:#666;">ID: {doc_id}</p>'
                f'</details>'
            )

        def _load_synth_queue(view_filter: str):
            sm = _get_synth_memory()
            if not sm:
                return (
                    "<p><em>Synthesis memory not available.</em></p>",
                    gr.update(choices=[], value=None),
                    {},
                    _load_synth_stats(),
                )

            label_to_id = {}
            cards = []

            if view_filter == "Accepted (ungraded)":
                items = sm.get_ungraded(status_filter="accepted")
            elif view_filter == "Composite rejects (ungraded)":
                items = sm.get_ungraded(status_filter="rejected")
            elif view_filter == "Graded history":
                items = sm.get_graded()
            else:
                items = []

            show_grade = (view_filter == "Graded history")
            for doc_id, result in items:
                ca = result.candidate.concept_a if result.candidate else "?"
                cb = result.candidate.concept_b if result.candidate else "?"
                label = f"{ca} <> {cb}"
                label_to_id[label] = doc_id
                cards.append(_render_synth_card(doc_id, result, show_grade=show_grade))

            if not cards:
                html = "<p><em>No items in this view.</em></p>"
            else:
                html = "\n".join(cards)

            choices = list(label_to_id.keys())
            first_val = choices[0] if choices else None

            return (
                html,
                gr.update(choices=choices, value=first_val),
                label_to_id,
                _load_synth_stats(),
            )

        def _grade_synth(selected_label, id_map, q_thinking, q_mechanism, q_heard, grade_num, notes=""):
            if not selected_label or not id_map:
                return "Select an insight first."
            doc_id = id_map.get(selected_label)
            if not doc_id:
                return f"Insight not found for: {selected_label}"
            sm = _get_synth_memory()
            if not sm:
                return "Synthesis memory not available."

            # Map radio values to storage format
            changes_thinking = True if q_thinking == "Yes" else (False if q_thinking == "No" else None)
            mechanism_real = q_mechanism.lower() if q_mechanism else None
            heard_before = True if q_heard == "Yes" else (False if q_heard == "No" else None)

            grade_str = str(int(grade_num))
            ok = sm.grade_result(
                doc_id, grade_str, notes,
                changes_thinking=changes_thinking,
                mechanism_real=mechanism_real,
                heard_before=heard_before,
            )
            labels = {
                "1": "Nonsense", "2": "Surface metaphor",
                "3": "True but obvious", "4": "Real insight",
                "5": "Breakthrough",
            }
            if ok:
                binary_summary = (
                    f"Thinking: {q_thinking or '?'} | "
                    f"Real: {q_mechanism or '?'} | "
                    f"Heard before: {q_heard or '?'}"
                )
                return f"Graded **{grade_str}** — {labels.get(grade_str, '')}. ({binary_summary})"
            return "Failed to apply grade."

        # --- Wire buttons ---
        synth_refresh_btn.click(
            fn=_load_synth_queue,
            inputs=[synth_queue_filter],
            outputs=[synth_queue_html, synth_selector, synth_id_map, synth_stats_md],
        )
        synth_queue_filter.change(
            fn=_load_synth_queue,
            inputs=[synth_queue_filter],
            outputs=[synth_queue_html, synth_selector, synth_id_map, synth_stats_md],
        )

        synth_grade_btn.click(
            _grade_synth,
            inputs=[synth_selector, synth_id_map, synth_q_thinking, synth_q_mechanism, synth_q_heard, synth_grade_slider, synth_notes_input],
            outputs=[synth_grade_status],
        ).then(
            _load_synth_queue,
            inputs=[synth_queue_filter],
            outputs=[synth_queue_html, synth_selector, synth_id_map, synth_stats_md],
        )
