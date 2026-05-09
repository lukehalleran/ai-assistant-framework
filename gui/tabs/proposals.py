"""
gui/tabs/proposals.py — Proposals tab UI for browsing, managing, and generating code proposals.

Extracted from gui/launch.py to reduce file size.
"""
import logging
import gradio as gr


def build_proposals_tab(orchestrator, _load_settings, _save_settings, _show_dev_tabs):
    """Build the Proposals tab. Must be called inside a gr.Blocks/Tabs context."""

    with gr.TabItem("Proposals", visible=_show_dev_tabs):
        gr.Markdown("### Code Proposals")

        # Filter controls
        with gr.Row():
            proposal_status_filter = gr.Dropdown(
                choices=["all", "pending", "approved", "rejected", "completed", "failed"],
                value="all", label="Status"
            )
            proposal_type_filter = gr.Dropdown(
                choices=["all", "feature", "refactor", "bugfix", "test", "docs", "infra"],
                value="all", label="Type"
            )
            refresh_proposals_btn = gr.Button("Refresh")

        # Collapsible proposal cards
        proposals_view = gr.HTML(value="<p><em>Loading proposals...</em></p>")

        # Proposal ideas generation
        with gr.Row():
            generate_btn = gr.Button("Generate Proposals Now", variant="secondary")
            check_impl_btn = gr.Button("Check Implementation", variant="secondary")
            generate_status = gr.Markdown(value="")

        # Shared state for both manage and codegen sections
        proposals_map = gr.State({})  # title -> proposal_id mapping

        # --- Manage Proposals ---
        gr.Markdown("---")
        gr.Markdown("### Manage Proposals")

        with gr.Row():
            manage_selector = gr.Dropdown(
                choices=[], label="Select Proposal to Manage",
                interactive=True, allow_custom_value=False,
            )

        with gr.Row():
            mark_built_btn = gr.Button("Mark Built", variant="primary")
            mark_rejected_btn = gr.Button("Reject", variant="stop")
            mark_approved_btn = gr.Button("Approve", variant="secondary")
            check_single_impl_btn = gr.Button("Check This", variant="secondary")

        with gr.Row():
            rejection_reason_input = gr.Textbox(
                label="Rejection reason (optional)",
                placeholder="Why is this being rejected?",
                lines=1,
            )

        manage_status = gr.Markdown(value="")

        # --- Code Generation ---
        gr.Markdown("---")
        gr.Markdown("### Generate Implementation Code")

        with gr.Row():
            codegen_selector = gr.Dropdown(
                choices=[], label="Select Proposal", interactive=True
            )
            codegen_btn = gr.Button("Generate Code", variant="primary")

        codegen_status = gr.Markdown(value="")
        codegen_output = gr.HTML(value="")

        def _load_proposals(status_f, type_f):
            """Load proposals from ChromaDB and render as collapsible HTML cards."""
            import json as _json
            try:
                chroma = getattr(orchestrator, 'memory_system', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                if not chroma_store:
                    return (
                        "<p><em>No chroma store available</em></p>",
                        gr.update(choices=[]), gr.update(choices=[]), {},
                    )
                all_items = chroma_store.list_all("proposals")
                if not all_items:
                    return (
                        "<p><em>No proposals yet. Click 'Generate Proposals Now' to create some.</em></p>",
                        gr.update(choices=[]), gr.update(choices=[]), {},
                    )

                # Collect and filter
                proposals = []
                for item in all_items:
                    meta = item.get("metadata") or {}
                    if not meta.get("proposal_id"):
                        continue
                    if status_f and status_f != "all" and meta.get("status") != status_f:
                        continue
                    if type_f and type_f != "all" and meta.get("proposal_type") != type_f:
                        continue
                    proposals.append(meta)

                # Sort by priority desc, then created_at desc
                proposals.sort(key=lambda m: (-int(m.get("priority", 5)), -float(m.get("created_at", 0))))

                if not proposals:
                    return (
                        "<p><em>No proposals match the current filters.</em></p>",
                        gr.update(choices=[]), gr.update(choices=[]), {},
                    )

                # Build dropdown choices and mapping
                dropdown_choices = []
                title_to_id = {}
                for meta in proposals:
                    title = meta.get("title", "Untitled")
                    priority = int(meta.get("priority", 5))
                    ptype = meta.get("proposal_type", "")
                    label = f"[P{priority}] {title} ({ptype})"
                    dropdown_choices.append(label)
                    title_to_id[label] = meta.get("proposal_id", "")

                # Render collapsible cards
                html_parts = []
                for i, meta in enumerate(proposals):
                    html_parts.append(_render_proposal_card(meta, i, _json))

                _first_val = dropdown_choices[0] if dropdown_choices else None
                return (
                    "\n".join(html_parts),
                    gr.update(choices=dropdown_choices, value=_first_val),
                    gr.update(choices=dropdown_choices, value=_first_val),
                    title_to_id,
                )
            except Exception as e:
                return (
                    f"<p><em>Error loading proposals: {e}</em></p>",
                    gr.update(choices=[]), gr.update(choices=[]),
                    {},
                )

        async def _generate_proposals_now():
            """On-demand proposal generation."""
            try:
                from knowledge.proposal_generator import GoalDirectedGenerator
                from memory.proposal_store import ProposalStore
                mm = getattr(orchestrator, 'model_manager', None)
                chroma = getattr(orchestrator, 'memory_system', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                if not mm or not chroma_store:
                    return "Model manager or chroma store not available."
                generator = GoalDirectedGenerator(model_manager=mm, repo_path=".")
                store = ProposalStore(chroma_store=chroma_store)
                extra_parts = []
                try:
                    cm = orchestrator.memory_system.corpus_manager
                    recent = cm.get_recent_memories(8)
                    for e in recent:
                        q = (e.get('query') or '').strip()
                        a = (e.get('response') or '').strip()
                        if q:
                            extra_parts.append(f"User: {q[:300]}")
                        if a:
                            extra_parts.append(f"Assistant: {a[:400]}")
                except (AttributeError, TypeError):
                    pass
                dedup_context = store.get_for_dedup()
                extra = "\n\n".join(extra_parts)
                if dedup_context:
                    extra += f"\n\n## Existing Proposals (avoid duplicates)\n{dedup_context}"
                proposals = await generator.generate_proposals(extra_context=extra)
                kept = 0
                for p in proposals:
                    if store.check_similarity(p):
                        continue
                    if store.store_proposal(p):
                        kept += 1
                return f"Generated {len(proposals)} proposal(s), stored {kept} new."
            except Exception as e:
                return f"Generation failed: {e}"

        async def _check_implementation_now():
            """Run full implementation detection on all pending+approved proposals."""
            try:
                from knowledge.implementation_detector import ImplementationDetector
                from memory.proposal_store import ProposalStore
                from knowledge.git_memory import GitMemoryExtractor

                chroma = getattr(orchestrator, 'memory_system', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                mm = getattr(orchestrator, 'model_manager', None)
                if not chroma_store:
                    return "Chroma store not available."

                store = ProposalStore(chroma_store=chroma_store)
                proposals = store.get_pending_and_approved()
                if not proposals:
                    return "No pending/approved proposals to check."

                detector = ImplementationDetector(
                    repo_path=".",
                    git_extractor=GitMemoryExtractor("."),
                    model_manager=mm,
                )
                results = await detector.detect_batch(proposals, lightweight=False)

                updated = 0
                for result in results:
                    if not result.skipped_reason:
                        if store.update_tracking_metadata(result.proposal_id, result):
                            updated += 1

                summary_parts = []
                for r in results:
                    if not r.skipped_reason:
                        summary_parts.append(f"- {r.status} ({r.confidence:.0%})")
                return f"Checked {len(proposals)} proposals, updated {updated}. " + " ".join(summary_parts[:5])

            except Exception as e:
                return f"Implementation check failed: {e}"

        async def _check_single_implementation(selected_label, title_map):
            """Run implementation detection on a single proposal."""
            if not selected_label or not title_map:
                return "Select a proposal first."

            proposal_id = title_map.get(selected_label)
            if not proposal_id:
                return f"Proposal not found for: {selected_label}"

            try:
                from knowledge.implementation_detector import ImplementationDetector
                from memory.proposal_store import ProposalStore
                from knowledge.git_memory import GitMemoryExtractor

                chroma = getattr(orchestrator, 'memory_system', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                mm = getattr(orchestrator, 'model_manager', None)
                if not chroma_store:
                    return "Chroma store not available."

                store = ProposalStore(chroma_store=chroma_store)
                proposal = store.get_proposal(proposal_id)
                if not proposal:
                    return f"Proposal {proposal_id} not found."

                detector = ImplementationDetector(
                    repo_path=".",
                    git_extractor=GitMemoryExtractor("."),
                    model_manager=mm,
                )
                result = detector.detect_single(proposal, lightweight=False)
                store.update_tracking_metadata(proposal_id, result)

                return (
                    f"**{result.status}** ({result.confidence:.0%})\n\n"
                    f"{result.evidence}"
                )

            except Exception as e:
                return f"Check failed: {e}"

        async def _generate_code_now(selected_label, title_map):
            """Generate full implementation code for the selected proposal."""
            import html as _html
            if not selected_label or not title_map:
                return "Select a proposal first.", ""

            proposal_id = title_map.get(selected_label)
            if not proposal_id:
                return f"Proposal not found for: {selected_label}", ""

            try:
                from knowledge.proposal_generator import GoalDirectedGenerator
                from memory.proposal_store import ProposalStore
                from memory.code_proposal import CodeProposal

                mm = getattr(orchestrator, 'model_manager', None)
                chroma = getattr(orchestrator, 'memory_system', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                if not mm or not chroma_store:
                    return "Model manager or chroma store not available.", ""

                store = ProposalStore(chroma_store=chroma_store)
                proposal = store.get_proposal(proposal_id)
                if not proposal:
                    return f"Proposal {proposal_id} not found in store.", ""

                if not proposal.implementation_steps:
                    return "This proposal has no implementation steps to generate code for.", ""

                generator = GoalDirectedGenerator(model_manager=mm, repo_path=".")
                result = await generator.generate_code_for_proposal(proposal)

                # Build HTML output with syntax-highlighted code blocks
                files = result.get("files", {})
                errors = result.get("errors", [])
                output_dir = result.get("output_dir", "")

                if not files:
                    err_msg = "; ".join(errors) if errors else "No files generated"
                    return f"Code generation produced no files: {err_msg}", ""

                html_parts = []
                html_parts.append(
                    f'<div style="padding:10px;margin-bottom:8px;background:#1a2332;'
                    f'border:1px solid #2563eb;border-radius:6px;color:#93c5fd;">'
                    f'Generated {len(files)} file(s) in <code>{_html.escape(output_dir)}</code>'
                    f'</div>'
                )

                for fpath, content in files.items():
                    escaped = _html.escape(content)
                    html_parts.append(
                        f'<details open style="margin-bottom:8px;border:1px solid #374151;'
                        f'border-radius:6px;background:#0d1117;">'
                        f'<summary style="padding:8px 12px;cursor:pointer;font-weight:600;'
                        f'color:#58a6ff;font-size:0.95em;background:#161b22;'
                        f'border-radius:6px 6px 0 0;">'
                        f'{_html.escape(fpath)}</summary>'
                        f'<pre style="margin:0;padding:12px;overflow-x:auto;'
                        f'color:#e6edf3;font-size:0.85em;line-height:1.5;'
                        f'max-height:600px;overflow-y:auto;">'
                        f'<code>{escaped}</code></pre>'
                        f'</details>'
                    )

                if errors:
                    html_parts.append(
                        f'<div style="padding:8px;color:#f87171;font-size:0.9em;">'
                        f'Errors: {_html.escape("; ".join(errors))}</div>'
                    )

                status_msg = f"Generated {len(files)} file(s) -> `{output_dir}`"
                if errors:
                    status_msg += f" ({len(errors)} errors)"

                return status_msg, "\n".join(html_parts)

            except Exception as e:
                import traceback
                logging.error(f"[GUI] Code generation error: {traceback.format_exc()}")
                return f"Code generation failed: {e}", ""

        def _update_proposal_status(selected_label, title_map, new_status, reason=""):
            """Update a proposal's status and return feedback message."""
            if not selected_label or not title_map:
                return "Select a proposal first."

            proposal_id = title_map.get(selected_label)
            if not proposal_id:
                return f"Proposal not found for: {selected_label}"

            try:
                from memory.proposal_store import ProposalStore
                from memory.code_proposal import ProposalStatus

                chroma = getattr(orchestrator, 'memory_system', None)
                chroma_store = getattr(chroma, 'chroma_store', None) if chroma else None
                if not chroma_store:
                    return "Chroma store not available."

                store = ProposalStore(chroma_store=chroma_store)
                status_enum = ProposalStatus(new_status)
                ok = store.update_status(proposal_id, status_enum, reason=reason)

                if ok:
                    return f"Proposal marked as **{new_status}**."
                else:
                    return f"Failed to update proposal status."
            except Exception as e:
                return f"Error updating status: {e}"

        # --- Wire event handlers ---
        refresh_proposals_btn.click(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

        generate_btn.click(
            _generate_proposals_now,
            inputs=[],
            outputs=[generate_status],
        ).then(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

        mark_built_btn.click(
            lambda sel, m: _update_proposal_status(sel, m, "completed"),
            inputs=[manage_selector, proposals_map],
            outputs=[manage_status],
        ).then(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

        mark_rejected_btn.click(
            lambda sel, m, r: _update_proposal_status(sel, m, "rejected", r),
            inputs=[manage_selector, proposals_map, rejection_reason_input],
            outputs=[manage_status],
        ).then(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

        mark_approved_btn.click(
            lambda sel, m: _update_proposal_status(sel, m, "approved"),
            inputs=[manage_selector, proposals_map],
            outputs=[manage_status],
        ).then(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

        codegen_btn.click(
            _generate_code_now,
            inputs=[codegen_selector, proposals_map],
            outputs=[codegen_status, codegen_output],
        )

        check_impl_btn.click(
            _check_implementation_now,
            inputs=[],
            outputs=[generate_status],
        ).then(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

        check_single_impl_btn.click(
            _check_single_implementation,
            inputs=[manage_selector, proposals_map],
            outputs=[manage_status],
        ).then(
            _load_proposals,
            inputs=[proposal_status_filter, proposal_type_filter],
            outputs=[proposals_view, manage_selector, codegen_selector, proposals_map],
        )

    # Return components needed by launch.py for demo.load
    return {
        "load_proposals": _load_proposals,
        "status_filter": proposal_status_filter,
        "type_filter": proposal_type_filter,
        "proposals_view": proposals_view,
        "manage_selector": manage_selector,
        "codegen_selector": codegen_selector,
        "proposals_map": proposals_map,
    }


def _render_proposal_card(meta, index, _json):
    """Render a single proposal as a collapsible HTML card."""
    title = meta.get("title", "Untitled")
    ptype = meta.get("proposal_type", "")
    status = meta.get("status", "")
    priority = int(meta.get("priority", 5))
    complexity = meta.get("estimated_complexity", "medium")
    reasoning = meta.get("reasoning", "")
    description = meta.get("description", "")
    created = float(meta.get("created_at", 0))
    from datetime import datetime as _dt
    created_str = _dt.fromtimestamp(created).strftime('%Y-%m-%d %H:%M') if created else ""

    open_attr = " open" if index == 0 else ""

    body = []
    body.append(f'<div style="padding:8px 12px;color:#d1d5db;font-size:0.9em;">')
    body.append(f'<b>Type:</b> {ptype} &nbsp; <b>Status:</b> {status} &nbsp; '
                f'<b>Priority:</b> {priority}/10 &nbsp; '
                f'<b>Complexity:</b> {complexity} &nbsp; '
                f'<b>Created:</b> {created_str}')

    if reasoning:
        body.append(f'<p><b>Reasoning:</b> {reasoning}</p>')
    if description:
        body.append(f'<p><b>Description:</b> {description}</p>')

    try:
        steps = _json.loads(meta.get("steps_json", "[]"))
        if steps:
            body.append('<p><b>Implementation Steps:</b></p><ol>')
            for s in steps:
                fp = f' <code>{s.get("file_path")}</code>' if s.get("file_path") else ""
                body.append(f'<li>[{s.get("action", "modify")}]{fp} {s.get("description", "")}</li>')
            body.append('</ol>')
    except (ValueError, TypeError):
        pass

    try:
        files = _json.loads(meta.get("affected_files_json", "[]"))
        if files:
            body.append('<p><b>Affected Files:</b> ' + ", ".join(f'<code>{f}</code>' for f in files) + '</p>')
    except (ValueError, TypeError):
        pass

    try:
        tags = _json.loads(meta.get("tags_json", "[]"))
        if tags:
            body.append('<p><b>Tags:</b> ' + ", ".join(tags) + '</p>')
    except (ValueError, TypeError):
        pass

    body.append('</div>')
    body_html = "\n".join(body)

    # Status badge color
    _badge_colors = {
        "pending": "#6b7280",
        "approved": "#f59e0b",
        "completed": "#10b981",
        "rejected": "#ef4444",
        "failed": "#ef4444",
    }
    badge_color = _badge_colors.get(status, "#6b7280")
    badge_html = (
        f'<span style="display:inline-block;padding:1px 8px;'
        f'border-radius:9999px;font-size:0.75em;font-weight:600;'
        f'color:#fff;background:{badge_color};margin-left:8px;'
        f'vertical-align:middle;">{status}</span>'
    )

    # Implementation tracking badge
    impl_status = meta.get("implementation_status", "not_checked")
    impl_conf = float(meta.get("implementation_confidence", 0))
    impl_badge_html = ""
    if impl_status != "not_checked":
        _impl_colors = {
            "confirmed": "#10b981",
            "likely": "#f59e0b",
            "uncertain": "#6b7280",
            "not_implemented": "#6b7280",
        }
        impl_color = _impl_colors.get(impl_status, "#6b7280")
        impl_evidence = meta.get("implementation_evidence", "")
        impl_badge_html = (
            f'<span style="display:inline-block;padding:1px 8px;'
            f'border-radius:9999px;font-size:0.75em;font-weight:600;'
            f'color:#fff;background:{impl_color};margin-left:4px;'
            f'vertical-align:middle;" title="{impl_evidence}">'
            f'{impl_status} {impl_conf:.0%}</span>'
        )

    return (
        f'<details{open_attr} style="margin-bottom:6px;border:1px solid #374151;'
        f'border-radius:6px;background:#1f2937;">'
        f'<summary style="padding:10px 14px;cursor:pointer;font-weight:600;'
        f'color:#f3f4f6;font-size:1.0em;">'
        f'{title} '
        f'<span style="font-weight:400;color:#9ca3af;font-size:0.85em;">'
        f'({ptype} | P{priority})</span>'
        f'{badge_html}'
        f'{impl_badge_html}'
        f'</summary>\n{body_html}\n</details>'
    )
