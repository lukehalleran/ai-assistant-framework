"""
gui/theme.py — Dark theme CSS and Gradio theme configuration.

Extracted from gui/launch.py to reduce file size.
"""
from gradio import themes

DARK_CHATBOT_CSS = """
/* Chatbot message bubbles - dark background with light text */
.message-wrap .message {
    color: #f3f4f6 !important;
}
.message-wrap .bot {
    background-color: #374151 !important;
    color: #f3f4f6 !important;
}
.message-wrap .user {
    background-color: #1e40af !important;
    color: #f3f4f6 !important;
}
/* Chatbot container */
.chatbot {
    background-color: #1f2937 !important;
}
/* Message text */
.message-wrap .message p,
.message-wrap .message span,
.message-wrap .message li {
    color: #f3f4f6 !important;
}
/* Code blocks in chat */
.message-wrap .message pre,
.message-wrap .message code {
    background-color: #111827 !important;
    color: #e5e7eb !important;
}

/* Debug Trace tab - Markdown content styling */
.prose pre,
.prose code,
.markdown-body pre,
.markdown-body code,
[class*="markdown"] pre,
[class*="markdown"] code {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border: 1px solid #374151 !important;
}
.prose,
.markdown-body,
[class*="markdown"] {
    color: #f3f4f6 !important;
}
.prose h1, .prose h2, .prose h3, .prose h4,
.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4,
[class*="markdown"] h1, [class*="markdown"] h2, [class*="markdown"] h3, [class*="markdown"] h4 {
    color: #f3f4f6 !important;
}
.prose strong, .prose b,
.markdown-body strong, .markdown-body b,
[class*="markdown"] strong, [class*="markdown"] b {
    color: #f3f4f6 !important;
}
.prose hr,
.markdown-body hr,
[class*="markdown"] hr {
    border-color: #4b5563 !important;
}

/* Dropdown/Select styling - fix white-on-white text */
select,
.dropdown,
[data-testid="dropdown"],
.svelte-select,
input[type="text"],
.wrap input {
    color: #f3f4f6 !important;
    background-color: #374151 !important;
}
/* Dropdown options/menu */
.dropdown-menu,
.svelte-select-list,
[role="listbox"],
[role="option"],
.options {
    color: #f3f4f6 !important;
    background-color: #374151 !important;
}
/* Dropdown option hover */
[role="option"]:hover,
.dropdown-menu li:hover {
    background-color: #4b5563 !important;
    color: #ffffff !important;
}
/* Selected dropdown value display */
.selected-item,
.single-value,
.wrap-inner span {
    color: #f3f4f6 !important;
}

/* Tab overflow menu (3-dot button) - fix white-on-white */
.tab-nav button,
.tab-nav [role="tab"],
.tabs > .tab-nav button {
    color: #f3f4f6 !important;
}
/* Tab overflow dropdown/popover menu */
[role="menu"],
[role="menuitem"],
.tab-nav [data-testid="overflow-menu"],
.tab-nav .overflow-menu,
.tab-nav details,
.tab-nav details summary,
.tab-nav details[open] > ul,
.tab-nav details ul li,
.tab-nav details ul li button {
    color: #f3f4f6 !important;
    background-color: #1f2937 !important;
}
.tab-nav details ul li:hover,
.tab-nav details ul li button:hover,
[role="menuitem"]:hover {
    background-color: #374151 !important;
    color: #ffffff !important;
}
"""


def get_dark_theme():
    """Create a dark theme for the Gradio interface."""
    main_font = [
        themes.GoogleFont("JetBrains Mono"),
        "ui-monospace",
        "Consolas",
        "monospace",
    ]
    return themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=main_font,
        font_mono=main_font,
        text_size="md",
    ).set(
        # Background colors
        body_background_fill="rgb(17, 24, 39)",
        body_background_fill_dark="rgb(17, 24, 39)",
        block_background_fill="rgb(31, 41, 55)",
        block_background_fill_dark="rgb(31, 41, 55)",
        block_border_color="rgb(55, 65, 81)",
        block_border_color_dark="rgb(55, 65, 81)",
        block_label_background_fill="rgb(31, 41, 55)",
        block_label_background_fill_dark="rgb(31, 41, 55)",
        input_background_fill="rgb(55, 65, 81)",
        input_background_fill_dark="rgb(55, 65, 81)",
        # Button colors
        button_primary_background_fill="rgb(59, 130, 246)",
        button_primary_background_fill_dark="rgb(59, 130, 246)",
        button_secondary_background_fill="rgb(75, 85, 99)",
        button_secondary_background_fill_dark="rgb(75, 85, 99)",
        # Text colors - white/light for dark background
        body_text_color="rgb(243, 244, 246)",
        body_text_color_dark="rgb(243, 244, 246)",
        block_title_text_color="rgb(243, 244, 246)",
        block_title_text_color_dark="rgb(243, 244, 246)",
        block_label_text_color="rgb(209, 213, 219)",
        block_label_text_color_dark="rgb(209, 213, 219)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        button_secondary_text_color="rgb(243, 244, 246)",
        button_secondary_text_color_dark="rgb(243, 244, 246)",
    )
