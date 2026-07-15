"""Shared mock-orchestrator factory + helpers for handler/API tests.

The canonical implementations live in tests/unit/test_handle_submit.py (kept
there so its 54 golden-path tests stay byte-for-byte unchanged); this module
re-exports them for the API tests. If test_handle_submit.py is ever
restructured, move the definitions here and re-export the other way.
"""

from tests.unit.test_handle_submit import (  # noqa: F401
    _FakeProcessedFilesResult,
    _async_gen_factory,
    _base_patches,
    _collect,
    _debug_record,
    _final_content,
    _make_file_processor_mock,
    _make_orchestrator,
)
