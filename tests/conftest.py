from __future__ import annotations

import os

import pytest

from ea_tdc.vintage import VINTAGE_ENV


@pytest.fixture(autouse=True)
def _isolate_active_vintage():
    """The vintage lanes set the active vintage in the process environment; no test inherits one."""
    saved = os.environ.pop(VINTAGE_ENV, None)
    yield
    os.environ.pop(VINTAGE_ENV, None)
    if saved is not None:
        os.environ[VINTAGE_ENV] = saved
