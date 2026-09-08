"""The vendor shim must be present and must announce itself if it stops applying."""

from __future__ import annotations

import numpy as np
import pytest

from minispatial.export.coreml import (
    COREML_SHIM_REASON,
    assert_shim_installed,
    install_numpy2_cast_shim,
)


def test_shim_is_installed_on_import():
    assert_shim_installed()


def test_install_is_idempotent():
    assert install_numpy2_cast_shim() is False, "already installed at import time"
    assert_shim_installed()


def test_guard_fires_if_coremltools_reverts():
    """A coremltools upgrade that replaces _cast must fail loudly, not silently."""
    from coremltools.converters.mil.frontend.torch import ops as torch_ops

    patched = torch_ops._cast
    torch_ops._cast = lambda *a, **k: None  # simulate an unpatched vendor upgrade
    try:
        with pytest.raises(AssertionError, match="shim is not installed"):
            assert_shim_installed()
    finally:
        torch_ops._cast = patched
    assert_shim_installed()


def test_shim_is_only_needed_under_numpy_2():
    """Pins why the shim exists, so a numpy downgrade makes this test read oddly."""
    assert np.__version__.split(".")[0] >= "2"
    assert "numpy" in COREML_SHIM_REASON
