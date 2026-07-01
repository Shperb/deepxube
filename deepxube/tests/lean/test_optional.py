import pytest
from deepxube.domains.lean._optional import require


def test_require_raises_helpful_error_when_missing():
    with pytest.raises(ImportError) as exc:
        require("definitely_not_installed_pkg_xyz", extra="lean")
    assert "pip install deepxube[lean]" in str(exc.value)


def test_require_returns_module_when_present():
    mod = require("json", extra="lean")
    assert mod.dumps({"a": 1}) == '{"a": 1}'
