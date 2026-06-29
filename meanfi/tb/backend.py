from __future__ import annotations

try:
    from lineartetrahedron import tb_to_tight_binding_model
except ImportError:
    from lineartetrahedron.backend import (
        _tb_to_tight_binding_model as tb_to_tight_binding_model,
    )

__all__ = ["tb_to_tight_binding_model"]
