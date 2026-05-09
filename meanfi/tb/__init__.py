"""Tight-binding dictionary operations."""

from meanfi.tb import tb as _tb_api

__all__ = _tb_api.__all__
globals().update({name: getattr(_tb_api, name) for name in __all__})
