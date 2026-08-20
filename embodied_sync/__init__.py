"""embodied-sync: sync-quality validation layer for robot-learning data.

Importing this package must never pull optional dependencies (mcap, lerobot,
pylsl, rerun-sdk). Keep this module dependency-free apart from the stdlib.

The live-session entry points (``init``, ``SyncSession``, ``StreamConfig``)
are re-exported here because ``embsync.init(...)`` is the API the design
leads with — but they are resolved through a PEP 562 module ``__getattr__``
rather than imported eagerly. :mod:`embodied_sync.session` pulls in the
alignment engine, which pulls in numpy; a caller who only wants
``embodied_sync.__version__`` (or the CLI's ``--version``) should not pay
for that. The ``TYPE_CHECKING`` import below gives type checkers the real
signatures without costing anything at runtime.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

from embodied_sync.core.sample import Modality, Sample

if TYPE_CHECKING:
    from embodied_sync.session import StreamConfig, SyncSession, init

#: Lazily resolved top-level names → the module that defines them.
_LAZY_EXPORTS: dict[str, str] = {
    "StreamConfig": "embodied_sync.session",
    "SyncSession": "embodied_sync.session",
    "init": "embodied_sync.session",
}

__all__ = [
    "Modality",
    "Sample",
    "StreamConfig",
    "SyncSession",
    "__version__",
    "init",
]


def __getattr__(name: str) -> Any:
    """Resolve the lazily re-exported session names on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
