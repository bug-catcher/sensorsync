"""MCAP adapter/exporter importability tests (Milestone 4).

The MCAP contract adapter/exporter modules must be importable
*without* the ``mcap`` extra installed (the top-level import block
does not touch ``mcap``), so an ordinary introspective import from
the base install works.

The stricter guard that a stray top-level ``import mcap`` cannot
sneak in later lives in ``tests/test_optional_deps.py``, which runs a
subprocess with a ``MetaPathFinder`` blocking ``mcap``. That test
also covers the ``embodied_sync.adapters`` and
``embodied_sync.exporters`` subpackages.

These tests intentionally avoid ``pytest.importorskip("mcap")``:
importing ``mcap`` in the pytest process would taint ``sys.modules``
for the whole session and break
``tests/test_package.py::test_import_pulls_no_optional_dependencies``.
Runtime behaviour is covered by ``tests/test_adapter_mcap.py`` and
``tests/test_exporter_mcap.py``.
"""

from __future__ import annotations


def test_adapter_mcap_is_importable_on_base_install() -> None:
    """``embodied_sync.adapters.mcap`` imports without the mcap extra."""
    from embodied_sync.adapters import mcap as adapter

    assert callable(adapter.load_mcap_run)


def test_exporter_mcap_is_importable_on_base_install() -> None:
    """``embodied_sync.exporters.mcap`` imports without the mcap extra."""
    from embodied_sync.exporters import mcap as exporter

    assert callable(exporter.save_mcap_run)
    assert callable(exporter.save_mcap_episode)
