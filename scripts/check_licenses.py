#!/usr/bin/env python3
"""Fail on a copyleft-licensed dependency anywhere in the install (D-0042).

`embodied-sync` is MIT and stays that way transitively: nothing in the base
install or any optional extra may pull in a copyleft package, because that
would put a stricter license on part of the dependency graph than the
project itself claims. The concrete incident that prompted this: `ffsubsync`
is MIT but hard-depends on `auditok`, which is GPL-3.0 — a package can look
clean from its own classifier and still be a GPL install by the time pip
finishes resolving it.

Two modes, run at different points in the lifecycle:

The default mode (no flags; what CI runs) inspects every distribution
actually installed in the current interpreter via `importlib.metadata`. Since pip
already resolved the full transitive graph to populate the environment, this
mode requires no dependency-graph walking of our own and catches everything,
direct or transitive. Run it after installing the widest extra set
(``.[full,dev]``) so every optional integration is covered, not just the
base install.

``--pyproject`` (what the pre-merge-commit hook runs) reads
`pyproject.toml`'s dependency lists directly and queries PyPI's JSON API for
each named package's declared license, without installing anything. It only
sees one level of the graph — good enough to block an obviously-copyleft
package the moment someone adds it to `pyproject.toml`, before the merge
that would carry it into history, cheaply enough to run on every merge.
It is not a substitute for ``--installed``: a clean top-level package can
still drag in a copyleft transitive dependency, which only the CI job
`pip install`s far enough to see.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Substrings matched case-insensitively against license classifiers, the
# `License` / `License-Expression` metadata fields, and PyPI's `info.license`
# string. Deliberately targets copyleft families (must-relicense-derivatives
# licenses), not merely non-permissive ones: LGPL/MPL are file-level
# weak-copyleft and are excluded because they do not force this project's
# license to change to redistribute it, only require sharing modifications
# to the covered files themselves.
# Phrase markers deliberately exclude anything spelled "gpl": that is
# _GPL_WORD_RE's job, word-bounded, below. A plain substring marker for
# "gpl-3" or "gplv2" would also match inside "LGPLv2" (LGPL is weak
# copyleft and intentionally allowed), which is exactly the false positive
# a naive substring scan produces.
COPYLEFT_MARKERS = [
    "gnu general public license",
    "gnu affero general public license",
    "agpl",
    "server side public license",
    "sspl",
    "cecill",
    "eupl",
    "european union public licence",
    "reciprocal public license",
    "open software license",
    "sleepycat",
]
# Match "gpl" only as a whole word/segment, not preceded by an alnum char --
# so "gpl-3.0" and "GPLv3" hit, while "LGPLv2" (the "l" right before "gpl"
# fails the lookbehind) and unrelated words like "google" or a package
# literally named "gplearn" do not.
_GPL_WORD_RE = re.compile(r"(?<![a-z0-9])gpl(?![a-z0-9])", re.IGNORECASE)

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def is_copyleft(license_text: str) -> bool:
    text = license_text.lower()
    if _GPL_WORD_RE.search(text):
        return True
    return any(marker in text for marker in COPYLEFT_MARKERS)


# Packages routinely paste an entire third-party license (e.g. a bundled
# font or a permissively-relicensed dependency's history) into the free-text
# `License` field, which can mention "GPL" in passing thousands of
# characters in; matplotlib's does, quoting FreeType's GPL-or-FTL choice.
# `License-Expression` and `Classifier` are always short and structured, so
# only the free-text field is capped -- a genuine top-declared GPL license
# text always opens with its name, so a short prefix still catches it.
_FREE_TEXT_LICENSE_PREFIX = 300


def _license_signal(license_field: str, license_expression: str, classifiers: list[str]) -> str:
    fields = [license_field[:_FREE_TEXT_LICENSE_PREFIX], license_expression]
    fields += classifiers
    return " | ".join(f for f in fields if f)


def check_installed() -> list[tuple[str, str]]:
    """Scan every distribution installed in the current interpreter."""
    import importlib.metadata as md

    violations: list[tuple[str, str]] = []
    for dist in md.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or "<unknown>"
        if name.lower() == "embodied-sync":
            continue  # our own MIT header, not a dependency
        classifiers = [c for c in dist.metadata.get_all("Classifier") or [] if "License" in c]
        combined = _license_signal(
            dist.metadata.get("License") or "",
            dist.metadata.get("License-Expression") or "",
            classifiers,
        )
        if combined and is_copyleft(combined):
            violations.append((name, combined))
    return violations


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requirement_name(spec: str) -> str:
    """`"pyarrow>=14.0; python_version<'3.11'"` -> `"pyarrow"`."""
    spec = spec.split(";", 1)[0]  # drop environment markers
    match = _REQ_NAME_RE.match(spec)
    return match.group(1) if match else spec.strip()


def _iter_pyproject_dependency_names() -> list[str]:
    """Extract bare package names from `project.dependencies` and every
    `project.optional-dependencies` extra, skipping self-references like
    ``embodied-sync[mcap]`` which name this project, not a third party."""
    import tomllib

    project = tomllib.loads(PYPROJECT_PATH.read_text())["project"]
    specs = list(project.get("dependencies", []))
    for extra_specs in project.get("optional-dependencies", {}).values():
        specs.extend(extra_specs)

    names = {_requirement_name(spec) for spec in specs}
    return sorted(name for name in names if not name.lower().startswith("embodied-sync"))


def check_pyproject() -> list[tuple[str, str]]:
    """Query PyPI for each direct dependency's declared license."""
    violations: list[tuple[str, str]] = []
    for name in _iter_pyproject_dependency_names():
        url = f"https://pypi.org/pypi/{name}/json"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
        except (
            urllib.error.URLError,
            OSError,
            http.client.HTTPException,  # e.g. IncompleteRead -- not an OSError
            json.JSONDecodeError,
        ) as exc:
            print(f"warning: could not fetch license for {name!r} ({exc}); skipping", file=sys.stderr)
            continue
        info = data.get("info", {})
        classifiers = [c for c in info.get("classifiers", []) if "License" in c]
        combined = _license_signal(
            info.get("license") or "",
            info.get("license_expression") or "",
            classifiers,
        )
        if combined and is_copyleft(combined):
            violations.append((name, combined))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        action="store_true",
        help="Check pyproject.toml's direct dependencies against PyPI metadata "
        "instead of the installed environment (fast, no install, top-level only).",
    )
    args = parser.parse_args()

    violations = check_pyproject() if args.pyproject else check_installed()

    if violations:
        print("Copyleft dependency check FAILED:", file=sys.stderr)
        for name, license_text in violations:
            print(f"  - {name}: {license_text}", file=sys.stderr)
        print(
            "\nembodied-sync is MIT; no base install or extra may pull in a "
            "copyleft package (see DECISIONS.md D-0042).",
            file=sys.stderr,
        )
        return 1

    mode = "pyproject (direct deps only)" if args.pyproject else "installed (transitive)"
    print(f"Copyleft dependency check passed [{mode}].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
