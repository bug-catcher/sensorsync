"""Reading a waveform off disk without adding a dependency (A5).

``calibrate/`` is numpy-only by policy (D-0038), which rules out
``soundfile``, ``librosa`` and ``scipy.io.wavfile`` — the three obvious
answers to "load this audio file". What remains is enough: Python's
stdlib :mod:`wave` module reads uncompressed PCM WAV, which is what a
calibration recording is, and :func:`numpy.load` reads the ``.npy``
arrays that anyone doing this from a notebook already has.

This module exists so the CLI stays a thin shell (A5's actual
requirement) rather than growing file parsing of its own, and so a
library caller gets the same loader the CLI uses.

Deliberately not supported: compressed formats. A calibration signal is
being measured to sub-millisecond precision, and a codec that
reconstructs the waveform "perceptually identically" is free to smear a
transient by exactly the amount being measured. Decode to PCM first,
with a tool that says what it did.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["SUPPORTED_SUFFIXES", "load_waveform"]

#: File suffixes :func:`load_waveform` understands.
SUPPORTED_SUFFIXES = (".wav", ".npy")

#: Sample width (bytes) → (numpy dtype, zero level, full-scale divisor).
#: WAV stores 8-bit as *unsigned* and everything wider as signed two's
#: complement — a quirk of the format, not of this code.
_PCM_FORMATS: dict[int, tuple[str, float, float]] = {
    1: ("u1", 128.0, 128.0),
    2: ("<i2", 0.0, 32768.0),
    4: ("<i4", 0.0, 2147483648.0),
}


def _load_wav(path: Path) -> tuple[NDArray[np.float64], float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = float(handle.getframerate())
        frames = handle.readframes(handle.getnframes())
    spec = _PCM_FORMATS.get(width)
    if spec is None:
        raise ValueError(
            f"{path}: {width * 8}-bit PCM is not supported (supported widths: "
            f"{sorted(bits * 8 for bits in _PCM_FORMATS)} bits). Convert to "
            f"16-bit PCM WAV, or pass a .npy array with --sample-rate-hz."
        )
    dtype, zero, full_scale = spec
    raw = np.frombuffer(frames, dtype=np.dtype(dtype)).astype(np.float64)
    samples = (raw - zero) / full_scale
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, rate


def load_waveform(
    path: str | Path, *, sample_rate_hz: float | None = None
) -> tuple[NDArray[np.float64], float]:
    """Load ``path`` as ``(samples, sample_rate_hz)``.

    ``.wav`` carries its own sample rate; passing ``sample_rate_hz`` for
    one is an error rather than an override, because silently believing
    a caller over the file header is how a calibration ends up scaled by
    48/44.1 and nobody notices. ``.npy`` carries no rate, so one is
    required.

    Samples come back as float64 in roughly ``[-1, 1)`` for WAV
    (integer PCM divided by full scale) and verbatim for ``.npy``. The
    absolute scale is irrelevant to everything downstream —
    :func:`~embodied_sync.calibrate.clap.detect_audio_onsets` works on
    log-energy *differences* and is gain-invariant by construction — so
    no normalisation is applied beyond that.

    Multi-channel WAV is returned as ``(frames, channels)``, the shape
    the detector already downmixes.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"no such audio file: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".wav":
        if sample_rate_hz is not None:
            raise ValueError(
                f"{resolved} is a WAV file and carries its own sample rate; "
                f"drop sample_rate_hz rather than overriding the header"
            )
        return _load_wav(resolved)
    if suffix == ".npy":
        if sample_rate_hz is None:
            raise ValueError(
                f"{resolved} is a raw .npy array and carries no sample rate; "
                f"pass sample_rate_hz"
            )
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
        array = np.load(resolved)
        return np.asarray(array, dtype=np.float64), float(sample_rate_hz)
    raise ValueError(
        f"{resolved}: unsupported audio format {suffix!r}; supported: "
        f"{list(SUPPORTED_SUFFIXES)}. Compressed formats are excluded on "
        f"purpose — a codec may smear a transient by more than the offset "
        f"being measured."
    )
