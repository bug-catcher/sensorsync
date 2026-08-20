"""What a dataset has to supply before a person can judge its alignment.

The renderer in :mod:`embodied_sync.inspect.render` asks four questions
of a stream — *what did it see at this instant*, *what did it hear*,
*what did its other channels read*, and *could anyone tell these two
instants apart by looking* — and knows nothing else about it. Everything
dataset-shaped (which file, which codec, which camera serial, which
resampling the dataset authors already did) lives behind this interface,
in the caller's code, because those answers differ for every rig and
none of them belong in a library.

Why the frame call is batched
-----------------------------
:meth:`EvidenceProvider.frames_at` takes a *list* of instants and returns
one entry per instant. That shape exists because the real
implementations shell out to a decoder: source tiles may come from a single
``ffmpeg`` pass with a ``select`` filter over ~48 frame indices, and a
per-tile interface would re-open and re-scan a 60-second MP4 forty-odd
times to produce the same pixels. A provider that can only decode one
frame at a time still satisfies the interface by looping — the cost is
paid by the implementation that has it, not imposed on the one that
does not.

Why the provider measures visual separation
-------------------------------------------
"Are these two candidate frames distinguishable?" looks like a question
for the renderer, and it is not one the renderer can answer honestly.
Answering it means comparing pixels, and the pixels that matter are the
ones the *detector* saw, not necessarily the display-sized
JPEGs on the page — otherwise the number changes when someone passes a
different ``--frame-width``, and a page's verdict must not depend on its
layout. Decoding pixels also needs the optional media stack, which this
package deliberately does not have. So the provider answers it, or
returns ``None`` and the page says the comparison was not measured
rather than implying the frames agreed.

What ``None`` means, everywhere
-------------------------------
Absent, never zero and never substituted. A provider with no frame for
an instant returns ``None`` and the page renders "unavailable" in the
tile; the one thing it must not do is hand back the nearest frame from
some other second, or silence in place of missing audio. The failure
mode this whole tool exists to catch is a confident answer with nothing
behind it, and a plausible-looking fabricated tile is that failure with
a picture attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "AudioClip",
    "BaseProvider",
    "ChannelReading",
    "EvidenceProvider",
    "FrameImage",
    "NullProvider",
    "StreamInfo",
]


@dataclass(frozen=True, slots=True)
class FrameImage:
    """One decoded frame, ready to embed, with the instant it depicts.

    ``time_ns`` is the frame's *own* capture time, not the time that was
    asked for. The page prints both: a frame 200 ms away from the
    instant under test is weak evidence about that instant, and hiding
    the gap would present it as strong.
    """

    payload: bytes
    #: Image MIME type, e.g. ``image/jpeg``. Carried rather than assumed
    #: so a provider that has PNGs does not have to transcode them.
    mime: str
    time_ns: int
    #: Position in the stream, when the provider has a natural one. Shown
    #: on the page purely so a reader can go back to the source file.
    index: int | None = None


@dataclass(frozen=True, slots=True)
class AudioClip:
    """A short excerpt a browser can play, and how much of it was there."""

    payload: bytes
    start_time_ns: int
    duration_s: float
    mime: str = "audio/wav"
    #: True when the requested window ran past one end of the recording.
    #: Reported rather than padded — padding a clip with silence invents
    #: the one thing a listener is checking for.
    clipped: bool = False


@dataclass(frozen=True, slots=True)
class ChannelReading:
    """Named scalar channels sampled nearest one instant.

    Free-form names because the useful channels are rig-specific: force
    magnitude and gripper width on a manipulator, head angular speed on
    a headset. The renderer compares one designated channel across
    candidates (:attr:`StreamInfo.corroboration_channel`) and prints the
    rest as context.
    """

    time_ns: int
    #: ``time_ns − requested_ns``. Same reasoning as
    #: :attr:`FrameImage.time_ns`: a reading 400 ms from the instant is
    #: not a reading *at* the instant.
    age_ns: int
    values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """Identity and provenance of one stream, as the page states it."""

    label: str
    #: Where the media came from — a path, a URI, a rig name. Printed
    #: verbatim so a reader can reproduce the page from the source.
    source: str = ""
    #: Median sampling interval. Residuals are quoted as a fraction of
    #: it, because "8 ms out" means something different at 15 Hz than at
    #: 200 Hz. ``None`` when the stream has no natural frame clock.
    frame_period_ns: int | None = None
    #: Which entry of :attr:`ChannelReading.values` is compared across
    #: candidates. Choose the one that is physically tied to the event —
    #: contact force, not battery voltage.
    corroboration_channel: str | None = None
    #: A caveat printed next to those numbers. Both real bindings need
    #: one: a source may resample a channel onto a camera frame clock or
    #: index a pose stream by video frame, so in each case the channel
    #: corroborates against the dataset's own synchronization rather than
    #: against something independent of it. That belongs on the page.
    channel_note: str = ""
    #: Named resources that were expected and absent (an unfetched robot
    #: state file, a missing audio directory). Rendered as warnings: a
    #: page that silently omits a column teaches the reader there was no
    #: such channel, which is a different claim from "it was not there".
    missing: tuple[str, ...] = ()


@runtime_checkable
class EvidenceProvider(Protocol):
    """The whole contract between a dataset and the inspector.

    Structural on purpose — an implementation does not have to inherit
    from anything, which matters because the natural place to implement
    this is next to the dataset loader, not next to the renderer.
    :class:`BaseProvider` exists for implementers who want the optional
    half filled in for them.
    """

    def stream_info(self) -> StreamInfo:
        """Identity, provenance and channel metadata for this stream."""
        ...

    def frames_at(self, times_ns: Sequence[int]) -> Sequence[FrameImage | None]:
        """Frames nearest each instant, one entry per request, in order.

        Returning a different number of entries than were asked for is a
        contract violation the renderer refuses, because the alternative
        — zipping a short result against the request — silently mislabels
        every tile after the gap.
        """
        ...

    def audio_at(self, centre_ns: int) -> AudioClip | None:
        """An excerpt centred on an instant, or ``None`` if uncovered."""
        ...

    def channels_at(self, t_ns: int) -> ChannelReading | None:
        """Scalar channels nearest an instant, or ``None`` if unavailable."""
        ...

    def channel_noise(self, channel: str, t_ns: int) -> float | None:
        """Typical sample-to-sample change in ``channel`` around ``t_ns``.

        The noise floor under any claim that one candidate's reading
        matches the reference better than another's. Without it a
        resampling jitter of 0.2 units reads as a synchronization
        verdict; with it, differences that small are reported as a tie
        and the pictures decide instead. ``None`` disables the tie band.
        """
        ...

    def visual_separation(self, first_ns: int, second_ns: int) -> float | None:
        """How visibly the frames at two instants differ, in gray levels.

        0-255, measured however the provider measures motion — ideally on
        the same pixels its detector saw. ``None`` when not measured.
        """
        ...


class BaseProvider(ABC):
    """Implements the optional half of :class:`EvidenceProvider`.

    Frames are the only thing the page cannot do without, so those two
    methods stay abstract and every other answer defaults to "not
    available" — which the page renders as an explicit absence. A
    subclass overrides exactly the questions its dataset can answer,
    and a dataset with no microphone says nothing about audio rather
    than implementing a stub that returns silence.
    """

    @abstractmethod
    def stream_info(self) -> StreamInfo:
        """Identity, provenance and channel metadata for this stream."""

    @abstractmethod
    def frames_at(self, times_ns: Sequence[int]) -> Sequence[FrameImage | None]:
        """Frames nearest each instant, one entry per request, in order."""

    def audio_at(self, centre_ns: int) -> AudioClip | None:
        return None

    def channels_at(self, t_ns: int) -> ChannelReading | None:
        return None

    def channel_noise(self, channel: str, t_ns: int) -> float | None:
        return None

    def visual_separation(self, first_ns: int, second_ns: int) -> float | None:
        return None


class NullProvider(BaseProvider):
    """A stream with no media at all.

    Not a placeholder for testing: it is what the CLI uses when it is
    handed two event trains and nothing else. The resulting page still
    carries the part of the diagnosis that is arithmetic — the competing
    ±1 pairings, their offset deltas, the ambiguity verdict, the residual
    plot — and states plainly, in every tile, that no media backed it.
    A reader can then tell "the pictures disagreed" from "there were no
    pictures", which one banner at the top would not achieve.
    """

    __slots__ = ("_info",)

    def __init__(self, label: str, *, source: str = "") -> None:
        self._info = StreamInfo(label=label, source=source)

    def stream_info(self) -> StreamInfo:
        return self._info

    def frames_at(self, times_ns: Sequence[int]) -> Sequence[FrameImage | None]:
        return [None] * len(times_ns)
