"""Sync-quality report generator (D-0023).

Covers ``build_report`` statistics, HTML rendering, and JSON summary
export. Rendering is asserted at the HTML-shape level (tags, escaping,
inlined style) rather than pixel-perfect output.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from embodied_sync.align import align_run
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    apply_profile,
)
from embodied_sync.reports import (
    REPORT_FORMAT_VERSION,
    SyncQualityReport,
    build_report,
    render_html,
    report_summary_dict,
    save_report_html,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def _clean_aligned():
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    return align_run(run, target_rate_hz=10.0)


def _corrupted_aligned_with_ground_truth():
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    profile = CorruptionProfile(
        seed=0,
        corruptions=(DroppedFramesCorruption(stream="cam_front", probability=0.7),),
    )
    corr = apply_profile(run, profile)
    return align_run(corr.run, target_rate_hz=10.0, ground_truth=corr.dropped)


class TestBuildReport:
    def test_frame_count_matches_input(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        assert isinstance(report, SyncQualityReport)
        assert report.frame_count == len(aligned.frames)

    def test_stream_stats_cover_every_stream(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        stream_names = {s.name for s in report.streams}
        assert stream_names == set(aligned.frames[0].samples.keys())

    def test_clean_run_regular_streams_have_zero_missing_and_full_confidence(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        # Regular streams have their samples on multiples of round(1e9/rate),
        # which line up exactly with 10 Hz grid multiples (0, 100M, ...);
        # median skew / |skew| are 0 and confidence 1. The events stream
        # is Poisson-like, so it never lands on the grid — assert its
        # metric shape instead of specific values.
        regular = {"cam_front", "cam_wrist", "robot_state", "tactile", "audio", "actions"}
        for stats in report.streams:
            assert stats.method == "nearest_neighbor"
            assert stats.ground_truth_missing_count == 0
            if stats.name in regular:
                assert stats.missing_count == 0
                assert stats.missing_rate == 0.0
                assert stats.median_skew_ns == 0
                assert stats.median_abs_skew_ns == 0
                assert stats.median_confidence == 1.0
            else:
                # Events stream: irregular, so median skew is non-zero,
                # but the values are still bounded and confidence in [0, 1].
                assert 0.0 <= stats.missing_rate <= 1.0
                if stats.median_confidence is not None:
                    assert 0.0 <= stats.median_confidence <= 1.0

    def test_missing_rate_matches_ratio(self) -> None:
        aligned = _corrupted_aligned_with_ground_truth()
        report = build_report(aligned)
        for stats in report.streams:
            assert stats.missing_rate == (
                stats.missing_count / stats.frame_count
                if stats.frame_count
                else 0.0
            )
        # cam_front should have some missing frames after 70% drops.
        cam = next(s for s in report.streams if s.name == "cam_front")
        assert cam.missing_count > 0
        assert cam.ground_truth_missing_count > 0

    def test_median_skew_is_none_when_every_frame_missing(self) -> None:
        run = generate_synthetic_run(duration_s=1.0, seed=0)
        # Wipe cam_front entirely so every aligned frame reports it missing.
        run["cam_front"] = []
        aligned = align_run(run, target_rate_hz=10.0)
        report = build_report(aligned)
        cam = next(s for s in report.streams if s.name == "cam_front")
        assert cam.missing_count == cam.frame_count
        assert cam.median_skew_ns is None
        assert cam.median_abs_skew_ns is None
        assert cam.median_confidence is None

    def test_medians_match_statistics_median(self) -> None:
        # Compute expected medians directly and compare.
        aligned = _corrupted_aligned_with_ground_truth()
        report = build_report(aligned)
        for stats in report.streams:
            skews = [
                frame.metadata[stats.name].skew_ns
                for frame in aligned.frames
                if not frame.metadata[stats.name].missing
                and frame.metadata[stats.name].skew_ns is not None
            ]
            if not skews:
                assert stats.median_skew_ns is None
            else:
                assert stats.median_skew_ns == int(statistics.median(skews))
                assert stats.median_abs_skew_ns == int(
                    statistics.median([abs(x) for x in skews])
                )

    def test_median_skew_matches_alignment_report_median_skew(self) -> None:
        # Two independent code paths compute median_skew_ns:
        # - engine._median_skew_ns_by_stream populates
        #   AlignmentReport.median_skew_ns at align_run time,
        # - reports.sync_quality.build_report recomputes it from the
        #   frames' metadata via statistics.median.
        # They must agree per stream on every run; this test guards
        # against a silent divergence between the two paths.
        aligned = _corrupted_aligned_with_ground_truth()
        report = build_report(aligned)
        assert set(aligned.report.median_skew_ns.keys()) == {
            s.name for s in report.streams
        }
        for stats in report.streams:
            assert stats.median_skew_ns == aligned.report.median_skew_ns[stats.name]


class TestReportSummaryDict:
    def test_summary_dict_shape_and_version(self) -> None:
        aligned = _corrupted_aligned_with_ground_truth()
        report = build_report(aligned)
        summary = report_summary_dict(report)
        assert summary["format_version"] == REPORT_FORMAT_VERSION
        assert summary["frame_count"] == report.frame_count
        assert len(summary["streams"]) == len(report.streams)
        expected_keys = {
            "name",
            "frame_count",
            "missing_count",
            "missing_rate",
            "median_skew_ns",
            "median_abs_skew_ns",
            "median_confidence",
            "method",
            "ground_truth_missing_count",
        }
        for entry in summary["streams"]:
            assert set(entry.keys()) == expected_keys


class TestHtmlRendering:
    def test_render_html_is_self_contained(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        html = render_html(report, title="Test")
        # No external assets: no href, no src, no url(...) rules.
        assert "http://" not in html
        assert "https://" not in html
        assert " src=" not in html
        assert " href=" not in html
        assert "url(" not in html
        assert "<style>" in html
        assert "<table>" in html
        assert "Test" in html

    def test_render_includes_stream_names_escaped(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        html = render_html(report)
        for stats in report.streams:
            assert stats.name in html

    def test_render_escapes_html_in_source_run(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        html = render_html(
            report,
            source_run="<script>alert(1)</script>",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_render_shows_ground_truth_column_when_populated(self) -> None:
        aligned = _corrupted_aligned_with_ground_truth()
        report = build_report(aligned)
        html = render_html(report)
        assert "Ground truth drops" in html

    def test_render_omits_ground_truth_column_when_empty(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        html = render_html(report)
        assert "Ground truth drops" not in html

    def test_render_shows_target_rate_when_supplied(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        html = render_html(report, target_rate_hz=10.0)
        assert "Target rate" in html
        assert "10" in html

    def test_render_omits_summary_bits_that_are_absent(self) -> None:
        aligned = _clean_aligned()
        report = build_report(aligned)
        html = render_html(report)  # no source_run, no target_rate
        assert "Source run" not in html
        assert "Target rate" not in html


class TestSaveReportHtml:
    def test_writes_html_file(self, tmp_path: Path) -> None:
        aligned = _clean_aligned()
        out = tmp_path / "report.html"
        path = save_report_html(aligned, out, target_rate_hz=10.0)
        assert path == out
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")
        assert "cam_front" in text

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        aligned = _clean_aligned()
        out = tmp_path / "nested" / "dir" / "report.html"
        save_report_html(aligned, out)
        assert out.exists()
