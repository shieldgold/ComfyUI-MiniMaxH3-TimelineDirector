"""Regression guards for the PSY operational coordinate and visual QC contract."""
import importlib.util
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

MODULE = Path(__file__).resolve().parents[2] / "ops" / "wan_psy_composite.py"
SPEC = importlib.util.spec_from_file_location("wan_psy_composite", MODULE)
wan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wan)


def b_assets():
    sx, sy = 766 / 1252, 432 / 706
    return {"source_size": [1920, 1080], "crop_xywh": [458, 374, 1252, 706],
            "canvas": [768, 432], "fit_size": [766, 432], "pad_xy": [1, 0],
            "affine_source_to_canvas": [[sx, 0, 1-458*sx], [0, sy, -374*sy]],
            "valid_frames": 44, "encoded_frames": 45, "fps": 24,
            "source_frame_indices": list(range(44))+[43]}


class GeometryTests(unittest.TestCase):
    def test_b_xywh_corners_roundtrip_with_one_pixel_letterbox(self):
        asset = b_assets()
        plan = wan.validate_manifest(asset)
        transform, inverse = asset["affine_source_to_canvas"], plan["inverse_affine"]
        for (x, y), expected in [((458, 374), (1, 0)), ((1710, 1080), (767, 432))]:
            u = transform[0][0]*x+transform[0][2]
            v = transform[1][1]*y+transform[1][2]
            self.assertAlmostEqual(u, expected[0])
            self.assertAlmostEqual(v, expected[1])
            self.assertAlmostEqual(inverse[0][0]*u+inverse[0][2], x)
            self.assertAlmostEqual(inverse[1][1]*v+inverse[1][2], y)

    def test_wrong_crop_interpretation_is_rejected_against_actual_affine(self):
        asset = b_assets()
        asset["crop_xywh"] = [458, 374, 1252-458, 706-374]
        with self.assertRaisesRegex(ValueError, "affine disagrees"):
            wan.validate_manifest(asset)

    def test_padding_may_not_replay_an_earlier_frame(self):
        asset = b_assets()
        asset["source_frame_indices"][-1] = 42
        with self.assertRaisesRegex(ValueError, "padding"):
            wan.validate_manifest(asset)

    def test_source_mapping_may_not_cross_or_skip_the_cut(self):
        asset = b_assets()
        asset["source_frame_indices"][20] = 21
        with self.assertRaisesRegex(ValueError, "consecutive"):
            wan.validate_manifest(asset)

    def test_manifest_and_affine_must_agree_on_padding(self):
        asset = b_assets()
        asset["pad_xy"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "affine disagrees"):
            wan.validate_manifest(asset)


class QualityGateTests(unittest.TestCase):
    def test_failed_raw_is_rejected_before_any_video_output(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, output = Path(folder)/"bad.mp4", Path(folder)/"final.mp4"
            with self.assertRaisesRegex(ValueError, "failed"):
                wan.compose({}, raw, {"visual_quality": "FAIL", "raw_server_file": str(raw)},
                            output, Path(folder)/"missing-model.pt")
            self.assertFalse(output.exists())

    def test_passing_qc_from_another_video_cannot_unlock_composition(self):
        with self.assertRaisesRegex(ValueError, "different raw"):
            wan.require_visual_pass({"visual_quality": "PASS", "raw_server_file": "/tmp/old.mp4"},
                                    Path("/tmp/new.mp4"))

    def test_passing_qc_must_identify_the_exact_raw(self):
        wan.require_visual_pass({"visual_quality": "PASS", "raw_server_file": "/tmp/new.mp4"},
                                Path("/tmp/new.mp4"))

    def test_explicit_structural_pass_allows_only_pending_inspection_record(self):
        report = {"visual_quality": "REVIEW_REQUIRED", "structural_quality": "PASS",
                  "compose_review_allowed": True, "final_quality": "PENDING",
                  "raw_server_file": "/tmp/new.mp4"}
        wan.require_visual_pass(report, Path("/tmp/new.mp4"))
        self.assertEqual(report["visual_quality"], "REVIEW_REQUIRED")
        self.assertEqual(report["final_quality"], "PENDING")

    def test_review_permission_without_structural_validation_is_rejected(self):
        report = {"visual_quality": "REVIEW_REQUIRED", "compose_review_allowed": True,
                  "final_quality": "PENDING", "raw_server_file": "/tmp/new.mp4"}
        with self.assertRaisesRegex(ValueError, "failed"):
            wan.require_visual_pass(report, Path("/tmp/new.mp4"))

    def test_structural_review_permission_is_still_bound_to_exact_raw(self):
        report = {"visual_quality": "REVIEW_REQUIRED", "structural_quality": "PASS",
                  "compose_review_allowed": True, "final_quality": "PENDING",
                  "raw_server_file": "/tmp/old.mp4"}
        with self.assertRaisesRegex(ValueError, "different raw"):
            wan.require_visual_pass(report, Path("/tmp/new.mp4"))


class AudioTests(unittest.TestCase):
    def test_explicit_wav_and_cli_priority_over_video_source(self):
        asset = {"source": "/tmp/silent.mp4", "audio_source": "/tmp/original.wav"}
        self.assertEqual(wan.audio_path(asset), (Path("/tmp/original.wav"), "asset_audio_source"))
        self.assertEqual(wan.audio_path(asset, Path("/tmp/override.wav")),
                         (Path("/tmp/override.wav"), "cli_override"))

    def test_missing_video_audio_reports_explicit_wav_remedy(self):
        with patch.object(wan.subprocess, "run", return_value=SimpleNamespace(stdout='{"streams":[]}')):
            with self.assertRaisesRegex(ValueError, "supply --audio-source"):
                wan.probe_audio(Path("/tmp/silent.mp4"))

    def test_c2_uses_original_frame83_not_local_frame77_or_double_offset(self):
        info = {"sample_rate": 48000, "channels": 2, "duration": 5.875, "codec": "pcm_s16le"}
        with patch.object(wan, "probe_audio", return_value=info):
            audio = wan.audio_plan({"source": "/tmp/C.mp4", "audio_source": "/tmp/C_audio.wav"},
                                   {"source_first_frame": 83, "valid_frames": 48, "fps": 24})
        self.assertEqual(audio["start_seconds"], 83/24)
        self.assertEqual(audio["duration_seconds"], 2.)
        self.assertEqual(audio["expected_pcm_frames"], 96000)

    def test_short_wav_cannot_silently_truncate_video_with_shortest(self):
        info = {"sample_rate": 48000, "channels": 2, "duration": 4., "codec": "pcm_s16le"}
        with patch.object(wan, "probe_audio", return_value=info):
            with self.assertRaisesRegex(ValueError, "audio ends before"):
                wan.audio_plan({"source": "/tmp/C.mp4"},
                               {"source_first_frame": 83, "valid_frames": 48, "fps": 24})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required for real PCM regression")
    def test_actual_wav_cut_starts_at_exact_source_frame_and_preserves_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            original, cut = Path(folder)/"original.wav", Path(folder)/"cut.wav"
            samples = b"".join(struct.pack("<hh", (i*37)%30000-15000, (i*53)%30000-15000)
                               for i in range(12000))
            with wave.open(str(original), "wb") as stream:
                stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(48000)
                stream.writeframes(samples)
            audio = wan.audio_plan({"source": "/tmp/silent.mp4", "audio_source": str(original)},
                                   {"source_first_frame": 2, "valid_frames": 3, "fps": 24})
            actual = wan.extract_audio(audio, cut)
            self.assertEqual(actual["pcm_frames"], 6000)
            with wave.open(str(cut), "rb") as stream:
                self.assertEqual(stream.readframes(stream.getnframes()), samples[4000*4:10000*4])


if __name__ == "__main__":
    unittest.main()
