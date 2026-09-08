"""tests/test_landmarking_pipeline.py
Synthetic tests for utils.landmarking_pipeline.place_landmarks -- the
in-memory counterpart to run_landmarking(), used by app/single_image.py.
Monkeypatches the four composed stage functions (each independently owned
by its own module) so no real image/detector/UNet model is needed here --
this is only about the ORCHESTRATION (which stage's failure short-circuits
the rest, what carries through on success), not about detection/landmark-
placement/renumbering correctness itself.
"""
from __future__ import annotations

import numpy as np

from landmarks.methods.base import NumberingResult
from utils import landmarking_pipeline as lp

DUMMY_BOX = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])


def _dummy_image():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def _place(monkeypatch=None):
    return lp.place_landmarks(
        _dummy_image(), detector_mode="light", detector_ctx={}, detector_args=None,
        padding=0.1, out_width=512, out_height=256, unet_model=None, unet_device="cpu",
        n_landmarks=19, reference_zones=np.zeros((19, 2)),
    )


def test_place_landmarks_ok(monkeypatch):
    monkeypatch.setattr(lp, "detect_one_image", lambda mode, ctx, image, args: {
        "status": "OK", "error_reason": "", "box": DUMMY_BOX,
    })
    crop = np.ones((256, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(lp, "normalize_one", lambda image, points, pad, out_width, out_height: (crop, 2.0))
    landmarks = np.zeros((19, 2))
    monkeypatch.setattr(lp, "predict_landmarks", lambda image, model, device, n_landmarks: ("OK", "", landmarks, 19))
    numbered = np.ones((19, 2))
    monkeypatch.setattr(lp, "numerate_one", lambda pts, zones, method="hungarian_umeyama": NumberingResult(
        numbered=numbered, status="OK", score=0.5,
    ))

    result = _place()

    assert result.status == "OK"
    assert result.stage is None
    assert np.array_equal(result.landmarks, numbered)
    assert result.registration_score == 0.5
    assert result.crop_image is crop


def test_place_landmarks_detection_failed(monkeypatch):
    monkeypatch.setattr(lp, "detect_one_image", lambda mode, ctx, image, args: {
        "status": "FAILED", "error_reason": "no_detection", "box": None,
    })

    result = _place()

    assert result.status == "FAILED"
    assert result.stage == "detection"
    assert result.error_reason == "no_detection"
    assert result.crop_image is None


def test_place_landmarks_crop_failed(monkeypatch):
    monkeypatch.setattr(lp, "detect_one_image", lambda mode, ctx, image, args: {
        "status": "OK", "error_reason": "", "box": DUMMY_BOX,
    })
    monkeypatch.setattr(lp, "normalize_one", lambda image, points, pad, out_width, out_height: (None, None))

    result = _place()

    assert result.status == "FAILED"
    assert result.stage == "crop"
    assert result.crop_image is None


def test_place_landmarks_landmarks_failed(monkeypatch):
    monkeypatch.setattr(lp, "detect_one_image", lambda mode, ctx, image, args: {
        "status": "OK", "error_reason": "", "box": DUMMY_BOX,
    })
    crop = np.ones((256, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(lp, "normalize_one", lambda image, points, pad, out_width, out_height: (crop, 2.0))
    monkeypatch.setattr(
        lp, "predict_landmarks",
        lambda image, model, device, n_landmarks: ("FAILED", "no_local_maximum", None, 0),
    )

    result = _place()

    assert result.status == "FAILED"
    assert result.stage == "landmarks"
    assert result.crop_image is crop


def test_place_landmarks_suspect_underdetection_fails_at_renumbering(monkeypatch):
    """A predict_landmarks SUSPECT (fewer points than expected) is not
    special-cased -- it must fall through into numerate_one and fail there
    on a point-count mismatch, per landmarks/renumber.py's own contract."""
    monkeypatch.setattr(lp, "detect_one_image", lambda mode, ctx, image, args: {
        "status": "OK", "error_reason": "", "box": DUMMY_BOX,
    })
    crop = np.ones((256, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(lp, "normalize_one", lambda image, points, pad, out_width, out_height: (crop, 2.0))
    landmarks = np.zeros((15, 2))
    monkeypatch.setattr(lp, "predict_landmarks", lambda image, model, device, n_landmarks: (
        "SUSPECT", "only_15_of_19_expected_maxima", landmarks, 15,
    ))
    monkeypatch.setattr(lp, "numerate_one", lambda pts, zones, method="hungarian_umeyama": NumberingResult(
        numbered=pts, status="FAILED", score=float("inf"), reason="15 landmarks, 19 expected",
    ))

    result = _place()

    assert result.status == "FAILED"
    assert result.stage == "renumbering"
    assert result.error_reason == "15 landmarks, 19 expected"
