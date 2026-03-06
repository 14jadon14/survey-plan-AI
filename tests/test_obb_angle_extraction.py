"""
test_obb_angle_extraction.py

Verifies that OBBUltralyticsDetectionModel correctly populates
ObjectPrediction.extra_data with a non-zero angle when processing a real
OBB model against a test image.

Run from the repo root:
    python -m pytest tests/test_obb_angle_extraction.py -v

Requirements: SAHI, ultralytics, opencv-python, numpy must be installed.
The test uses the pre-trained model at yolo_training/yolo26l-obb.pt and
the first available image in yolo_training/data or yolo_training/custom_data.
"""

import os
import sys
import glob
import json
import tempfile

import pytest

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_SRC = os.path.join(ROOT, "yolo_training", "src")
if YOLO_SRC not in sys.path:
    sys.path.insert(0, YOLO_SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_test_image():
    """Return the first image found in the project's data directories."""
    search_roots = [
        os.path.join(ROOT, "yolo_training", "data"),
        os.path.join(ROOT, "yolo_training", "custom_data"),
        os.path.join(ROOT, "yolo_training", "survey_obb_dataset"),
    ]
    exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    for base in search_roots:
        for ext in exts:
            matches = glob.glob(os.path.join(base, "**", ext), recursive=True)
            if matches:
                return matches[0]
    return None


def find_model():
    """Return path to best.pt training output, or the bundled yolo26l-obb.pt."""
    candidates = [
        os.path.join(ROOT, "yolo_training", "runs", "train", "weights", "best.pt"),
        os.path.join(ROOT, "yolo_training", "yolo26l-obb.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(find_model() is None, reason="No .pt model found in yolo_training/")
@pytest.mark.skipif(find_test_image() is None, reason="No test image found in yolo_training/data or custom_data")
def test_obb_angle_populated_in_extra_data():
    """
    OBBUltralyticsDetectionModel should populate ObjectPrediction.extra_data
    with 'angle', 'rect_w', 'rect_h' for each detection on an OBB model.
    """
    try:
        import torch
        from detection.inference import OBBUltralyticsDetectionModel, run_inference
    except ImportError as e:
        pytest.skip(f"Required package not available: {e}")

    model_path = find_model()
    image_path = find_test_image()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\n  [TEST] model: {os.path.basename(model_path)}")
    print(f"  [TEST] image: {os.path.basename(image_path)}")
    print(f"  [TEST] device: {device}")

    # Instantiate the custom model directly
    detection_model = OBBUltralyticsDetectionModel(
        model_path=model_path,
        confidence_threshold=0.25,  # low threshold to maximise detections on test image
        device=device,
    )
    detection_model.load_model()

    # Run on the test image (no SAHI slicing needed for this unit test)
    from PIL import Image
    import numpy as np
    img = np.array(Image.open(image_path).convert("RGB"))

    detection_model.perform_inference(img)
    detection_model._create_object_prediction_list_from_original_predictions(
        shift_amount_list=[[0, 0]],
        full_shape_list=[[img.shape[0], img.shape[1]]],
    )

    predictions = detection_model.object_prediction_list

    if len(predictions) == 0:
        pytest.skip("No detections on the test image — lower conf threshold or use a different image.")

    print(f"  [TEST] {len(predictions)} detections found.")

    angles_found = []
    for pred in predictions:
        assert hasattr(pred, "extra_data"), "ObjectPrediction must have extra_data attribute"
        if pred.extra_data:
            assert "angle" in pred.extra_data, "extra_data must contain 'angle'"
            assert "rect_w" in pred.extra_data, "extra_data must contain 'rect_w'"
            assert "rect_h" in pred.extra_data, "extra_data must contain 'rect_h'"
            angles_found.append(pred.extra_data["angle"])
            print(f"    bbox={[round(x) for x in pred.bbox.to_xyxy()]}  "
                  f"angle={pred.extra_data['angle']:.1f}  "
                  f"rect_w={pred.extra_data['rect_w']:.1f}  "
                  f"rect_h={pred.extra_data['rect_h']:.1f}")

    # For an OBB model, at least SOME detections should have a non-zero angle
    # (even a very small rotation counts — 0.0 exactly is extremely unlikely for real text)
    assert len(angles_found) > 0, "No predictions had extra_data populated — subclass may not be working"
    non_zero = [a for a in angles_found if a != 0.0]
    assert len(non_zero) > 0, (
        f"All {len(angles_found)} angles were exactly 0.0 — OBB subclass is not extracting angle data. "
        f"Angles: {angles_found}"
    )
    print(f"\n  [PASS] {len(non_zero)}/{len(angles_found)} detections have a non-zero OBB angle.")


@pytest.mark.skipif(find_model() is None, reason="No .pt model found in yolo_training/")
@pytest.mark.skipif(find_test_image() is None, reason="No test image found in yolo_training/data or custom_data")
def test_run_inference_json_has_angles():
    """
    run_inference() should write a JSON file where at least one entry has a
    non-zero angle, confirming the full end-to-end pipeline fix.
    """
    try:
        import torch
        from detection.inference import run_inference
    except ImportError as e:
        pytest.skip(f"Required package not available: {e}")

    model_path = find_model()
    image_path = find_test_image()

    with tempfile.TemporaryDirectory() as tmp:
        json_output = os.path.join(tmp, "crop_parameters.json")
        run_inference(
            model_path=model_path,
            source=image_path,
            output_dir=tmp,
            conf_thres=0.25,
            json_output=json_output,
        )

        assert os.path.exists(json_output), (
            "run_inference() did not create the JSON file — check that json_output path is respected."
        )

        with open(json_output) as f:
            detections = json.load(f)

        assert len(detections) > 0, "JSON is empty — no detections on test image."

        angles = [d.get("angle", 0) for d in detections]
        non_zero = [a for a in angles if a != 0.0]
        print(f"\n  [TEST] {len(detections)} detections in JSON. Non-zero angles: {len(non_zero)}/{len(angles)}")
        for d in detections[:5]:
            print(f"    {d.get('label')} | angle={d.get('angle'):.2f} | score={d.get('score'):.2f}")

        assert len(non_zero) > 0, (
            f"All angles in JSON are 0.0 — the OBB subclass angle extraction is not working.\n"
            f"Sample detections: {detections[:3]}"
        )
        print(f"\n  [PASS] JSON contains non-zero OBB angles.")
