"""
standardize_faces.py — detect the face, crop around it, grayscale, resize.
Uses mediapipe's Tasks FaceDetector (works headless; no OpenCV / libGL).

Why this exists: candidate photos are NOT all centered, so a center crop fails.
This detects where the face actually is and crops around it wherever it sits.

Pipeline per image:
  1. mediapipe FaceDetector finds the face bounding box,
  2. expand the box by a margin and make it square,
  3. crop, optionally mask outside an oval (kills residual background/flags),
  4. grayscale (matches the retrospective B&W stimuli; removes colour cues),
  5. resize to a fixed square.

Faces that aren't detected (or multiple faces) are FLAGGED, not silently
mis-cropped — you review those and either hand-crop or pass a manual box.

ONE-TIME SETUP (downloads the ~230 KB detector model):
    import urllib.request, pathlib
    url = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
           "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
    pathlib.Path("models").mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, "models/blaze_face_short_range.tflite")

Requires: mediapipe, Pillow, numpy(<2)  — mediapipe already installed.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import RunningMode


MODEL_PATH = "models/blaze_face_short_range.tflite"

# --- tunable params (LOCK IN THE PREREGISTRATION) ---
OUT_SIZE = 224
CROP_MARGIN = 0.6        # expand the detected face box by this fraction on each
                         # side. face detectors return a TIGHT box (just eyes-to-
                         # chin); ~0.6 adds forehead/hair/chin for a natural crop.
MASK_BACKGROUND = False  # True -> grey outside an oval (removes background/flags)
GRAYSCALE = True
MIN_CONFIDENCE = 0.5


def _make_detector(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run the one-time download in the "
            "module docstring first.")
    opts = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=MIN_CONFIDENCE)
    return vision.FaceDetector.create_from_options(opts)


def _largest_face_box(detection_result, img_w, img_h):
    """Return (x, y, w, h) of the largest detected face, or None."""
    if not detection_result.detections:
        return None, 0
    boxes = []
    for d in detection_result.detections:
        bb = d.bounding_box
        boxes.append((bb.origin_x, bb.origin_y, bb.width, bb.height))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[0], len(boxes)


def _square_box(x, y, w, h, img_w, img_h, margin):
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    left = int(max(0, cx - half)); top = int(max(0, cy - half))
    right = int(min(img_w, cx + half)); bottom = int(min(img_h, cy + half))
    return left, top, right, bottom


def _oval_mask(size):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    px, py = int(size * 0.10), int(size * 0.04)
    d.ellipse([px, py, size - px, size - py], fill=255)
    return m


def standardize_one(path, out_path, detector, size=OUT_SIZE, margin=CROP_MARGIN,
                    grayscale=GRAYSCALE, mask_background=MASK_BACKGROUND,
                    manual_box=None):
    """Standardize one image. Returns 'ok' / 'multi_face' / 'no_face' / 'error'.

    manual_box: optional (left, top, right, bottom) to override detection for a
    specific hard image (you read it off the raw photo by eye).
    """
    try:
        pil = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        img_w, img_h = pil.size

        if manual_box is not None:
            box, status = manual_box, "ok"
        else:
            arr = np.asarray(pil)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            result = detector.detect(mp_img)
            face, n = _largest_face_box(result, img_w, img_h)
            if face is None:
                return "no_face"                    # flag; do NOT guess-crop
            status = "multi_face" if n > 1 else "ok"
            box = _square_box(*face, img_w, img_h, margin)

        face_img = pil.crop(box).resize((size, size), Image.LANCZOS)
        if grayscale:
            face_img = ImageOps.grayscale(face_img)
        if mask_background:
            base = Image.new("L" if grayscale else "RGB", (size, size),
                             128 if grayscale else (128, 128, 128))
            base.paste(face_img, (0, 0), _oval_mask(size))
            face_img = base

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        face_img.save(out_path)
        return status
    except Exception as e:
        return f"error: {e}"


def standardize_dir(in_dir, out_dir, size=OUT_SIZE, margin=CROP_MARGIN,
                    grayscale=GRAYSCALE, mask_background=MASK_BACKGROUND,
                    manual_boxes=None, model_path=MODEL_PATH):
    """Standardize every image in in_dir -> out_dir.

    manual_boxes: optional {filename: (l,t,r,b)} overrides for hard images.
    Returns a report list; flags no_face / multi_face for your review.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    manual_boxes = manual_boxes or {}
    detector = _make_detector(model_path)
    exts = {".jpg", ".jpeg", ".png"}
    report = []
    for p in sorted(in_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        status = standardize_one(
            p, out_dir / f"{p.stem}.png", detector, size=size, margin=margin,
            grayscale=grayscale, mask_background=mask_background,
            manual_box=manual_boxes.get(p.name))
        report.append({"file": p.name, "status": status})
        print(f"  {p.name:40s} -> {status}")

    ok = sum(1 for r in report if r["status"] in ("ok", "multi_face"))
    flagged = [r for r in report if r["status"] not in ("ok", "multi_face")]
    print(f"\n{ok}/{len(report)} cropped. {len(flagged)} need a look:")
    for r in flagged:
        print(f"  {r['file']}: {r['status']}")
    return report