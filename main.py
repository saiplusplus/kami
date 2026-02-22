"""
Real-time Hair Segmentation (Beard-robust) using ONLY MediaPipe Tasks APIs
- Hair segmentation: ImageSegmenter (hair_segmentation.tflite)  [expects RGBA]
- Face landmarks: FaceLandmarker (face_landmarker.task)         [RGB]
- Beard suppression:
    1) Ear-line cutoff (remove everything below a line slightly ABOVE ear level)
    2) Prefer-top connected component filtering (ditch lower blobs)
- Live sliders (trackbars) for tuning

Install:
  pip install mediapipe opencv-python numpy

Download models (put next to this main.py):
  curl -L -o hair_segmentation.tflite https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite
  curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Run:
  python main.py

Keys:
  q / ESC - quit
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp


# ------------------- Paths -------------------
HAIR_MODEL_PATH = os.environ.get("HAIR_MODEL_PATH", "./hair_segmentation.tflite")
FACE_MODEL_PATH = os.environ.get("FACE_MODEL_PATH", "./face_landmarker.task")

# ------------------- Camera / Perf -------------------
DEFAULT_CAP_WIDTH = 640
DEFAULT_CAP_HEIGHT = 480
SEGMENTATION_DOWNSCALE = 0.5  # 0.35-0.6 typically good

# ------------------- Visualization -------------------
HAIR_TINT_COLOR = (0, 255, 0)  # BGR

# ------------------- Face landmark indices (approx ear-side points) -------------------
# These are stable face-edge points near ears (not perfect "ear tips" but good for a cutoff line).
LEFT_EAR_APPROX_IDX = 234
RIGHT_EAR_APPROX_IDX = 454


def die(msg: str, code: int = 1):
    print(msg)
    raise SystemExit(code)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def keep_best_components(mask_u8: np.ndarray, keep_k: int = 1, top_weight: float = 3.5) -> np.ndarray:
    """
    Keep K connected components from a binary mask (0/255).
    Score = area * (1 + top_weight * topness), where topness favors blobs higher in the image.
    This helps remove beard/clothes blobs in head+chest framing.
    """
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8

    H, W = mask_u8.shape[:2]
    scored = []
    for i in range(1, num):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        _, cy = centroids[i]
        topness = 1.0 - (cy / max(1.0, H))  # top=1 bottom=0
        score = area * (1.0 + top_weight * topness)
        scored.append((score, i))

    scored.sort(reverse=True, key=lambda x: x[0])
    keep = {idx for _, idx in scored[:keep_k]}

    out = np.zeros_like(mask_u8)
    for idx in keep:
        out[labels == idx] = 255
    return out


def make_trackbars():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 520, 320)

    # Confidence threshold
    cv2.createTrackbar("Hair thr (0-100)", "Controls", 80, 100, lambda x: None)

    # Keep components (1 is best)
    cv2.createTrackbar("Keep comps (1-3)", "Controls", 1, 3, lambda x: None)

    # Ear margin in SEG-RES pixels (how far ABOVE ear line to cut)
    cv2.createTrackbar("Ear margin px", "Controls", 10, 40, lambda x: None)

    # Morph cleanup kernel size (odd)
    cv2.createTrackbar("Morph ksize", "Controls", 3, 15, lambda x: None)

    # Feather edges
    cv2.createTrackbar("Blur sigma", "Controls", 2, 10, lambda x: None)

    # Temporal smoothing (EMA). Higher = smoother but laggier.
    cv2.createTrackbar("Temporal smooth", "Controls", 88, 100, lambda x: None)


def read_params():
    thr = cv2.getTrackbarPos("Hair thr (0-100)", "Controls") / 100.0
    keep_comps = cv2.getTrackbarPos("Keep comps (1-3)", "Controls")
    ear_margin_px = cv2.getTrackbarPos("Ear margin px", "Controls")
    morph_ksize = cv2.getTrackbarPos("Morph ksize", "Controls")
    blur_sigma = cv2.getTrackbarPos("Blur sigma", "Controls")
    temporal = cv2.getTrackbarPos("Temporal smooth", "Controls") / 100.0

    thr = clamp(thr, 0.0, 1.0)
    keep_comps = clamp(keep_comps, 1, 3)
    ear_margin_px = clamp(ear_margin_px, 0, 200)

    morph_ksize = clamp(morph_ksize, 1, 31)
    if morph_ksize % 2 == 0:
        morph_ksize += 1

    blur_sigma = clamp(blur_sigma, 0, 50)
    temporal = clamp(temporal, 0.0, 0.99)

    return thr, keep_comps, ear_margin_px, morph_ksize, blur_sigma, temporal


def main():
    # --------- Check models ----------
    if not os.path.exists(HAIR_MODEL_PATH):
        die(
            f"Missing hair model: {HAIR_MODEL_PATH}\n"
            "Download:\n"
            "  curl -L -o hair_segmentation.tflite https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite"
        )
    if not os.path.exists(FACE_MODEL_PATH):
        die(
            f"Missing face model: {FACE_MODEL_PATH}\n"
            "Download:\n"
            "  curl -L -o face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

    # --------- MediaPipe Tasks: Hair Segmenter ----------
    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode

    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=HAIR_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    try:
        segmenter = ImageSegmenter.create_from_options(seg_options)
    except Exception as e:
        die(f"Error initializing ImageSegmenter: {e}")

    # --------- MediaPipe Tasks: Face Landmarker ----------
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    try:
        face_landmarker = FaceLandmarker.create_from_options(face_options)
    except Exception as e:
        die(f"Error initializing FaceLandmarker: {e}")

    # --------- OpenCV Camera ----------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        die("Error: Could not open webcam. Try VideoCapture(1) or close other camera apps.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAP_HEIGHT)

    make_trackbars()

    prev_alpha = None
    prev_t = time.time()

    debug_left = 5

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Warning: Failed to read frame.")
                break

            H, W = frame_bgr.shape[:2]

            # timestamps must be monotonic and shared across tasks
            ts_ms = int(time.monotonic() * 1000)

            # ---------- Face landmarks (RGB) ----------
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            face_result = face_landmarker.detect_for_video(face_mp_image, ts_ms)

            ear_cut_y_full = None
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                lm = face_result.face_landmarks[0]
                ly = lm[LEFT_EAR_APPROX_IDX].y
                ry = lm[RIGHT_EAR_APPROX_IDX].y
                ear_y = max(ly, ry)  # lower of the two
                ear_cut_y_full = int(ear_y * H)

            # ---------- Hair segmentation (RGBA) ----------
            frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)

            if SEGMENTATION_DOWNSCALE < 1.0:
                seg_w = max(1, int(W * SEGMENTATION_DOWNSCALE))
                seg_h = max(1, int(H * SEGMENTATION_DOWNSCALE))
                seg_rgba = cv2.resize(frame_rgba, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
                seg_mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=seg_rgba)
            else:
                seg_w, seg_h = W, H
                seg_mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)

            seg_result = segmenter.segment_for_video(seg_mp_image, ts_ms)

            # ---------- Params ----------
            thr, keep_comps, ear_margin_px, morph_ksize, blur_sigma, temporal = read_params()
            kernel = np.ones((morph_ksize, morph_ksize), np.uint8)

            output_frame = frame_bgr.copy()
            mask_display = np.zeros((H, W), dtype=np.uint8)

            if seg_result is not None and seg_result.confidence_masks and len(seg_result.confidence_masks) > 0:
                # hair index is usually 1 for hair/background models
                if len(seg_result.confidence_masks) >= 2:
                    hair_conf = seg_result.confidence_masks[1].numpy_view().astype(np.float32)
                else:
                    hair_conf = seg_result.confidence_masks[0].numpy_view().astype(np.float32)

                if debug_left > 0:
                    print(f"Debug: hair_conf min/max (seg): {hair_conf.min():.3f}/{hair_conf.max():.3f}, masks={len(seg_result.confidence_masks)}")
                    debug_left -= 1

                # Threshold confidence -> binary (seg-res)
                hair_bin = (hair_conf >= thr).astype(np.uint8) * 255

                # ----- Ear-line cutoff (remove beard) -----
                # Convert full-res y to seg-res y, then cut BELOW (ear_line - margin)
                if ear_cut_y_full is not None:
                    ear_cut_y_seg = int((ear_cut_y_full / H) * seg_h)
                    cut_y = max(0, ear_cut_y_seg - ear_margin_px)
                    hair_bin[cut_y:, :] = 0
                else:
                    # If face not found, fall back to keeping top ~60% of seg frame (head+chest framing)
                    hair_bin[int(seg_h * 0.60):, :] = 0

                # Morph cleanup
                hair_bin = cv2.morphologyEx(hair_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
                hair_bin = cv2.morphologyEx(hair_bin, cv2.MORPH_OPEN, kernel, iterations=1)

                # Prefer-top component(s)
                hair_bin = keep_best_components(hair_bin, keep_k=keep_comps, top_weight=3.5)

                # Upscale to full res
                if SEGMENTATION_DOWNSCALE < 1.0:
                    hair_bin_full = cv2.resize(hair_bin, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    hair_bin_full = hair_bin

                # Feather edges -> alpha
                if blur_sigma > 0:
                    hair_soft = cv2.GaussianBlur(hair_bin_full, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                else:
                    hair_soft = hair_bin_full

                alpha = hair_soft.astype(np.float32) / 255.0

                # Temporal smoothing (EMA)
                if prev_alpha is None:
                    prev_alpha = alpha
                alpha = temporal * prev_alpha + (1.0 - temporal) * alpha
                prev_alpha = alpha

                # Blend tinted overlay
                overlay = np.empty_like(frame_bgr, dtype=np.uint8)
                overlay[:] = HAIR_TINT_COLOR
                output_frame = (
                    frame_bgr.astype(np.float32) * (1.0 - alpha[..., None]) +
                    overlay.astype(np.float32) * alpha[..., None]
                ).astype(np.uint8)

                mask_display = (alpha * 255).astype(np.uint8)

            # ---------- Draw debug ear cutoff line ----------
            # Show the cutoff line on output so you can confirm beard is removed.
            if ear_cut_y_full is not None:
                # Convert ear_margin_px (seg px) into full-res px for drawing line consistently
                ear_margin_full = int((ear_margin_px / max(1, seg_h)) * H)
                yline = max(0, ear_cut_y_full - ear_margin_full)
                cv2.line(output_frame, (0, yline), (W - 1, yline), (255, 255, 255), 2)

            # ---------- FPS ----------
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            cv2.putText(
                output_frame,
                f"FPS:{fps:.1f} thr:{thr:.2f} earMarginPx:{ear_margin_px} comps:{keep_comps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Original Webcam Feed", frame_bgr)
            cv2.imshow("Hair Mask (Grayscale)", mask_display)
            cv2.imshow("Hair Overlay", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            segmenter.close()
        except Exception:
            pass
        try:
            face_landmarker.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()