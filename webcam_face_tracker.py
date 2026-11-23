"""Real-time webcam face tracker for Windows.

MediaPipe Tasks is used when the optional model file exists next to this script.
If MediaPipe or its model is unavailable, the program automatically uses the
OpenCV Haar Cascade bundled with cv2.

Controls:
    q or Esc  Quit
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2


WINDOW_TITLE = "Webcam Face Tracker"
MODEL_FILENAME = "face_detection_short_range.tflite"
HAND_MODEL_FILENAME = "hand_landmarker.task"
SMOOTHING = 0.72
FULLSCREEN = False


def application_dir() -> Path:
    """Return the folder containing the script, or PyInstaller's temp folder."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def try_create_mediapipe_detector():
    """Create a modern MediaPipe Tasks detector, if its model is available."""
    model_candidates = [
        application_dir() / MODEL_FILENAME,
        application_dir() / "models" / MODEL_FILENAME,
    ]
    model_path = next((path for path in model_candidates if path.is_file()), None)
    if model_path is None:
        return None


def try_create_hand_detector():
    """Create a MediaPipe hand detector when the optional hand model exists."""
    model_path = application_dir() / HAND_MODEL_FILENAME
    if not model_path.is_file():
        model_path = application_dir() / "models" / HAND_MODEL_FILENAME
    if not model_path.is_file():
        return None
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Hand detection unavailable ({exc}); continuing without it.")
        return None

    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        return vision.FaceDetector.create_from_options(options)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"MediaPipe unavailable ({exc}); using OpenCV Haar Cascade.")
        return None


def detect_with_mediapipe(detector, frame) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, width, height) for the highest-confidence face."""
    import mediapipe as mp

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    if not result.detections:
        return None

    detection = max(
        result.detections,
        key=lambda item: item.categories[0].score if item.categories else 0.0,
    )
    box = detection.bounding_box
    return box.origin_x, box.origin_y, box.width, box.height


def detect_with_haar(cascade, frame) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    return tuple(max(0, int(value)) for value in max(faces, key=lambda item: item[2] * item[3]))


def detect_hands(detector, frame):
    """Return MediaPipe hand landmark results, or an empty list."""
    import mediapipe as mp

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect(mp_image).hand_landmarks


def draw_hands(frame, hands) -> None:
    if not hands:
        return
    height, width = frame.shape[:2]
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    for hand in hands:
        points = [(int(point.x * width), int(point.y * height)) for point in hand]
        for first, second in connections:
            cv2.line(frame, points[first], points[second], (255, 120, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 220, 255), -1)


def hand_gesture(hand) -> str:
    """Classify gestures using the four fingers (thumb is ignored)."""
    # Landmark tips: index, middle, ring, pinky. Their lower joints are used
    # as a simple, camera-friendly test for whether each finger is extended.
    tips = (8, 12, 16, 20)
    joints = (6, 10, 14, 18)
    extended = [hand[tip].y < hand[joint].y for tip, joint in zip(tips, joints)]
    if extended == [True, False, False, False]:
        return "DRAW"
    if not any(extended):
        return "PAUSE"
    if all(extended):
        return "CLEAR"
    if extended[:2] == [True, True] and not any(extended[2:]):
        return "PAUSE"
    return "PAUSE"


def draw_air_gesture(frame, hands, canvas, previous_point, clear_time):
    """Draw with an index finger and return updated drawing state."""
    gesture = "NO HAND"
    current_point = None
    now = time.perf_counter()
    if hands:
        hand = hands[0]
        gesture = hand_gesture(hand)
        height, width = frame.shape[:2]
        index_tip = hand[8]
        current_point = (int(index_tip.x * width), int(index_tip.y * height))

        if gesture == "DRAW":
            if previous_point is not None:
                cv2.line(canvas, previous_point, current_point, (70, 230, 255), 6, cv2.LINE_AA)
            cv2.circle(canvas, current_point, 8, (255, 255, 255), -1, cv2.LINE_AA)
        elif gesture == "CLEAR" and now - clear_time > 1.0:
            canvas[:] = 0
            clear_time = now
            previous_point = None
        else:
            previous_point = None
    else:
        previous_point = None

    if cv2.countNonZero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)) > 0:
        frame[:] = cv2.addWeighted(frame, 0.78, canvas, 0.95, 0)
    cv2.putText(
        frame,
        f"Gesture: {gesture} | Index finger: draw | Open palm: clear | F: fullscreen",
        (18, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return current_point if gesture == "DRAW" else None, clear_time


def smooth_box(previous, current):
    """Smooth a bounding box using an exponential moving average."""
    if current is None:
        return previous
    if previous is None:
        return current
    return tuple(
        int(SMOOTHING * old + (1.0 - SMOOTHING) * new)
        for old, new in zip(previous, current)
    )


def available_cameras(max_index: int = 10):
    """Return camera indexes that Windows exposes to OpenCV."""
    found = []
    for index in range(max_index):
        test_camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if test_camera.isOpened():
            ok, _ = test_camera.read()
            if ok:
                found.append(index)
        test_camera.release()
    return found


def open_camera(index: int):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def draw_overlay(frame, face_box, fps: float, detector_name: str) -> None:
    height, width = frame.shape[:2]
    if face_box is None:
        status = "Face: NOT DETECTED"
        color = (0, 0, 255)
    else:
        x, y, box_width, box_height = face_box
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        box_width = min(box_width, width - x)
        box_height = min(box_height, height - y)
        center_x = x + box_width // 2
        center_y = y + box_height // 2
        color = (0, 220, 0)
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
        cv2.drawMarker(
            frame,
            (center_x, center_y),
            (255, 180, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)
        status = f"Face: DETECTED  Center: ({center_x}, {center_y})"

    cv2.rectangle(frame, (8, 8), (min(width - 8, 530), 76), (20, 20, 20), -1)
    cv2.putText(frame, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    cv2.putText(
        frame,
        f"Detector: {detector_name}  FPS: {fps:.1f}  |  q/Esc: quit",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
    )


def main() -> int:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV cascade: {cascade_path}")

    mediapipe_detector = try_create_mediapipe_detector()
    hand_detector = try_create_hand_detector()
    detector_name = "MediaPipe Tasks" if mediapipe_detector is not None else "OpenCV Haar"

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    camera_indexes = available_cameras()
    if not camera_indexes:
        raise RuntimeError("No cameras were found. Connect a webcam or expose your phone as a Windows webcam first.")
    selected_position = 0
    camera = open_camera(camera_indexes[selected_position])
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check Windows camera permissions and device selection.")
    cv2.createTrackbar("Camera index", WINDOW_TITLE, selected_position, len(camera_indexes) - 1, lambda _: None)

    previous_time = time.perf_counter()
    fps = 0.0
    smoothed_face_box = None
    drawing_canvas = None
    previous_draw_point = None
    last_clear_time = 0.0
    fullscreen = False
    try:
        while True:
            requested_position = cv2.getTrackbarPos("Camera index", WINDOW_TITLE)
            if requested_position != selected_position:
                camera.release()
                selected_position = requested_position
                camera = open_camera(camera_indexes[selected_position])
                smoothed_face_box = None
                drawing_canvas = None
                previous_draw_point = None
                if not camera.isOpened():
                    raise RuntimeError("The selected camera could not be opened.")

            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            if drawing_canvas is None or drawing_canvas.shape != frame.shape:
                drawing_canvas = __import__("numpy").zeros_like(frame)
            if mediapipe_detector is not None:
                face_box = detect_with_mediapipe(mediapipe_detector, frame)
            else:
                face_box = detect_with_haar(cascade, frame)
            smoothed_face_box = smooth_box(smoothed_face_box, face_box)
            hands = detect_hands(hand_detector, frame) if hand_detector is not None else []

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
            previous_time = now
            draw_hands(frame, hands)
            draw_overlay(frame, smoothed_face_box, fps, detector_name)
            previous_draw_point, last_clear_time = draw_air_gesture(
                frame, hands, drawing_canvas, previous_draw_point, last_clear_time
            )
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:
        camera.release()
        if mediapipe_detector is not None:
            mediapipe_detector.close()
        if hand_detector is not None:
            hand_detector.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"Error: {error}")
        raise SystemExit(1)

