# VisionSketch

Real-time Windows webcam vision with face tracking, hand gestures, air drawing, fullscreen mode, and selectable camera input—including phones exposed to Windows as webcams.

## Features

- Smooth face bounding box, center marker, coordinates, and FPS
- MediaPipe hand landmarks with gesture controls
- Draw in the air with one raised index finger
- Closed fist or two fingers: pause drawing
- Open palm: clear the canvas
- Press `F` to toggle fullscreen
- Camera selector bar for multiple webcams
- Automatic OpenCV Haar Cascade fallback for face detection

## Quick start

Use Python 3.12 or 3.13 for the broadest compatibility.

```powershell
py -m pip install --upgrade pip
py -m pip install opencv-python mediapipe
py webcam_face_tracker.py
```

The optional hand model is required for hand gestures and air drawing. Place it here:

```text
models\hand_landmarker.task
```

If MediaPipe is unavailable, face tracking still works through OpenCV.

## Phone webcam

A USB cable alone does not automatically make a phone a Windows webcam. Install a phone-webcam app and its Windows companion/driver, connect through USB, and confirm the phone appears in Windows camera apps. Then start VisionSketch and choose it with the `Camera index` bar.

## Controls

| Control | Action |
|---|---|
| Index finger only | Draw |
| Closed fist | Pause |
| Two fingers | Pause |
| Open palm | Clear canvas |
| `F` | Toggle fullscreen |
| `Q` or `Esc` | Exit |

## Build the Windows executable

```powershell
py -m pip install pyinstaller
py -m PyInstaller --onefile --windowed --name VisionSketch `
  --add-data "models\hand_landmarker.task;models" `
  webcam_face_tracker.py
```

The executable is created at `dist\VisionSketch.exe`. If the hand model is not bundled, face tracking still runs, but hand gestures and air drawing are disabled.

## License

Released under the MIT License. See `LICENSE`.
