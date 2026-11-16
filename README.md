# WillowFlow: Cricket Biomechanics Analyzer

WillowFlow is a computer-vision web application that converts a side-on cricket batting video into an annotated analysis of a batter's center-of-gravity movement and estimated front-/back-foot weight transfer.

Built as an Image Processing and Computer Vision (IPCV) project for B.E. Artificial Intelligence & Machine Learning at BNMIT, it demonstrates an end-to-end applied-ML workflow: video ingestion, human-pose inference, temporal feature extraction, visualisation, and an accessible browser interface.

## Why it matters

Batting balance is normally assessed through manual video review. WillowFlow offers a fast, visual first-pass analysis: it marks key body points on each frame, tracks movement through the shot, and produces a concise report that is easier for a player, coach, or analyst to review.

## What it does

- Accepts an uploaded cricket batting video through a Gradio interface.
- Uses **YOLOv8-Pose** to locate body keypoints and selects the primary batter in frames containing multiple people.
- Calculates the midpoint between the hips as a screen-space center-of-gravity (CoG) proxy.
- Establishes a baseline CoG position and measures horizontal displacement throughout the video.
- Estimates front- and back-foot distribution by projecting the CoG between detected ankle positions.
- Labels the dominant foot, overlays a CoG trail and live balance bar, then exports an annotated video.
- Generates a summary containing frame count, resolution, FPS, foot-distribution percentages, shot classification, and CoG displacement statistics.

## Technical approach

```text
Input video
    -> YOLOv8 pose estimation
    -> hip + ankle keypoint filtering
    -> CoG and ankle-span feature calculation per frame
    -> temporal aggregation and dominant-foot classification
    -> annotated MP4 + text report in the browser
```

The application exposes confidence-threshold, facing-direction, and trail-length controls so a user can adapt the analysis to video quality and camera orientation.

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Pose estimation | Ultralytics YOLOv8-Pose |
| Video processing | OpenCV, NumPy |
| User interface | Gradio |
| Output | Annotated MP4 and analysis report |

## Run locally

**Prerequisite:** Python 3.10+ and a webcam is not required.

```bash
git clone https://github.com/<your-username>/WillowFlow.git
cd WillowFlow
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
python cricbalance_app.py
```

The first run downloads `yolov8n-pose.pt` automatically if it is not already present. The app opens at `http://127.0.0.1:7860`.

## Recommended input

For the clearest result, use a short side-on clip in which the batter's full body, especially both hips and feet, remains visible. If labels are reversed, switch the facing-direction option. Lower the keypoint-confidence threshold for more difficult footage.

## Important limitation

The displayed weight split is a **vision-based estimate**, not a force-plate measurement. It is derived from the screen-space relationship between a hip-midpoint CoG proxy and the detected ankle positions. Results depend on camera angle, body visibility, pose-estimation confidence, and foot occlusion. It is best used as a coaching/analysis aid rather than a clinical or biomechanical ground-truth measurement.

## Recruiter snapshot

This project highlights practical experience with:

- applying a pretrained deep-learning model to a domain-specific video problem;
- designing a per-frame feature pipeline and aggregating it into understandable metrics;
- handling uncertain detections and multi-person video frames;
- producing human-readable visual and textual outputs; and
- shipping the workflow as an interactive, local web application.

## Repository structure

```text
WillowFlow/
├── cricbalance_app.py   # Application, video pipeline, and Gradio UI
├── requirements.txt     # Python dependencies
├── README.md            # Project overview and setup instructions
└── .gitignore           # Excludes environments, model weights, and media
```

## Future improvements

- Add persistent per-frame CSV/JSON exports and charts.
- Calibrate measurements against force-plate or motion-capture data.
- Add multi-player selection and camera-perspective validation.
- Package the app with Docker and deploy it for remote use.
