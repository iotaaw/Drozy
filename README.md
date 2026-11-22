# Drowsiness Detection System Using Computer Vision

A real-time drowsiness detection system built using Python and computer vision. The system monitors eye closure through a webcam and generates a visual and audio alert when prolonged eye closure is detected.

## Features
- Live webcam monitoring
- Face detection using dlib
- 68-point facial landmark detection
- Left and right eye tracking
- Eye Aspect Ratio (EAR) calculation
- Drowsiness detection using threshold and frame counter
- Visual alert: DROWSINESS DETECTED
- Audio alarm using winsound

## Technologies Used
- Python
- OpenCV
- dlib
- imutils
- NumPy
- SciPy
- winsound

## Project Files
- `main.py` – Main application
- `Drowsiness_Detection_Presentation.pptx` – Project presentation
- `Drowsiness_Detection_Project_Report.docx` – Project report
- `Drowsiness_Detection_Project_Report.pdf` – PDF report
- `PROJECT_REPORT.md` – Report in Markdown format

## Requirements
- Windows computer
- Webcam
- Python
- Facial landmark model file: `shape_predictor_68_face_landmarks.dat`

## Installation

Install the required libraries:

```bash
pip install opencv-python imutils scipy numpy dlib
```

Download the dlib 68-point facial landmark model and place it in the project folder.

## Run the Project

```bash
python main.py
```

Press `Q` to exit the application.

## Detection Parameters
- EAR Threshold: 0.3
- Consecutive Frames: 48
- Alarm Frequency: 2500 Hz
- Alarm Duration: 1000 milliseconds

## Project Purpose
This project demonstrates how computer vision can be used to monitor eye closure and identify signs of possible drowsiness. It is an educational prototype and is not a certified automotive safety system.