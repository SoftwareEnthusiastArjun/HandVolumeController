# Hand Volume Control with Mediapipe

This Python script demonstrates hand tracking and volume control using Mediapipe and OpenCV. It uses a webcam feed to detect hand landmarks in real-time and adjusts the system volume based on the distance between the thumb and index finger tips.

## Features:
- **Hand Tracking**: Utilizes Mediapipe library to detect and track hand landmarks from webcam frames.
  
- **Visual Feedback**: Draws landmarks, connections, circles, and lines on the video feed to visualize hand movements and gestures.
  
- **Volume Control**: Maps the distance between thumb and index finger tips to adjust the system volume using the Windows Core Audio API (pycaw library).

## Requirements:
- Python (3.x recommended)
- OpenCV (`cv2`) for capturing and processing webcam frames.
- Mediapipe (`mediapipe`) for hand tracking and landmark detection.
- `pycaw` library for system audio control.

## Usage:
### Installation:
Install required Python packages using pip:
  `pip install opencv-python mediapipe pycaw`

## Interaction:

- Hold your hand in front of the webcam.
- Adjust the distance between your thumb and index finger to control the displayed rectangle height, percentage text, and system volume.
## Code Structure:
### Initialization:
- Imports necessary libraries (cv2, mediapipe, pycaw, time, math, numpy).
- Sets up Mediapipe for hand tracking and initializes audio control using pycaw.
### Main Class (HandControlVolume):
- __init__: Initializes Mediapipe modules and audio control settings.
- recognize: Main function for hand tracking and volume control loop.
### Processing Loop (recognize function):
- Captures frames from the webcam.
- Processes each frame for hand landmarks using Mediapipe.
- Calculates hand gestures and maps them to control system volume.
- Displays visual overlays and updates volume based on hand movements.
## References:
- [Hand Tracking 30 FPS using CPU | OpenCV Python (2021) | Computer Vision](https://youtu.be/NZde8Xt78Iw?si=HkRO2jhmPc-12KkJ)

- [Gesture Volume Control | OpenCV Python | Computer Vision](https://youtu.be/9iEPzbG-xLE?si=hWVbv4rJwhphfQ_z)
