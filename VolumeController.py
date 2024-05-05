import cv2  # Importing OpenCV library for computer vision tasks
import mediapipe as mp  # Importing Mediapipe library for hand tracking
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # Importing libraries for audio control
import time  # Importing time module for time-related operations
import math  # Importing math module for mathematical calculations
import numpy as np  # Importing NumPy library for numerical operations

class HandControlVolume:
    def __init__(self):
        # Initialize Mediapipe modules for hand tracking
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # Initialize audio control using Windows Core Audio API (pycaw library)
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volume.SetMute(0, None)  # Unmute the system volume
        self.volume_range = self.volume.GetVolumeRange()  # Get the range of system volume levels

    def recognize(self):
        # Calculate FPS (Frames Per Second) for performance monitoring
        fpsTime = time.time()

        # OpenCV video capture from webcam (index 0)
        cap = cv2.VideoCapture(0)
        resize_w = 640  # Resize width of video frame
        resize_h = 480  # Resize height of video frame

        # Initialize parameters for visual display
        rect_height = 0
        rect_percent_text = 0

        # Start hand tracking using Mediapipe
        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            while cap.isOpened():
                # Read a frame from the webcam
                success, image = cap.read()
                image = cv2.resize(image, (resize_w, resize_h))  # Resize the frame

                if not success:
                    print("Empty frame.")
                    continue

                # Improve performance by setting image to non-writable
                image.flags.writeable = False
                # Convert BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Flip the image horizontally for mirror effect
                image = cv2.flip(image, 1)
                # Process the frame using Mediapipe hand tracking
                results = hands.process(image)

                image.flags.writeable = True
                # Convert RGB image back to BGR for display
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Check if hands are detected in the frame
                if results.multi_hand_landmarks:
                    # Iterate through each detected hand
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks and connections on the image
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # Extract hand landmark coordinates
                        landmark_list = []
                        for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y, finger_axis.z
                            ])

                        if landmark_list:
                            # Get coordinates of thumb tip and index finger tip
                            thumb_finger_tip = landmark_list[4]
                            thumb_finger_tip_x = math.ceil(thumb_finger_tip[1] * resize_w)
                            thumb_finger_tip_y = math.ceil(thumb_finger_tip[2] * resize_h)
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = math.ceil(index_finger_tip[1] * resize_w)
                            index_finger_tip_y = math.ceil(index_finger_tip[2] * resize_h)

                            # Calculate middle point between thumb tip and index finger tip
                            finger_middle_point = (thumb_finger_tip_x + index_finger_tip_x) // 2, (
                                    thumb_finger_tip_y + index_finger_tip_y) // 2

                            # Draw circles at thumb tip, index finger tip, and middle point
                            image = cv2.circle(image, (thumb_finger_tip_x, thumb_finger_tip_y), 10, (255, 0, 255), -1)
                            image = cv2.circle(image, (index_finger_tip_x, index_finger_tip_y), 10, (255, 0, 255), -1)
                            image = cv2.circle(image, finger_middle_point, 10, (255, 0, 255), -1)

                            # Draw line between thumb tip and index finger tip
                            image = cv2.line(image, (thumb_finger_tip_x, thumb_finger_tip_y),
                                             (index_finger_tip_x, index_finger_tip_y), (255, 0, 255), 5)

                            # Calculate distance between thumb tip and index finger tip using Pythagorean theorem
                            line_len = math.hypot((index_finger_tip_x - thumb_finger_tip_x),
                                                  (index_finger_tip_y - thumb_finger_tip_y))

                            # Get system volume range
                            min_volume = self.volume_range[0]
                            max_volume = self.volume_range[1]

                            # Map finger distance to control system volume
                            vol = np.interp(line_len, [50, 300], [min_volume, max_volume])
                            self.volume.SetMasterVolumeLevel(vol, None)  # Set system volume level

                            # Map finger distance to rectangle height and percentage text
                            rect_height = np.interp(line_len, [50, 300], [0, 200])
                            rect_percent_text = np.interp(line_len, [50, 300], [0, 100])

                # Display rectangle and percentage text on the frame
                cv2.putText(image, str(math.ceil(rect_percent_text)) + "%", (10, 350),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                image = cv2.rectangle(image, (30, 100), (70, 300), (255, 0, 0), 3)
                image = cv2.rectangle(image, (30, math.ceil(300 - rect_height)), (70, 300), (255, 0, 0), -1)

                # Display FPS (Frames Per Second) on the frame
                cTime = time.time()
                fps_text = 1 / (cTime - fpsTime)
                fpsTime = cTime
                cv2.putText(image, "FPS: " + str(int(fps_text)), (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                # Show the processed image with overlays
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('MediaPipe Hands', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cap.release()

# Start the program
control = HandControlVolume()
control.recognize()
