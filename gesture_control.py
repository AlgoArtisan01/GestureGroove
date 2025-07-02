import cv2
import mediapipe as mp
import numpy as np
import math
import time
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Play/Pause key event setup
user32 = ctypes.WinDLL('user32')
VK_MEDIA_PLAY_PAUSE = 0xB3

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume_ctrl.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# FPS tracking
p_time = 0

# Play/Pause debounce
last_pause_time = 0
pause_cooldown = 1  # seconds

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Get hand label
            hand_label = result.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            
            # Get landmark list
            lm_list = []
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
            
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # LEFT HAND → VOLUME
            if hand_label == "Left":
                x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)

                # 🔧 FIXED: Use volume scalar for accurate linear system volume
                vol_scalar = np.interp(length, [20, 180], [0.0, 1.0])
                vol_bar = np.interp(length, [20, 180], [400, 150])
                vol_percent = np.interp(length, [20, 180], [0, 100])
                volume_ctrl.SetMasterVolumeLevelScalar(vol_scalar, None)

                # Draw volume bar
                cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 3)
                cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f'{int(vol_percent)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            # RIGHT HAND → PLAY/PAUSE
            elif hand_label == "Right":
                x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]
                x_pinky, y_pinky = lm_list[20][1], lm_list[20][2]
                pinky_dist = math.hypot(x_pinky - x_thumb, y_pinky - y_thumb)

                if pinky_dist < 40 and (time.time() - last_pause_time > pause_cooldown):
                    user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
                    last_pause_time = time.time()
                    cv2.putText(frame, 'Play/Pause', (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # FPS counter
    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time - p_time != 0 else 0
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    cv2.imshow("Gesture Control: Volume + Play/Pause", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
