import cv2
import numpy as np
import os
import math
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
currentVolumeDb = volume.GetMasterVolumeLevel()


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    black_image_shape = image.shape
    black_image = np.zeros(black_image_shape, dtype = "uint8")
    (h, w) = image.shape[:2]
    center_coordinates = (w//2, h//2)
    # Radius of circle
    radius = 50
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = -1
    black_image = cv2.circle(black_image, center_coordinates, radius, color, thickness)
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmark_style(),
            drawing_styles.get_default_hand_connection_style())
        #center point
        x = w//2
        y = h//2
        #Thumb Coordinates
        x22 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w)
        y22 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)
        #angle between two lines
        m1 = (h//2 - h//2) / (0 - w//2)
        A = math.atan(m1) * 180 / math.pi
        try:
          m2 = (y22 - h//2) / (x22 - w//2)
        except:
          m2 = 0
        B = math.atan(m2) * 180 / math.pi
        degree =  B-A
        if degree < 0:
          degree = 0
        if degree > 65:
          degree = 65
        #reverse the volume levels  
        lvl = (65 - degree)
        volume.SetMasterVolumeLevel(-lvl, None)
        
    cv2.imshow('Volume Knob', black_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
