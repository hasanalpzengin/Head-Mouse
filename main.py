import cv2
import dlib
import numpy as np
import pyautogui
import imutils
import time
from imutils import face_utils

WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)
MOUTH_AR_THRESH = 0.6

shape_predictor = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(l_eye_start, l_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(r_eye_start, r_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
(nose_start, nose_end) = face_utils.FACIAL_LANDMARKS_IDXS['nose']

#webcm
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
mouse_x = 0
mouse_y = 0
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h
padding_x, padding_y = 50, 50
control_padding = 20
#set guide rect
rect_start = (cam_w//2-100, cam_h//2-100)
rect_end = (cam_w//2+100, cam_h//2+100)
process = False
counter = 0
cursor_coordinates = ()
pyautogui.FAILSAFE = False

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    D = np.linalg.norm(mouth[12] - mouth[16])

    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2 * D)

    # Return the mouth aspect ratio
    return mar

while True:
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    rects = detector(gray, 0)
    # if face detected
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    if(process == True):
        mouth = shape[mouth_start:mouth_end]
        nose = shape[nose_start: nose_end]

        cv2.circle(frame, (nose[3, 0], nose[3, 1]), 5, BLUE_COLOR, 1)
        cv2.rectangle(frame, rect_start, rect_end, RED_COLOR, 2)
        cv2.line(frame, (cursor_coordinates[0]-padding_x, cursor_coordinates[1]), (cursor_coordinates[0]+padding_x, cursor_coordinates[1]), YELLOW_COLOR, 2)
        cv2.line(frame, (cursor_coordinates[0], cursor_coordinates[1]-padding_y), (cursor_coordinates[0], cursor_coordinates[1]+padding_y), YELLOW_COLOR, 2)
        cv2.imshow("Frame", frame)
        if nose[3,0] > cursor_coordinates[0]+control_padding:
            if mouse_x <= 1910:
                mouse_x += 5
        elif nose[3,0] < cursor_coordinates[0]-control_padding:
            if mouse_x >= 10:
                mouse_x -= 5
        if nose[3,1] > cursor_coordinates[1]+control_padding:
            if mouse_y <= 1080:
                mouse_y += 5
        elif nose[3,1] < cursor_coordinates[1]-control_padding:
            if mouse_y >= 10:
                mouse_y -= 5

        #if mouth open click
        mar = mouth_aspect_ratio(mouth)
        if(mar>MOUTH_AR_THRESH):
            pyautogui.click(mouse_x, mouse_y)

        pyautogui.moveTo(mouse_x, mouse_y)
        key = cv2.waitKey(1) & 0xFF
    else:
        #get eyes
        left_eye = shape[l_eye_start:l_eye_end]
        right_eye = shape[r_eye_start:r_eye_end]
        nose = shape[nose_start: nose_end]
        # swap left and right
        temp = left_eye
        left_eye = right_eye
        right_eye = temp

        #is face inside of rectangle
        if(left_eye[3,0]>rect_start[0] and left_eye[3,0]<rect_end[0]
            and right_eye[3,0]>rect_start[0] and right_eye[3,0]<rect_end[0]
            and left_eye[3,1]>rect_start[1] and left_eye[3,1]<rect_end[1]
            and right_eye[3,1]>rect_start[1] and right_eye[3,1]<rect_end[1]):

            cv2.putText(frame, str(counter//10), (cam_w//2-100, cam_h//2+100), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR)
            counter += 1
            if(counter/10 > 10):
                cursor_coordinates = nose[3]
                process = True
        else:
            counter = 0
        cv2.rectangle(frame, rect_start, rect_end, WHITE_COLOR, 2)
        cv2.putText(frame, "Hold your face inside of rectangle for 10 sec", (cam_w//2-100, cam_h//2+200), cv2.FONT_HERSHEY_PLAIN, 1, GREEN_COLOR)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(10) & 0xFF
