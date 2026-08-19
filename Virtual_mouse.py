import random
from operator import index
import cv2
import mediapipe as mp
from scipy.ndimage import label
import util
import pyautogui
from pynput.mouse import Button, Controller
import time

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False
mouse = Controller()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
last_left_click = 0
last_right_click = 0
last_double_click = 0
last_screenshot = 0

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x,y)

def is_left_click(landmarks_list, thumb_index_dist):
    return (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 90 and thumb_index_dist > 50)

def is_right_click(landmarks_list, thumb_index_dist):
    return (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and thumb_index_dist > 50)

def is_double_click(landmarks_list, thumb_index_dist):
    return (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and thumb_index_dist > 50)

def is_screenshot(landmarks_list, thumb_index_dist):
    return (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and thumb_index_dist < 50)

def detect_gestures(frame, landmarks_list, processed):
    global last_left_click, last_right_click, last_double_click, last_screenshot

    if len(landmarks_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]])

        current_time = time.time()

        if thumb_index_dist < 50 and util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)

        # LEFT CLICK: BEND INDEX FINGER, THUMB FAR
        elif is_left_click(landmarks_list, thumb_index_dist):
            if current_time - last_left_click > 1:
                mouse.press(Button.left)
                mouse.release(Button.left)
                last_left_click = current_time
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # RIGHT CLICK: BEND MIDDLE FINGER, THUMB FAR
        elif is_right_click(landmarks_list, thumb_index_dist):
            if current_time - last_right_click > 1:
                mouse.press(Button.right)
                mouse.release(Button.right)
                last_right_click = current_time
                cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # DOUBLE CLICK: BEND INDEX AND MIDDLE FINGER,
        elif is_double_click(landmarks_list, thumb_index_dist):
            if current_time - last_double_click > 1.5:
                pyautogui.doubleClick()
                last_double_click = current_time
                cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # SCREENSHOT: FIST
        elif is_screenshot(landmarks_list, thumb_index_dist):
            if current_time - last_screenshot > 3:
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f'my_screenshot_{label}.png')
                last_screenshot = current_time
                cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 120), 2)

        # SCROLL UP: CLOSE THUMB AND RING FINGER
        # thumb_ring_dist = util.get_distance([landmarks_list[4], landmarks_list[16]])
        elif util.get_distance([landmarks_list[4], landmarks_list[16]]) < 30:
            pyautogui.scroll(30)
            cv2.putText(frame, "Scroll Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # SCROLL DOWN: CLOSE THUMB AND PINKY FINGER
        # thumb_pinky_dist = util.get_distance([landmarks_list[4], landmarks_list[20]])
        elif util.get_distance([landmarks_list[4], landmarks_list[20]]) < 30:
            pyautogui.scroll(-25)
            cv2.putText(frame, "Scroll Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)


def main():
    cap = cv2.VideoCapture(0)

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # You can change to 1920 for Full HD
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Change to 1080 for Full HD

    draw = mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmarks_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

            detect_gestures(frame, landmarks_list, processed)

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                break


    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
