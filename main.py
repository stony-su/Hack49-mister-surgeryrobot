import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.)

cap = cv2.VideoCapture(0)

def generate_gcode(x, y, z):
    z = z * 1e7
    return f"G0 X{x} Y{y} Z{z:.2f}"

# Mainf
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame horizontally for later selfie-view display
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    output_frame = frame.copy()
    hand_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # pos of wrist (landmark 0) for Z depth
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_x = int(wrist.x * frame.shape[1])
            hand_y = int(wrist.y * frame.shape[0])
            hand_z = wrist.z

            # hand landmarks on output frame
            mp.solutions.drawing_utils.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # stores hand position
            hand_position = (hand_x, hand_y, hand_z)

            # gcode
            gcode = generate_gcode(hand_x, hand_y, hand_z)
            print(gcode)

    # original video feed
    cv2.imshow("Camera Feed", frame)
    cv2.moveWindow("Camera Feed", 0, 0)

    # processed video stream (with hand tracking)
    cv2.imshow("Processed Feed", output_frame)
    cv2.imshow("Camera Feed", frame)
    cv2.moveWindow("Processed Feed", 920, 0)

    # visualize hand point
    visualization = np.zeros((480, 640, 3), dtype=np.uint8)
    if hand_position:
        cv2.circle(visualization, (hand_position[0], hand_position[1]), 10, (0, 255, 0), -1)

    cv2.imshow("Hand Position Visualization", visualization)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
