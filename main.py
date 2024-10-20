import cv2
import numpy as np

def detect_hand_position(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def generate_gcode(hand_position):
    if hand_position:
        x, y = hand_position
        return f"G1 X{x} Y{y} Z0.0 F300\n"
    return None

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hand_position = detect_hand_position(frame)
        gcode = generate_gcode(hand_position)

        if hand_position:
            cv2.circle(frame, hand_position, 10, (0, 255, 0), -1)

        cv2.imshow('Hand Tracking', frame)

        if gcode:
            print(gcode)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
