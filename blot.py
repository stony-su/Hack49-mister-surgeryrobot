import cv2
import numpy as np
import mediapipe as mp
import serial
import time
import math
import cv2.cuda

# Initialize the hand-tracking model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("Using GPU for processing.")
else:
    print("Using CPU for processing.")
    
# Open a connection to the 3D printer (adjust port and baudrate for your printer)
printer_port = 'COM10'  # or '/dev/ttyUSB0' on Linux
baudrate = 115200
ser = serial.Serial(printer_port, baudrate, timeout=1)

# Wait for printer to initialize
time.sleep(2)  # Some printers need a delay after opening serial communication

cap = cv2.VideoCapture(0)

def generate_gcode(x, y):
    # Round values to 2 decimal places and increase feed rate for high-speed movement
    x = round(x, 1)
    y = round(y, 1)
    return f"G0 X{x} Y{y} F300000"  # Set feed rate (speed) to 3000 mm/min

def control_servo(pinch_distance):
    # Convert pinch distance to servo angle (0 to 90 degrees)
    servo_angle = int((1 - pinch_distance) * 90)  # Invert distance for servo control
    return f"M280 P0 S{90 - servo_angle}"  # Example G-code for controlling a servo

def calculate_pinch_distance(thumb_tip, index_tip):
    # Calculate the Euclidean distance between thumb and index finger tips
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance

def send_gcode(gcode):
    if ser.is_open:
        ser.write(f"{gcode}\n".encode())
        print(f"Sent: {gcode}")

# Variables to track previous values for pinch distance
prev_pinch_distance = -1

# Main loop
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for later selfie-view display
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    output_frame = frame.copy()
    hand_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the wrist (landmark 0) for X and Y
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            hand_x = int(wrist.x * frame.shape[1])
            hand_y = int(wrist.y * frame.shape[0])
            
            # Scale the pixel coordinates to fit the 120x120 workspace
            hand_x = int((hand_x / frame.shape[1]) * 120)
            hand_y = int((hand_y / frame.shape[0]) * 120)

            # Calculate pinch distance between thumb and index finger
            pinch_distance = calculate_pinch_distance(thumb_tip, index_tip)

            # Normalize pinch distance between 0 (pinched) and 1 (open)
            pinch_distance_normalized = min(max(pinch_distance, 0), 0.1) / 0.1

            # Draw hand landmarks on the output frame
            mp.solutions.drawing_utils.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Store hand position
            hand_position = (hand_x, hand_y, pinch_distance_normalized)

            # Generate and send G-code for X and Y movements
            gcode = generate_gcode(hand_x, hand_y)
            send_gcode(gcode)

            # Check if pinch distance has significantly changed to control the servo
            if abs(prev_pinch_distance - pinch_distance_normalized) > 0.01:  # Adjust threshold as needed
                servo_command = control_servo(pinch_distance_normalized)  # Adjust servo based on pinch
                send_gcode(servo_command)
                prev_pinch_distance = pinch_distance_normalized  # Update the previous pinch distance

    # Show original video feed
    # cv2.imshow("Camera Feed", frame)

    # Show processed video stream (with hand tracking)
    cv2.imshow("Processed Feed", output_frame)

    # Visualize hand point
    visualization = np.zeros((480, 640, 3), dtype=np.uint8)
    if hand_position:
        cv2.circle(visualization, (hand_position[0], hand_position[1]), 10, (0, 255, 0), -1)

    cv2.imshow("Hand Position Visualization", visualization)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Close the serial connection
ser.close()
