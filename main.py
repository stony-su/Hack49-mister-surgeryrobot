import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# To store previous hand positions for the trail
trail_points = []

def generate_gcode(x, y, z):
    z = z * 1e7
    return f"G0 X{x} Y{y} Z{z:.2f}"

previous_points = []

# Function to round and smooth points (you can adjust the rounding as needed)
def smooth_point(x, y, window_size=5):
    global previous_points

    # Add the new point to the list
    previous_points.append((x, y))

    # Keep only the last 'window_size' points
    if len(previous_points) > window_size:
        previous_points.pop(0)

    # Compute the average of the points in the buffer
    avg_x = sum(p[0] for p in previous_points) / len(previous_points)
    avg_y = sum(p[1] for p in previous_points) / len(previous_points)

    return int(avg_x), int(avg_y)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for hand tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe to detect hands
    results = hands.process(rgb_frame)

    output_frame = frame.copy()
    hand_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the wrist coordinates (landmark 0) for Z depth
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_x = wrist.x * frame.shape[1]
            hand_y = wrist.y * frame.shape[0]
            hand_z = wrist.z

            # Draw hand landmarks on the output frame
            mp.solutions.drawing_utils.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Smooth the hand position
            hand_x, hand_y = smooth_point(hand_x, hand_y)
            hand_position = (hand_x, hand_y, hand_z)

            # Generate G-code (for external use)
            gcode = generate_gcode(hand_x, hand_y, hand_z)
            print(gcode)

            # Append the current hand position to the trail
            trail_points.append((hand_x, hand_y))

    # Visualization for hand movement trail
    visualization = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw trail if we have more than 1 point in the trail
    if len(trail_points) > 1:
        for i in range(1, len(trail_points)):
            cv2.line(visualization, trail_points[i - 1], trail_points[i], (0, 255, 0), 1)  # Thinner line

    # Draw polygon if at least 3 points are available (no fill)
    if len(trail_points) >= 3:
        pts = np.array(trail_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(visualization, [pts], isClosed=True, color=(0, 255, 0), thickness=1)  # Thinner polygon

    # Show hand position with a green circle
    if hand_position:
        cv2.circle(visualization, (hand_position[0], hand_position[1]), 5, (0, 255, 0), -1)  # Smaller circle

    # Show the original video feed
    cv2.imshow("Camera Feed", frame)
    cv2.moveWindow("Camera Feed", 0, 0)

    # Show the processed video feed with hand landmarks
    cv2.imshow("Processed Feed", output_frame)
    cv2.moveWindow("Processed Feed", 920, 0)

    # Show the visualization with the trail and polygon
    cv2.imshow("Hand Position Visualization", visualization)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
