import cv2
import numpy as np

def detect_hand_position(frame):
    # Blur the image to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Define a range for skin color in HSV (adjust for your environment)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Perform morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (likely the hand)
        hand_contour = max(contours, key=cv2.contourArea)

        # Calculate the moments of the contour to get the centroid (center of the hand)
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # X-coordinate (horizontal)
            cy = int(M["m01"] / M["m00"])  # Y-coordinate (vertical)

            # Get the bounding rectangle around the hand
            x, y, w, h = cv2.boundingRect(hand_contour)

            # Estimate depth using the height of the bounding rectangle (h)
            depth = 1 / (h + 1e-5)  # Smaller height -> farther, larger height -> closer

            # Return the position (cx, cy), depth, and mask
            return (cx, cy), depth, mask

    return None, None, mask

def generate_gcode(hand_position, depth):
    if hand_position:
        x, y = hand_position
        # Incorporating the depth into the Z coordinate of the G-code
        return f"G1 X{x} Y{y} Z{depth:.2f} F300\n"
    return None

def main():
    available_cameras = list_cameras()

    if not available_cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for i, cam in enumerate(available_cameras):
        print(f"{i}: Camera {cam}")

    cam_index = int(input(f"Select a camera (0-{len(available_cameras)-1}): "))

    if cam_index < 0 or cam_index >= len(available_cameras):
        print("Invalid camera selection.")
        return

    # Input user-defined dimensions for the window
    user_width = int(input("Enter the width of the new window: "))
    user_height = int(input("Enter the height of the new window: "))

    cap = cv2.VideoCapture(available_cameras[cam_index])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hand_position, depth, processed_mask = detect_hand_position(frame)
        gcode = generate_gcode(hand_position, depth)

        if hand_position:
            # Draw a circle at the center of the hand
            cv2.circle(frame, hand_position, 10, (0, 255, 0), -1)
            # Display depth value as text on the frame
            cv2.putText(frame, f"Depth: {depth:.2f}", (hand_position[0], hand_position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Create a blank image of user-specified dimensions
        blank_image = np.zeros((user_height, user_width, 3), np.uint8)

        if hand_position:
            # Map hand position to new window dimensions
            frame_height, frame_width = frame.shape[:2]
            x_ratio = user_width / frame_width
            y_ratio = user_height / frame_height
            mapped_x = int(hand_position[0] * x_ratio)
            mapped_y = int(hand_position[1] * y_ratio)

            # Draw the hand position as a dot in the user-defined window
            cv2.circle(blank_image, (mapped_x, mapped_y), 10, (0, 255, 0), -1)

        # Display the original frame, the processed mask, and the user-defined window
        cv2.imshow('Hand Tracking', frame)
        cv2.imshow('Processed Mask', processed_mask)
        cv2.imshow(f'Custom {user_width}x{user_height} Window', blank_image)

        if gcode:
            print(gcode)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def list_cameras():
    """Lists available camera indexes."""
    available_cameras = []
    for index in range(10):  # Check camera indexes from 0 to 9
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    main()
