import cv2
import numpy as np

# Load a frame from webcam or video
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Convert the frame to a CUDA-enabled matrix
gpu_frame = cv2.cuda_GpuMat()
gpu_frame.upload(frame)

# Perform operations such as resizing on the GPU
gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))

# Download the result back to the CPU
resized_frame = gpu_resized.download()

# Show the frame using OpenCV (on CPU)
cv2.imshow("Resized Frame", resized_frame)
cv2.waitKey(0)
