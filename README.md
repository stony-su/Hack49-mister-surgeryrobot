# MR surgery robot
![logo.jpeg](https://cdn.dorahacks.io/static/files/192b1d04029e5f3f67019d747fd8ee00.jpeg)

# The problem
One of the biggest problems in surgery is medical error. Our project aims to overcome this problem by creating a very precise and cost-effective robot!

## What it is
MR surgery robot is a machine that is like an extension of your hand. All you need to do is move your hand around in the view of the camera that connects to the robot. Then, the robot will move around accordingly. If you make a pinching movement, the knife/medical cutting tool will move down, and will move back up when you open your hand. It has an easily replaceable medical tool chamber, made with a hole and a screw to secure it.

## How it works
It uses TensorFlow + OpenCV to detect your hand movements (including a hand database called mediapipe) and sends GCode to the robot, which takes the code and translates it into movements.
