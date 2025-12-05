# Euler ↔ Quaternion Converter & Visualizer

A small Python utility that converts Z-Y-X (yaw, pitch, roll) Euler angles to quaternions and back, demonstrates gimbal-lock handling, and visualizes the original and rotated 3D coordinate frames.

## Requirements

* Python
* NumPy
* Matplotlib
  
  Install dependencies:
  ```
  pip install numpy matplotlib
  ```
## Usage
```
python rotation_converter.py
```
## How to modify for your needs

* To visualize additional orientations, call run_demonstration(yaw_deg, pitch_deg, roll_deg, title) with the desired angles.
