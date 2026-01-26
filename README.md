# Vision-Based Teleoperation for xArm 7 

This project implements a gesture-control interface for the xArm 7 robotic manipulator. It utilizes **Google MediaPipe** for hand tracking and maps coordinates to the robot via a custom Python wrapper.

##  Key Features

* **Hardware Abstraction Layer:** Includes a wrapper class for the xArm SDK to simplify safety checks and connection handling.
* **Simulation Mode (Mock Driver):** A custom "Digital Twin" driver that mimics the robot's behavior. This allows for offline testing and logic verification without physical hardware.
* **Safety Protocols:** Implements automatic "Home" positioning and error checking upon initialization.
* **Real-Time Telemetry:** Bi-directional communication monitoring (FPS & Cartesian Coordinates).

## Project Structure

```text
xArm_Project/
├── myLibs/
│   └── ufactory/
│       ├── xarm_class_joint_space.py  # Real Robot Driver (Supervisor's Wrapper)
│       └── mock_xarm.py               # Simulated Robot Driver (For offline testing)
├── test1_verification.py              # Main Test Script (Connection & Latency Check)
└── .gitignore                         # Git configuration