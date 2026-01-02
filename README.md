# Forward Collision Warning System with Explainable AI

## Overview
This project implements an agent-based Forward Collision Warning (FCW) system. It utilizes a modular architecture with specialized agents for perception (YOLOv8), distance estimation, decision-making, and alerting. A key feature is the Explainable AI (XAI) module, which provides visual and textual justifications for every warning.

## Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `carla` package requires a specific version matching your simulator.*
3.  Ensure `yolov8n.pt` is present in the root directory.

## Usage
The system supports three modes of operation:

### 1. Dataset Evaluation Mode
Processes static images from a dataset (e.g., BDD100K) to evaluate the pipeline on diverse real-world scenarios.
```bash
python main.py --source dataset --data_path data/
```

### 2. CARLA Simulation Mode
Connects to the CARLA simulator to run real-time scenarios with ground truth validation.
```bash
python main.py --source carla
```

### 3. Video Processing Mode
Processes a video file for demonstration purposes.
```bash
python main.py --source video --data_path data/
```

## CARLA Simulation Validation
To validate the efficacy and accuracy of the Forward Collision Warning system, we integrate the **CARLA Simulator** as a high-fidelity virtual test environment. This setup allows for quantitative performance analysis against absolute ground truth, which is difficult to obtain in real-world drive testing.

### Real-Time Data Generation
The system interfaces with CARLA to spawn an ego vehicle (Tesla Model 3) equipped with a front-facing RGB camera. This camera streams high-resolution frames (640x480) directly to the **Perception Agent** in real-time. This synthetic data stream mimics the characteristics of real-world dashcam footage, allowing us to test the computer vision pipeline's robustness in a controlled, deterministic environment.

### Ground Truth Validation
A significant advantage of simulation-based validation is the availability of perfect ground truth (GT) data. 
*   **Distance Validation**: We extract the exact Euclidean distance between the ego vehicle and the lead vehicle using CARLA's world coordinate system. This scalar value serves as the benchmark for evaluating the **Distance Estimation Agent**, enabling the calculation of error metrics (e.g., Mean Absolute Error) for the monocular distance estimation algorithm.
*   **Time-To-Collision (TTC) Accuracy**: By retrieving the exact velocity vectors of both vehicles, we compute the ground truth TTC. The **Decision Agent** utilizes this GT data to trigger warnings, ensuring that the system's "Unsafe" alerts are physically justified.
*   **Analysis**: Discrepancies between the vision-based estimates and the physics-based ground truth are logged to `outputs/carla_events.csv`, facilitating granular error analysis and system tuning.

## Collision Avoidance Extension (Automatic Braking)
To move beyond passive warnings, the system has been extended with a **Collision Avoidance Agent** capable of active vehicle control within the CARLA simulation environment. This module transforms the system from a passive ADAS (Advanced Driver Assistance System) to an active safety system.

### Control Logic
The agent operates on a deterministic, tiered logic based on Time-To-Collision (TTC) thresholds, strictly separating warning generation from control intervention:
*   **Warning (1.5s < TTC ≤ 2.5s)**: The system issues a textual and visual alert but maintains vehicle speed (`Warning Only`).
*   **Critical Braking (0.8s < TTC ≤ 1.5s)**: The system disables autopilot and applies moderate braking (50% intensity) to mitigate collision risk.
*   **Emergency Braking (TTC ≤ 0.8s)**: The system applies full braking (100% intensity) to execute an emergency stop.

### Integration & Verification
*   **CARLA Control API**: The agent interfaces directly with the CARLA vehicle control API, overriding the autopilot during critical events to apply longitudinal control (throttle/brake). Use `python main.py --source carla` to activate this mode.
*   **Explainable Actions**: The **XAI Module** now explains control decisions, detailing the specific action taken (e.g., "ACTION: Apply Full Braking") and the reasoning (TTC, Distance) alongside the visual alert.
*   **Event Logging**: All avoidance actions are logged to `outputs/carla_events.csv`, capturing the applied brake intensity, TTC, and vehicle states for post-simulation verification of collision prevention.
