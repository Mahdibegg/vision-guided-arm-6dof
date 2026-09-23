import time
from pathlib import Path

from arm.config_loader import load_simulation_camera_config, SimulationCameraConfig
from arm.vision.simulation_camera import SimulationCamera

def main():
    
    print(f"Loading simulation camera config...")
    config = load_simulation_camera_config()
    print(f"Loaded config: {config}")

    print(f"Connecting to CoppeliaSim at {config.host}:{config.port} on '{config.sensor_path}'...")
    cam = SimulationCamera(config)
    print("Connected successfully! Streaming frames (press 'q' in the window to quit)...")

    frame_count = 0
    start_time = time.time()

    while cam.is_open:
        frame = cam.read()
        frame_count += 1

        if not cam.display(frame):
            break

    elapsed = time.time() - start_time
    if elapsed > 0:
        fps_measured = frame_count / elapsed
        print(f"Stream finished: captured {frame_count} frames in {elapsed:.2f}s ({fps_measured:.1f} FPS)")

    cam.close()

if __name__ == "__main__":
    main()