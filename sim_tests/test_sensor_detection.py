from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time

client = RemoteAPIClient()
sim = client.require("sim")

robot_base = sim.getObject("/UR5")
target = sim.getObject("/UR5/target")
tip = sim.getObject("/UR5/tip")   # change path if needed
pickup_sensor = sim.getObject("/proximitySensor")
cube = sim.getObject("/Object")

def move_target(destination, step_size=0.005):
    while True:
        current = np.array(
            sim.getObjectPosition(target, robot_base),
            dtype=float
        )

        goal = np.array(destination, dtype=float)

        diff = goal - current
        distance = np.linalg.norm(diff)

        if distance < step_size:
            sim.setObjectPosition(
                target,
                destination,
                robot_base
            )
            break

        direction = diff / distance
        new_pos = current + direction * step_size

        sim.setObjectPosition(
            target,
            new_pos.tolist(),
            robot_base
        )

        time.sleep(0.02)

def lower_until_cube():
    print("Lowering toward cube...")

    while True:

        result, distance, point, obj, normal = \
            sim.checkProximitySensor(
                pickup_sensor,
                cube
            )

        if result == 1:
            print("CUBE DETECTED")
            print("Distance:", distance)
            return

        current = sim.getObjectPosition(
            target,
            robot_base
        )

        # Lower by 2 mm
        current[2] -= 0.002

        sim.setObjectPosition(
            target,
            current,
            robot_base
        )

        time.sleep(0.03)

# Get cube position directly from simulator for this test
cube_pos = sim.getObjectPosition(
    cube,
    robot_base
)

print("Cube position:", cube_pos)

# First move 20 cm above cube
approach = [
    cube_pos[0],
    cube_pos[1],
    cube_pos[2] + 0.20
]

print("Moving above cube...")
move_target(approach)

# Then lower until sensor sees cube
lower_until_cube()

print("Finished test")