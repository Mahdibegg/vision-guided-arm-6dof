from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import array
import math
import time

APPROACH_HEIGHT = 0.20

client = RemoteAPIClient()
sim = client.require("sim")

sensor = sim.getObject("/Vision_sensor")
robot_base = sim.getObject("/UR5")
target = sim.getObject("/UR5/target")

# Use for debugging just incase ----------
cube = sim.getObject("/Object")
actual_cube_position = sim.getObjectPosition(cube, robot_base)

image, resolution = sim.getVisionSensorImg(sensor)

width, height = resolution

# Obtain the depth of the camera
depth_bytes, depth_resolution = sim.getVisionSensorDepth(sensor, 1)
depth_array = array.array("f")
depth_array.frombytes(depth_bytes)
depth = np.array(depth_array, dtype=np.float32)
depth = depth.reshape(height, width)

depth = cv2.flip(depth, 0)

view_angle = sim.getObjectFloatParam(
    sensor,
    sim.visionfloatparam_perspective_angle
)
horizontal_fov = view_angle
vertical_fov = 2 * math.atan((height / width) * math.tan(horizontal_fov / 2))

cx = width / 2
cy = height / 2

fx = width / (2 * math.tan(horizontal_fov / 2))
fy = height / (2 * math.tan(vertical_fov / 2))

# Using CoppeliaSim API to obtain a matrix calibrating sensor and robot_base
m = sim.getObjectMatrix(sensor, robot_base)

# Applying the matrix transformation to a 2d point
def pixel_to_robot(u, v):

    D = float(depth[v, u])

    # Pixel -> Camera XYZ
    x_cam = -(u - cx) * D / fx
    y_cam = -(v - cy) * D / fy
    z_cam = D

    # Camera XYZ -> Robot XYZ
    x_robot = m[0] * x_cam + m[1] * y_cam + m[2] * z_cam + m[3]
    y_robot = m[4] * x_cam + m[5] * y_cam + m[6] * z_cam + m[7]
    z_robot = m[8] * x_cam + m[9] * y_cam + m[10] * z_cam + m[11]

    return [x_robot, y_robot, z_robot]

# Moving the target smoothly to a destination
def move_target_smoothly(target, robot_base, destination, steps=100):

    start = sim.getObjectPosition(target, robot_base)

    for i in range(1, steps + 1):
        t = i / steps

        new_position = [
            start[0] + (destination[0] - start[0]) * t,
            start[1] + (destination[1] - start[1]) * t,
            start[2] + (destination[2] - start[2]) * t
        ]

        sim.setObjectPosition(
            target,
            new_position,
            robot_base
        )

        time.sleep(0.01)