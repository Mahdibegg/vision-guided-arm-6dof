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

cube = sim.getObject("/Object")

actual_cube_position = sim.getObjectPosition(
    cube,
    robot_base
)

print(
    "Actual cube centre:",
    [round(x, 4) for x in actual_cube_position]
)

# -----------------------
# GET RGB IMAGE
# -----------------------

image, resolution = sim.getVisionSensorImg(sensor)

width, height = resolution

frame = np.frombuffer(image, dtype=np.uint8)
frame = frame.reshape(height, width, 3)

frame = cv2.flip(frame, 0)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# -----------------------
# GET DEPTH IMAGE
# -----------------------

depth_bytes, depth_resolution = sim.getVisionSensorDepth(sensor, 1)

depth_array = array.array("f")
depth_array.frombytes(depth_bytes)

depth = np.array(depth_array, dtype=np.float32)
depth = depth.reshape(height, width)

depth = cv2.flip(depth, 0)

# -----------------------
# CAMERA PROPERTIES
# -----------------------

view_angle = sim.getObjectFloatParam(
    sensor,
    sim.visionfloatparam_perspective_angle
)

horizontal_fov = view_angle

vertical_fov = 2 * math.atan(
    (height / width) * math.tan(horizontal_fov / 2)
)

cx = width / 2
cy = height / 2

fx = width / (2 * math.tan(horizontal_fov / 2))
fy = height / (2 * math.tan(vertical_fov / 2))

# Camera -> Robot transform
m = sim.getObjectMatrix(sensor, robot_base)


# -----------------------
# PIXEL -> ROBOT XYZ
# -----------------------

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

    print("\n--- CLICKED POINT ---")
    print("Pixel:", u, v)
    print("Depth:", D)

    print(
        "Camera XYZ:",
        round(x_cam, 4),
        round(y_cam, 4),
        round(z_cam, 4)
    )

    print(
        "Robot XYZ:",
        round(x_robot, 4),
        round(y_robot, 4),
        round(z_robot, 4)
    )

    return [x_robot, y_robot, z_robot]


# -----------------------
# MOUSE CALLBACK
# -----------------------

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        robot_point = pixel_to_robot(x, y)

        approach = [
            robot_point[0],
            robot_point[1],
            robot_point[2] + APPROACH_HEIGHT
        ]

        print("Approach position:", approach)

        move_target_smoothly(
            target,
            robot_base,
            approach
        )

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

cv2.namedWindow("CoppeliaSim Camera")
cv2.setMouseCallback("CoppeliaSim Camera", mouse_callback)

while True:

    cv2.imshow("CoppeliaSim Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()