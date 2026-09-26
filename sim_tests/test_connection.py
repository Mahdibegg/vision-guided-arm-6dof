from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

sensor = sim.getObject("/Vision_sensor")
robot_base = sim.getObject("/UR5")
target = sim.getObject("/UR5/target")

print("Connected!")
print("Vision sensor:", sensor)
print("Robot:", robot_base)
print("Target:", target)