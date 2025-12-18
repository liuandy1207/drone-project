import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)

# 1) Add PyBullet built-in URDFs (plane, r2d2, etc)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")  # WORKS NOW ✔

p.setAdditionalSearchPath("./drone")

drone = p.loadURDF(
    "drone1.urdf",
    basePosition=[0, 0, 1],
    useFixedBase=False,
    globalScaling=1
)

p.changeVisualShape(drone, -1, rgbaColor=[1,0,0,1])

# Loop

while True:
    p.stepSimulation()
    time.sleep(0.001)
