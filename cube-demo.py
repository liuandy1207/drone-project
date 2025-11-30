import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

cube_start_pos = [0, 0, 1]        # 1 meter above the plane
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
cube_visual     = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1],
                                     rgbaColor=[1,0,0,1])  # red cube

cube_body = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=cube_collision,
    baseVisualShapeIndex=cube_visual,
    basePosition=cube_start_pos,
    baseOrientation=cube_start_orientation
)

# Run the simulation
while True:
    p.stepSimulation()
    time.sleep(0.0005)