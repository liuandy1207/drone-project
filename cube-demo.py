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

# CONSTANT FORCE VECTOR  
# Example: push **left** along -X direction with 5 Newtons
force_vector = [-5, 0, 0]   # (Fx, Fy, Fz)

# Run the simulation
while True:
    # Apply force to the cube’s base link (-1 means base link)
    p.applyExternalForce(
        objectUniqueId=cube_body,
        linkIndex=-1,
        forceObj=force_vector,
        posObj=[0, 0, 0],        # apply force at the cube center
        flags=p.WORLD_FRAME      # force defined in world coordinates
    )

    p.stepSimulation()
    time.sleep(0.0005)
