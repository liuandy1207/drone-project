################################################################################
#
#                           DRONE SIMULATION
#
# Authors: Andy Liu, John Chen, Melyssa Choi, Yawen Zheng
################################################################################

################################################################################
#                         venv set up instructions
#-------------------------------------------------------------------------------
# 1. install conda if you don't already have it 
#    you can check if you have it with `conda --version`
# 2. create a conda environment if you don't already have it:
#    `conda create -n pybullet-env python=3.11`
# 3. activate the environment:
#    `conda activate pybullet-env`
# 4. install pybullet:
#    `conda install -c conda-forge pybullet`
# remember to re-activate the environment whenever you want to run the program!
################################################################################

# imports
import pybullet as p        # physics simulator
import pybullet_data        # some default data for pybullet
import time

p.connect(p.GUI)        # start the GUI

# generate background plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

# generate temporary cube
cube_start_pos = [0, 0, 1]        # 1 meter above the plane
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
cube_visual     = p.createVisualShape(p.GEOM_BOX, 
                                      halfExtents=[0.1, 0.1, 0.1],
                                      rgbaColor=[1,0,0,1])  # red cube
cube = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=cube_collision,
    baseVisualShapeIndex=cube_visual,
    basePosition=cube_start_pos,
    baseOrientation=cube_start_orientation
)

# force set up
p.setGravity(0, 0, -9.81)


# run the simulation
while True:
    p.stepSimulation()
    time.sleep(0.005)      # this depends on hardware