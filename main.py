################################################################################
#
#                       THIS THE TITLE OF THE CODE
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

p.connect(p.GUI)




