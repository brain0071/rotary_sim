import os
import numpy as np
import sys

from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
from copy import copy
import time



class FF_PI_Dynamics:

    def __init__(self, robot, k_acc_p, acc_ref_max, dynamics_dt):
        
        
        self.robot = robot
        self.k_acc_p = k_acc_p
        self.acc_ref_max = acc_ref_max
        self.dynamics_dt = dynamics_dt
        
        self.u_max = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.u_min = np.array([-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        self.vel_ref = np.zeros(6, dtype=float)
   
 
    def set_reference_state(self, x_target):
        
        self.vel_ref = x_target

    def run_optimize(self, initial_state, last_u, sim_a):
        
        # -----------------------------------
        # Outer-loop P control:
        # vel error -> acceleration reference
        # -----------------------------------
        vel_err = self.vel_ref - initial_state

        acc_ref = self.k_acc_p * vel_err
        acc_ref = np.clip(acc_ref, -self.acc_ref_max, self.acc_ref_max)
        
        
        force_u = (acc_ref[0] - sim_a[0]) * self.robot.mass[0] + last_u[0] * self.robot.max_force_moment[0]
        force_v = (acc_ref[1] - sim_a[1]) * self.robot.mass[0] + last_u[1] * self.robot.max_force_moment[1]
        force_w = (acc_ref[2] - sim_a[2]) * self.robot.mass[0] + last_u[2] * self.robot.max_force_moment[2]
        
        force_p = (acc_ref[3] - sim_a[3]) * self.robot.inertia[0] + last_u[3] * self.robot.max_force_moment[3]
        force_q = (acc_ref[4] - sim_a[4]) * self.robot.inertia[1] + last_u[4] * self.robot.max_force_moment[4]
        force_r = (acc_ref[5] - sim_a[5]) * self.robot.inertia[2] + last_u[5] * self.robot.max_force_moment[5]
        

        uu = force_u / self.robot.max_force_moment[0]
        uv = force_v / self.robot.max_force_moment[1]
        uw = force_w / self.robot.max_force_moment[2]
        
        up = force_p / self.robot.max_force_moment[3]
        uq = force_q / self.robot.max_force_moment[4]
        ur = force_r / self.robot.max_force_moment[5]
        
        if uu > 0.6:
            uu = 0.6

        if uu < -0.6:
            uu = -0.6
            
        if uv > 0.6:
            uv = 0.6

        if uv < -0.6:
            uv = -0.6
            
        if uw > 0.6:
            uw = 0.6

        if uw < -0.6:
            uw = -0.6
            
        if up > 0.6:
            up = 0.6

        if up < -0.6:
            up = -0.6
            
        if uq > 0.6:
            uq = 0.6

        if uq < -0.6:
            uq = -0.6
            
        if ur > 0.6:
            ur = 0.6

        if ur < -0.6:
            ur = -0.6
        
        u_indi = np.array([uu, uv, uw, up, uq, ur])
       

        return u_indi
        

        