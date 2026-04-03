import os
import numpy as np
import sys

from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
from copy import copy
import time



class FF_PI_Dynamics:

    def __init__(self, robot, k_acc_p, acc_ref_max, kp_pi, ki_pi, vel_err_int_min, vel_err_int_max, dynamics_dt):
        
        
        self.robot = robot
        self.k_acc_p = k_acc_p
        self.acc_ref_max = acc_ref_max
        self.kp_pi = kp_pi
        self.ki_pi = ki_pi
        self.vel_err_int_min = vel_err_int_min
        self.vel_err_int_max = vel_err_int_max
        self.dynamics_dt = dynamics_dt
        
        self.max_u = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.min_u = np.array([-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        self.vel_ref = np.zeros(6, dtype=float)
   
    def get_effective_inertia_terms(self):

        robot = self.robot
        eff = np.array([robot.mass[0] + robot.add_mass[0], robot.mass[0] + robot.add_mass[1], robot.mass[0] + robot.add_mass[2],
                        robot.inertia[0] + robot.add_mass[3],
                        robot.inertia[1] + robot.add_mass[4],
                        robot.inertia[2] + robot.add_mass[5],
                        ], dtype=float)

        return eff
     
    def set_reference_state(self, x_target):
        
        self.vel_ref = x_target

    def run_optimize(self, initial_state):
        
        max_fm = np.array(self.robot.max_force_moment, dtype=float)
        quad_damp = np.array(self.robot.quadratic_damp, dtype=float)
        eff_inertia = self.get_effective_inertia_terms()
        
        vel_err = self.vel_ref - initial_state
        acc_ref = self.k_acc_p * vel_err
        acc_ref = np.clip(acc_ref, -self.acc_ref_max, self.acc_ref_max)

        # -----------------------------------
        #    u_ff = (M*a_ref + D(v)) / max_force_moment
        # -----------------------------------
        drag_term = quad_damp * np.abs(initial_state) * initial_state
        u_ff = (eff_inertia * acc_ref + drag_term) / max_fm

        # -----------------------------------
        # PI
        # -----------------------------------
        if self.dynamics_dt > 1e-9:
            self.vel_err_int += vel_err * self.dynamics_dt
            self.vel_err_int = np.clip(
            self.vel_err_int,
            self.vel_err_int_min,
            self.vel_err_int_max
        )

        u_pi = self.kp_pi * vel_err + self.ki_pi * self.vel_err_int

   
        u_cmd = u_ff + u_pi
        u_cmd = np.clip(u_cmd, self.u_min, self.u_max)

        return u_cmd
        

        