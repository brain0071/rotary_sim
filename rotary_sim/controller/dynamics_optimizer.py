import os
import numpy as np
import sys

from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
from copy import copy
import time



class FF_PI_Dynamics:

    def __init__(self, robot, k_acc_p, acc_ref_max, kp_pid, ki_pid, kd_pid, vel_err_prev, vel_err_dot, d_filter_alpha, vel_err_int, vel_err_int_min, vel_err_int_max, dynamics_dt):
        
        
        self.robot = robot
        self.k_acc_p = k_acc_p
        self.acc_ref_max = acc_ref_max
        
        self.kp_pid = kp_pid 
        self.ki_pid = ki_pid  
        self.kd_pid = kd_pid 
        self.vel_err_prev = vel_err_prev  
        self.vel_err_dot = vel_err_dot
        self.d_filter_alpha = d_filter_alpha
        
        self.vel_err_int = vel_err_int
        self.vel_err_int_min = vel_err_int_min
        self.vel_err_int_max = vel_err_int_max
        self.dynamics_dt = dynamics_dt
        
        self.u_max = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.u_min = np.array([-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
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
        
        """
        Cascaded velocity control.

        Outer loop:
            velocity error -> desired acceleration

        Inner loop:
            feedforward dynamics compensation + PID feedback

        Output:
            normalized force/moment command in [-1, 1]
        """

        # -----------------------------------
        # Model parameters
        # -----------------------------------
        max_fm = np.array(self.robot.max_force_moment, dtype=float)
        quad_damp = np.array(self.robot.quadratic_damp, dtype=float)
        eff_inertia = self.get_effective_inertia_terms()

        # -----------------------------------
        # Outer-loop P control:
        # vel error -> acceleration reference
        # -----------------------------------
        vel_err = self.vel_ref - initial_state

        acc_ref = self.k_acc_p * vel_err
        acc_ref = np.clip(acc_ref, -self.acc_ref_max, self.acc_ref_max)

        # -----------------------------------
        # Feedforward dynamics compensation:
        #
        # tau_ff = M_eff * acc_ref + D(v)
        # u_ff   = tau_ff / tau_max
        # -----------------------------------
        drag_term = quad_damp * np.abs(initial_state) * initial_state

        u_ff = (eff_inertia * acc_ref + drag_term) / max_fm

        # -----------------------------------
        # PID feedback
        # -----------------------------------
        if self.dynamics_dt > 1e-9:

            # Integral term with anti-windup clipping
            self.vel_err_int += vel_err * self.dynamics_dt
            self.vel_err_int = np.clip(
                self.vel_err_int,
                self.vel_err_int_min,
                self.vel_err_int_max
            )

            # Derivative term
            vel_err_dot_raw = (vel_err - self.vel_err_prev) / self.dynamics_dt

            # Low-pass filtered derivative
            self.vel_err_dot = (
                self.d_filter_alpha * vel_err_dot_raw
                + (1.0 - self.d_filter_alpha) * self.vel_err_dot
            )

            self.vel_err_prev = vel_err.copy()

        else:
            self.vel_err_dot = np.zeros_like(vel_err)

        u_pid = (
            self.kp_pid * vel_err
            + self.ki_pid * self.vel_err_int
            + self.kd_pid * self.vel_err_dot
        )

        # -----------------------------------
        # Total normalized input
        # -----------------------------------
        u_cmd_unsat = u_ff + u_pid

        # Actuator saturation
        u_cmd = np.clip(u_cmd_unsat, self.u_min, self.u_max)

        return u_cmd
        

        