import numpy as np
from rotary_sim.controller.mpc import MPC
from rotary_sim.controller.robot import ROBOT

class ROS_MPC:
    
    

    def __init__(self, mass, inertia, add_mass, quadratic_damp, max_force_moment, max_velocity, max_angular_velocity, 
                 kinematics_n_nodes, kinematics_q_cost, kinematics_r_cost, kinematics_t_horizon,
                 k_acc_p, acc_ref_max, kp_pid, ki_pid, kd_pid, vel_err_prev, vel_err_dot, d_filter_alpha, 
                 vel_err_int, vel_err_int_min, vel_err_int_max, dynamics_dt):
       
        self.robot = ROBOT(mass, inertia, add_mass, quadratic_damp, max_force_moment, max_velocity, max_angular_velocity)
        self.mpc = MPC(self.robot, kinematics_n_nodes, kinematics_q_cost, kinematics_r_cost, kinematics_t_horizon, 
                       k_acc_p, acc_ref_max, kp_pid, ki_pid, kd_pid, vel_err_prev, vel_err_dot, d_filter_alpha, vel_err_int, vel_err_int_min, vel_err_int_max, dynamics_dt)
        

    # kinematics
    def set_kinematics_reference(self, x_ref):
        return self.mpc.set_kinematics_reference(x_ref)
    
    def kinematics_optimize(self):
        delta_u = self.mpc.kinematics_optimize()    
        return delta_u
    
    def kinematics_simulate(self, kinematics_dt, kinematics_opt_u):
        self.mpc.kinematics_simulate(kinematics_dt, kinematics_opt_u)


    def get_kinematics_sim_state(self):
        return self.mpc.get_sim_kinematics_state()
    
    # dynamics
    def set_dynamics_reference(self, x_ref):
        
        return self.mpc.set_dynamics_reference(x_ref)
    
    def dynamics_optimize(self):
        u = self.mpc.dynamics_optimize()    
        return u
    
    def dynamics_simulate(self, dynamics_dt, dynamics_opt_u):
        self.mpc.dynamics_simulate(dynamics_dt, dynamics_opt_u)

    def get_dynamics_sim_state(self):
        return self.mpc.get_sim_dynamics_state()