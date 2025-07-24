import numpy as np
from rotary_controller.controller.mpc import MPC
from rotary_controller.controller.robot import ROBOT

class ROS_MPC:

    def __init__(self, mass, inertia, add_mass, quadratic_damp, max_force_moment, max_accel,
                 n_nodes, dt, q_cost, r_cost, exp_type):
       
        self.robot = ROBOT(mass, inertia, add_mass, quadratic_damp,  max_force_moment, max_accel)   
        self.mpc = MPC(self.robot, dt, n_nodes, q_cost, r_cost, exp_type)
        
    def set_pos(self, z):
        self.robot.set_pos(z)

    def set_att(self, att):
        self.robot.set_att(att)

    def set_rate(self, rate):
        self.robot.set_rate(rate)  
        
    def set_reference(self, x_ref):
        return self.mpc.set_reference(x_ref)
    
    def optimize(self):
        u = self.mpc.optimize()    
        return u

    def simulate(self, dt, az, u):
        self.mpc.simulate(dt, az, u)

    def get_current_real_state(self):
        return self.mpc.get_real_state()
    
    def get_current_sim_state(self):
        return self.mpc.get_sim_state()