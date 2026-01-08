import numpy as np
from rotary_sim.controller.mpc import MPC
from rotary_sim.controller.robot import ROBOT

class ROS_MPC:

    def __init__(self, mass, inertia, add_mass, quadratic_damp, max_force_moment, t_horizon, n_nodes, q_cost, r_cost):
       
        self.robot = ROBOT(mass, inertia, add_mass, quadratic_damp,  max_force_moment)   
        self.mpc = MPC(self.robot, t_horizon, n_nodes, q_cost, r_cost)
        
    def set_reference(self, x_ref):
        return self.mpc.set_reference(x_ref)
    
    def optimize(self,state):
        u = self.mpc.optimize(state)    
        return u
        
    #lsz
    def simulate(self, dt, opt_u):
        self.mpc.simulate(dt, opt_u)
    
    def get_current_sim_state(self):
        return self.mpc.get_sim_state()