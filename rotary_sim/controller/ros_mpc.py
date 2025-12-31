import numpy as np
from rotary_sim.controller.mpc import MPC
from rotary_sim.controller.robot import ROBOT

class ROS_MPC:

    def __init__(self, mass, inertia, add_mass, quadratic_damp, max_force_moment, max_vel, n_nodes, mpc_dt, q_cost, r_cost):
       
        self.robot = ROBOT(mass, inertia, add_mass, quadratic_damp,  max_force_moment, max_vel)   
        self.mpc = MPC(self.robot, mpc_dt, n_nodes, q_cost, r_cost)
        
    def set_reference(self, x_ref):
        return self.mpc.set_reference(x_ref)
    
    def optimize(self):
        u = self.mpc.optimize()    
        return u

    def simulate(self, mpc_dt, a, u):
        self.mpc.simulate(mpc_dt, a, u)
        
    # indi sim
    def simulate_indi(self, u):
        return self.mpc.simulate_indi(u)
    
    def get_current_sim_state(self):
        return self.mpc.get_sim_state()