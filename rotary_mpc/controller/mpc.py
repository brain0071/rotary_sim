
import numpy as np
from rotary_mpc.controller.optimizer import MPC_Optimizer

class MPC:
    
    def __init__(self, robot, dt, n_nodes, q_cost, r_cost, exp_type):
        
        self.exp_type = exp_type
        self.robot = robot
        self.n_nodes = n_nodes
        self.opt = MPC_Optimizer(robot, n_nodes, q_cost, r_cost, dt)
    
    def get_sim_state(self):
        return self.robot.get_sim_state()
    
    def get_real_state(self):
        return self.robot.get_real_state()
    
    def set_reference(self, ref):

        if ref.ndim < 2:
            return self.opt.set_reference_state(ref)
        else:
            return self.opt.set_reference_trajectory(ref)
    
    def optimize(self):
        
        if self.exp_type == "real":
            robot_current_state = self.get_real_state()
        elif self.exp_type == "sim":
            robot_current_state = self.get_sim_state()
        else:
            print("Parameter exp Error.")
    
        out = self.opt.run_optimize(robot_current_state)
        return out
    
    def simulate(self, dt, az, u):
        self.robot.update(dt, az, u)

   
