
import numpy as np
from rotary_sim.controller.optimizer import MPC_Optimizer

class MPC:
    
    def __init__(self, robot, dt, n_nodes, q_cost, r_cost):
        
        self.robot = robot
        self.opt = MPC_Optimizer(robot, n_nodes, q_cost, r_cost, dt)
    
    def get_sim_state(self):
        return self.robot.get_sim_state()
    
    def set_reference(self, ref):
        return self.opt.set_reference_state(ref)
        
    
    def optimize(self):
        robot_current_state = self.get_sim_state()
        out = self.opt.run_optimize(robot_current_state)
        return out
    
    def simulate(self, dt, a, u):
        self.robot.update(dt, a, u)
        
    def simulate_indi(self, u):
        return self.robot.update_indi(u)

   
