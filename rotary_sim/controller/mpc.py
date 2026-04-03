import numpy as np
from rotary_sim.controller.kinematics_optimizer import MPC_Kinematics_Optimizer
from rotary_sim.controller.dynamics_optimizer import FF_PI_Dynamics

class MPC:

    def __init__(self, robot, kinematics_n_nodes, kinematics_q_cost, kinematics_r_cost, kinematics_t_horizon, k_acc_p, acc_ref_max, kp_pi, ki_pi, vel_err_int_min, vel_err_int_max, dynamics_dt):
        
        self.robot = robot
        self.kinematics_opt = MPC_Kinematics_Optimizer(robot, kinematics_n_nodes, kinematics_q_cost, kinematics_r_cost, kinematics_t_horizon)
        self.dynamics_opt = FF_PI_Dynamics(robot, k_acc_p, acc_ref_max, kp_pi, ki_pi, vel_err_int_min, vel_err_int_max, dynamics_dt)
            
            
    def get_sim_kinematics_state(self):
        return self.robot.get_sim_kinematics_state()
    
    def set_kinematics_reference(self, ref):
        return self.kinematics_opt.set_reference_state(ref)
   
    def kinematics_optimize(self):
       
        robot_kinematics_state = self.get_sim_kinematics_state()    
        out = self.kinematics_opt.run_optimize(robot_kinematics_state)
        return out
    
    def kinematics_simulate(self, kinematics_dt, kinematics_opt_u):
        self.robot.kinematics_update(kinematics_dt, kinematics_opt_u)

    # dynamics
    def get_sim_dynamics_state(self):
        return self.robot.get_sim_dynamics_state()
    
    def set_dynamics_reference(self, ref):
        return self.dynamics_opt.set_reference_state(ref)
       
    def dynamics_optimize(self):   
        robot_dynamics_state = self.get_sim_dynamics_state()
        out = self.dynamics_opt.run_optimize(robot_dynamics_state)
        return out
    
    def dynamics_simulate(self, dynamics_dt, dynamics_opt_u):
        self.robot.dynamics_update(dynamics_dt, dynamics_opt_u)