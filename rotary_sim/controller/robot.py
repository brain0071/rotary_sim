#
import numpy as np
from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
import math

class ROBOT:
    
    def __init__(self, mass, inertia, add_mass, quadratic_damp, max_force_moment, max_velocity, max_angular_velocity):
        
        self.mass = mass
        self.inertia = inertia
        self.add_mass = add_mass
        self.quadratic_damp = quadratic_damp
        self.max_force_moment = max_force_moment
        self.max_velocity = max_velocity 
        self.max_angular_velocity = max_angular_velocity

        self.sim_pos = np.zeros((3,))
        self.sim_att = np.array([1.0, 0.0, 0.0, 0.0])  
        self.sim_vel = np.zeros((3,))
        self.sim_rate = np.zeros((3,))
    
    def get_sim_kinematics_state(self):
        x_sim = np.concatenate((self.sim_pos, self.sim_att))
        return x_sim
    
    def get_sim_dynamics_state(self):
        x_sim = np.concatenate((self.sim_vel, self.sim_rate))
        return x_sim

    def p_kinematics(self, x, velocity):
        return np.dot(q_to_rot_mat(x[3:7]), velocity)

    def q_kinematics(self, x, angular):
        return 1 / 2 * np.dot(skew_symmetric(angular), x[3:7])

    def kinematics_update(self, dt, kinematics_opt_u):
        
        # x y z qw qx qy qz
        x = self.get_sim_kinematics_state()
        
        k1 = np.concatenate((self.p_kinematics(x, kinematics_opt_u[0:3]), self.q_kinematics(x, kinematics_opt_u[3:6])))
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(7)]
        k2 = np.concatenate((self.p_kinematics(x_aux, kinematics_opt_u[0:3]), self.q_kinematics(x_aux, kinematics_opt_u[3:6])))
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(7)]

        k3 = np.concatenate((self.p_kinematics(x_aux, kinematics_opt_u[0:3]), self.q_kinematics(x_aux, kinematics_opt_u[3:6])))
        x_aux = [x[i] + dt / 2 * k3[i] for i in range(7)]
        k4 = np.concatenate((self.p_kinematics(x_aux, kinematics_opt_u[0:3]), self.q_kinematics(x_aux, kinematics_opt_u[3:6])))
        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(7)]

        self.sim_pos = x[0:3]
        self.sim_att = x[3:7]
        
        

    def v_dynamics(self, x, u):
        
        du = (u[0] * self.max_force_moment[0] - (self.quadratic_damp[0] * np.abs(x[0]) * x[0])) / (self.mass[0] + self.add_mass[0])
        dv = (u[1] * self.max_force_moment[1] - (self.quadratic_damp[1] * np.abs(x[1]) * x[1])) / (self.mass[0] + self.add_mass[1])
        dw = (u[2] * self.max_force_moment[2] - (self.quadratic_damp[2] * np.abs(x[2]) * x[2])) / (self.mass[0] + self.add_mass[2])
        
        return np.array([du, dv, dw])

    def r_dynamics(self, x, u):
        dp = (u[0] * self.max_force_moment[3] - (self.quadratic_damp[3] * np.abs(x[3]) * x[3])) / (self.inertia[0] + self.add_mass[3])
        dq = (u[1] * self.max_force_moment[4] - (self.quadratic_damp[4] * np.abs(x[4]) * x[4])) / (self.inertia[1] + self.add_mass[4])
        dr = (u[2] * self.max_force_moment[5] - (self.quadratic_damp[5] * np.abs(x[5]) * x[5])) / (self.inertia[2] + self.add_mass[5])
        return np.array([dp, dq, dr])

        
    def dynamics_update(self, dt, dynamics_opt_u):
        
        x = self.get_sim_dynamics_state()
        
        k1 = np.concatenate((self.v_dynamics(x, dynamics_opt_u[0:3]), self.r_dynamics(x, dynamics_opt_u[3:6])))
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(6)]
        k2 = np.concatenate((self.v_dynamics(x_aux, dynamics_opt_u[0:3]), self.r_dynamics(x_aux, dynamics_opt_u[3:6])))
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(6)]

        k3 = np.concatenate((self.v_dynamics(x_aux, dynamics_opt_u[0:3]), self.r_dynamics(x_aux, dynamics_opt_u[3:6])))
        x_aux = [x[i] + dt / 2 * k3[i] for i in range(6)]
        k4 = np.concatenate((self.v_dynamics(x_aux, dynamics_opt_u[0:3]), self.r_dynamics(x_aux, dynamics_opt_u[3:6])))
        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(6)]

        self.sim_vel = x[0:3]
        self.sim_rate = x[3:6]
        
        
        
        
  
        

