#
import numpy as np
from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
import math

class ROBOT:
    
    def __init__(self, mass, inertia, add_mass, quadratic_damp,  max_force_moment):
        
        self.mass = mass 
        self.inertia = inertia
        self.add_mass = add_mass
        self.quadratic_damp = quadratic_damp
        self.max_force_moment = max_force_moment
        
        self.sim_pos = np.zeros((3,))
        self.sim_vel = np.zeros((3,))
        self.sim_att = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion format: qw, qx, qy, qz
        self.sim_rate = np.zeros((3,))
    

    def get_sim_state(self):
    
        x_sim = np.concatenate((np.concatenate((self.sim_pos, self.sim_att)), np.concatenate((self.sim_vel, self.sim_rate)),))
        return x_sim

    def p_dynamics(self, x):
        return np.dot(q_to_rot_mat(x[3:7]), x[7:10])
    
    def q_dynamics(self, x):
        return 1 / 2 * np.dot(skew_symmetric(x[10:13]), x[3:7])
    
    def v_dynamics(self, x, u):
        dot_u = (u[0] * self.max_force_moment[0] - (self.quadratic_damp[0] * np.abs(x[7]) * x[7])) / (self.mass[0] + self.add_mass[0])
        dot_v = (u[1] * self.max_force_moment[1] - (self.quadratic_damp[1] * np.abs(x[8]) * x[8])) / (self.mass[0] + self.add_mass[1])
        dot_w = (u[2] * self.max_force_moment[2] - (self.quadratic_damp[2] * np.abs(x[9]) * x[9])) / (self.mass[0] + self.add_mass[2])
        
        return np.array([dot_u, dot_v, dot_w])
        
    def r_dynamics(self, x, u):

        dot_p = (u[3] * self.max_force_moment[3] - (self.quadratic_damp[3] * np.abs(x[10]) * x[10])) / (self.inertia[0] + self.add_mass[3])
        dot_q = (u[4] * self.max_force_moment[4] - (self.quadratic_damp[4] * np.abs(x[11]) * x[11])) / (self.inertia[1] + self.add_mass[4])
        dot_r = (u[5] * self.max_force_moment[5] - (self.quadratic_damp[5] * np.abs(x[12]) * x[12])) / (self.inertia[2] + self.add_mass[5])
        
        return np.array([dot_p, dot_q, dot_r])

        
  
        
    def update(self, dt, opt_u):
        
        x = self.get_sim_state()
        
        # RK4 integration
        k1 = np.concatenate((self.p_dynamics(x), self.q_dynamics(x),
                             np.concatenate((self.v_dynamics(x, opt_u), self.r_dynamics(x, opt_u)))))

        # x_aux = x + (dt / 2) * k1
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(13)]

        k2 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux), 
                             np.concatenate((self.v_dynamics(x_aux, opt_u), self.r_dynamics(x_aux, opt_u)))))
        
        # x_aux = x + (dt / 2) * k2
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(13)]

        k3 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux),
                             np.concatenate((self.v_dynamics(x_aux, opt_u), self.r_dynamics(x_aux, opt_u)))))
        
        # x_aux = x + dt * k3
        x_aux = [x[i] + dt / 2 * k3[i] for i in range(13)]

        k4 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux),
                             np.concatenate((self.v_dynamics(x_aux, opt_u), self.r_dynamics(x_aux, opt_u)))))

        # x = x + dt * (1.0 / 6.0 * k1 + 2.0 / 6.0 * k2 + 2.0 / 6.0 * k3 + 1.0 / 6.0 * k4)
        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(13)]

        # Ensure unit quaternion
        # x[3:7] = unit_quat(x[3:7])
        # print(dt)
        
        self.sim_pos = x[0:3]
        self.sim_att = x[3:7]
        self.sim_vel = x[7:10]
        self.sim_rate = x[10:13]


        

        

