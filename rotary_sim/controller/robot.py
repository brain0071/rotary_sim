#
import numpy as np
from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class ROBOT:
    
    def __init__(self, mass, inertia, add_mass, quadratic_damp,  max_force_moment, max_vel):
        
        self.mass = mass 
        self.inertia = inertia
        self.add_mass = add_mass
        self.quadratic_damp = quadratic_damp
        self.max_force_moment = max_force_moment
        self.max_vel = max_vel
        
        self.sim_pos_delta = np.zeros((3,))
        self.sim_pos = np.zeros((3,))
        self.sim_att = np.array([1.0, 0.0, 0.0, 0.0])  
        self.sim_rate = np.zeros((3,))
        
        self.sim_vel = np.zeros((3,))
        self.sim_pos_last = np.zeros((3,))

    def get_sim_state(self):
    
        x_sim = np.concatenate((self.sim_pos_delta, np.concatenate((self.sim_pos, np.concatenate((self.sim_att, self.sim_rate))))))
        return x_sim

    
    def p_dynamics(self, x):
        return np.dot(q_to_rot_mat(x[3:7]), x[7:10])
    
    def q_dynamics(self, x):
        return 1 / 2 * np.dot(skew_symmetric(x[10:13]), x[3:7])
    
        
    def r_dynamics(self, x, u):

        dot_p = (u[3] * self.max_force_moment[3] - (self.quadratic_damp[3] * np.abs(x[10]) * x[10])) / (self.inertia[0] + self.add_mass[3])
        dot_q = (u[4] * self.max_force_moment[4] - (self.quadratic_damp[4] * np.abs(x[11]) * x[11])) / (self.inertia[1] + self.add_mass[4])
        dot_r = (u[5] * self.max_force_moment[5] - (self.quadratic_damp[5] * np.abs(x[12]) * x[12])) / (self.inertia[2] + self.add_mass[5])
        
        return np.array([dot_p, dot_q, dot_r])

        
    def update(self, dt, a, u):
        
        x_sim = self.get_sim_state()
        # x_sim: {delta_z, z, qw, qx, qy, qz, p, q, r}
        # x: {z, qw, qx, qy, qz, vz, p, q, r}
        
        # x_sim: {delta_x, delta_y, delta_z, x, y, z, qw, qx, qy, qz, p, q, r}
        # x: {x, y, z, qw, qx, qy, qz, u, v, w, p, q, r}
        
        
        x = np.concatenate((x_sim[3:10], np.concatenate((self.sim_vel, x_sim[10:13]))))
        
        # RK4 integration
        k1 = np.concatenate((self.p_dynamics(x), self.q_dynamics(x), np.concatenate((a, self.r_dynamics(x, u)))))

        x_aux = [x[i] + dt / 2 * k1[i] for i in range(9)]
        
        k2 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux), np.concatenate((a, self.r_dynamics(x_aux, u)))))
        
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(9)]
        k3 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux), np.concatenate((a, self.r_dynamics(x_aux, u)))))
        
        x_aux = [x[i] + dt / 2 * k3[i] for i in range(9)]
        k4 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux), np.concatenate((a, self.r_dynamics(x_aux, u)))))

        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(9)]
        
        self.sim_pos = x[0:3]
        self.sim_vel = x[7:10]
        self.sim_att = x[3:7]
        self.sim_rate = x[10:13]

        self.sim_pos_delta = x[0:3] - self.sim_pos_last
        self.sim_pos_last = x[0:3]

    def update_indi(self, u):
        
        dot_u = (u[0] * self.max_force_moment[0] - (self.quadratic_damp[0] * np.abs(self.sim_vel[0]) * self.sim_vel[0])) / (self.mass[0] + self.add_mass[0])
        dot_v = (u[1] * self.max_force_moment[1] - (self.quadratic_damp[1] * np.abs(self.sim_vel[1]) * self.sim_vel[1])) / (self.mass[0] + self.add_mass[1])
        dot_w = (u[2] * self.max_force_moment[2] - (self.quadratic_damp[2] * np.abs(self.sim_vel[2]) * self.sim_vel[2])) / (self.mass[0] + self.add_mass[2])
        return np.array([dot_u, dot_v, dot_w])


        

        

