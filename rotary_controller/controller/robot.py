#
import numpy as np
from ocean_mpc.utils.utils import q_to_rot_mat, skew_symmetric, unit_quat
import math
import tf


class ROBOT:
    
    def __init__(self, mass, inertia, add_mass, quadratic_damp,  max_force_moment, max_accel):
        
        self.mass = mass 
        self.inertia = inertia
        self.add_mass = add_mass
        self.quadratic_damp = quadratic_damp
        self.max_force_moment = max_force_moment
        self.max_accel = max_accel

        self.real_z_delta = 0
        self.real_z = 0
        self.real_att = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion format: qw, qx, qy, qz
        self.real_rate = np.zeros((3,))
        
        self.sim_z_delta = 0
        self.sim_z = 0
        self.sim_att = np.array([1.0, 0.0, 0.0, 0.0])  
        self.sim_rate = np.zeros((3,))
        
        self.sim_vz = 0
        self.sim_z_last = 0
    
    def set_pos(self, z):
        self.real_z = z
    
    def set_pos_delta(self, z_delta):
        self.real_z_delta = z_delta

    def set_att(self, att):
        self.real_att = att

    def set_rate(self, rate):
        self.real_rate = rate

    def get_real_state(self):
        x_real = np.concatenate((self.real_z_delta, np.concatenate((self.real_z, np.concatenate((self.real_att, self.real_rate))))))
        return x_real

    def get_sim_state(self):
        x_sim = np.concatenate((self.sim_z_delta, np.concatenate((self.sim_z, np.concatenate((self.sim_att, self.sim_rate))))))
        return x_sim

    def p_dynamics(self, x):
        return np.dot(q_to_rot_mat(x[1:5])[2, 2], x[5]) 

    def q_dynamics(self, x):
        return 1 / 2 * np.dot(skew_symmetric(x[6:9]), x[1:5])
    
    def v_dynamics(self, az):
        return az
        
    def r_dynamics(self, x, u):

        dot_p = (u[1] * self.max_force_moment[3] - (self.quadratic_damp[3] * np.abs(x[6] * x[6]))) / (self.inertia[0] + self.add_mass[3])
        dot_q = (u[2] * self.max_force_moment[4] - (self.quadratic_damp[4] * np.abs(x[7] * x[7]))) / (self.inertia[1] + self.add_mass[4])
        dot_r = (u[3] * self.max_force_moment[5] - (self.quadratic_damp[5] * np.abs(x[8] * x[8]))) / (self.inertia[2] + self.add_mass[5])
        
        return np.array([dot_p, dot_q, dot_r])

        
    def update(self, dt, az, u):
        
        x_sim = self.get_sim_state()
        # x_sim: {delta_z, z, qw, qx, qy, qz, p, q, r}
        # x: {z, qw, qx, qy, qz, vz, p, q, r}
        x = np.concatenate((x_sim[1:6], np.concatenate((self.sim_vz, x_sim[6:9]))))
        
        # RK4 integration
        k1 = np.concatenate((self.p_dynamics(x), self.q_dynamics(x),
                             np.concatenate((self.v_dynamics(az), self.r_dynamics(x, u)))))

        x_aux = [x[i] + dt / 2 * k1[i] for i in range(9)]
        k2 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux), 
                             np.concatenate((self.v_dynamics(az), self.r_dynamics(x_aux, u)))))
        
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(9)]
        k3 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux),
                             np.concatenate((self.v_dynamics(az), self.r_dynamics(x_aux, u)))))
        
        x_aux = [x[i] + dt / 2 * k3[i] for i in range(9)]
        k4 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux),
                             np.concatenate((self.v_dynamics(az), self.r_dynamics(x_aux, u)))))

        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(9)]
        
        
        self.sim_z = x[0]
        self.sim_vz = x[5]
        self.sim_att = x[1:5]
        self.sim_rate = x[6:9]

        self.sim_z_delta = x[0] - self.sim_z_last
        self.sim_z_last = x[0]


        

        

