import os
import numpy as np
import sys

import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
from copy import copy
import time



class MPC_Dynamics_Optimizer:

    def __init__(self, robot, dynamics_n_nodes, dynamics_q_cost, dynamics_r_cost, dynamics_dt):
        
        self.robot = robot
        self.max_du_u = np.array([0.3, 0.3, 0.3, 0.6, 0.6, 0.6])
        self.min_du_u = np.array([-0.3, -0.3, -0.3, -0.6, -0.6, -0.6])
        self.max_u = np.array([0.6, 0.6, 0.6])
        self.min_u = np.array([-0.6, -0.6, -0.6])
        self.N = dynamics_n_nodes
        self.T = self.N * dynamics_dt
        self.dt = dynamics_dt
        
        # du dv dw p q r uu uv uw
        self.dv = cs.MX.sym("dv", 3)
        self.r = cs.MX.sym("r", 3)
        self.u_x = cs.MX.sym("u_x", 3)
        self.x = cs.vertcat(self.dv, self.r, self.u_x)

        duu = cs.MX.sym("duu")
        duv = cs.MX.sym("duv")
        duw = cs.MX.sym("duw")
        up = cs.MX.sym("up")
        uq = cs.MX.sym("uq")
        ur = cs.MX.sym("ur")
        self.u = cs.vertcat(duu, duv, duw, up, uq, ur)

        self.acados_ocp_solver = {}
        self.acados_models_dir = ("/home/naodai/Workspace/rotary/ros2_ws/src/rotary_sim/acados_models")
    
        ocp = AcadosOcp()
        ocp.dims.N = self.N
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W = np.diag(np.concatenate((dynamics_q_cost, dynamics_r_cost)))
        ocp.cost.W_e = np.diag(dynamics_q_cost)

        self.standard_dynamics = self.robot_dynamics()
        self.model_name = "dynamics"
        stand_acados_model = self.acados_setup_model(self.standard_dynamics(x=self.x, u=self.u)["x_next"], self.model_name)

        ocp.model = stand_acados_model
        nx = stand_acados_model.x.size()[0]
        nu = stand_acados_model.u.size()[0]

        ny = nx + nu
        x_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_ref_e = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        ocp.cost.yref = y_ref
        ocp.cost.yref_e = y_ref_e
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        
        ocp.constraints.x0 = x_ref
        ocp.constraints.lbu = np.array(self.min_du_u)
        ocp.constraints.ubu = np.array(self.max_du_u)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])
        
        # state constraints
        ocp.constraints.idxbx = np.array([6, 7, 8])  
        ocp.constraints.lbx = np.array(self.min_u)
        ocp.constraints.ubx = np.array(self.max_u)
        
        ocp.solver_options.tf = self.T
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 2000

        json_file = os.path.join(self.acados_models_dir, stand_acados_model.name + "_acados_ocp.json")
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

        self.max_opt_time = 0
        self.sum_opt_time = 0
        self.num_opt = 0

    def robot_dynamics(self):

        x_next = cs.vertcat(self.dv_dynamics(), self.r_dynamics(), self.u_x_dynamics())
        return cs.Function("x_next", [self.x, self.u], [x_next], ["x", "u"], ["x_next"])
    
    def dv_dynamics(self):
        # u_x[0] = u_x[0] + u[0] 
       
        self.dv[0] = (self.robot.max_force_moment[0] * (self.u[0] + self.u_x[0])) / (self.robot.mass[0] + self.robot.add_mass[0])
        self.dv[1] = (self.robot.max_force_moment[1] * (self.u[1] + self.u_x[1])) / (self.robot.mass[0] + self.robot.add_mass[1])
        self.dv[2] = (self.robot.max_force_moment[2] * (self.u[2] + self.u_x[2])) / (self.robot.mass[0] + self.robot.add_mass[2])
        return self.dv

    def r_dynamics(self):
        
        self.r[0] = self.r[0] + ((self.u[3] * self.robot.max_force_moment[3] - (self.robot.quadratic_damp[3] * cs.fabs(self.r[0]) * self.r[0])) / (self.robot.inertia[0] + self.robot.add_mass[3])) * self.dt
        self.r[1] = self.r[1] + ((self.u[4] * self.robot.max_force_moment[4] - (self.robot.quadratic_damp[4] * cs.fabs(self.r[1]) * self.r[1])) / (self.robot.inertia[1] + self.robot.add_mass[4])) * self.dt
        self.r[2] = self.r[2] + ((self.u[5] * self.robot.max_force_moment[5] - (self.robot.quadratic_damp[5] * cs.fabs(self.r[2]) * self.r[2])) / (self.robot.inertia[2] + self.robot.add_mass[5])) * self.dt
        return self.r

    def u_x_dynamics(self):
        self.u_x[0] = self.u[0] + self.u_x[0] 
        self.u_x[1] = self.u[1] + self.u_x[1] 
        self.u_x[2] = self.u[2] + self.u_x[2] 
        return self.u_x

    def acados_setup_model(self, dynamics, model_name):

        def fill_in_acados_model(x, u, p, dynamics, name):
            model = AcadosModel()
            model.disc_dyn_expr = dynamics 
            model.x = x
            model.u = u
            model.p = p
            model.name = name
            return model

        acados_models = {}
        dynamics_ = dynamics
        acados_models = fill_in_acados_model(x=self.x, u=self.u, p=[], dynamics=dynamics_, name=model_name)
        return acados_models
        
    def set_reference_state(self, x_target):

        target = np.array((x_target))
        ref_u = np.array([0, 0, 0, 0, 0, 0])
        ref = np.concatenate((target, ref_u), axis=0)
        for j in range(self.N):
            # {x y z qw qx qy qz uu uv uw up uq ur}
            self.acados_ocp_solver.cost_set(j, "yref", ref)

        # {x y z qw qx qy qz}
        self.acados_ocp_solver.cost_set(self.N, "yref", target)

    def run_optimize(self, initial_state):

        x_init = initial_state
        x_init = np.stack(x_init)
        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        t = time.time()
        status = self.acados_ocp_solver.solve()
        opt_time = time.time() - t

        cost = self.acados_ocp_solver.get_cost()
        if opt_time > self.max_opt_time:
            self.max_opt_time = opt_time

        self.sum_opt_time += opt_time
        self.num_opt += 1
        u_opt_acados = np.ndarray((self.N, 6))

        for i in range(self.N):
            u_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")

        next_u = np.array(u_opt_acados[0, :])
        return next_u