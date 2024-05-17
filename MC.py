import torch
import numpy as np

class LQR:

    def __init__(self, H, M, sigma, C, D, T, R): #define parameters of LQR problem
        self.H = H
        self.M = M
        self.s = sigma
        self.C = C
        self.D = D
        self.T = T
        self.R = R

    def params(self, batch_size): #multidimensional version of parameters
        
        R = torch.cat([self.R.unsqueeze(0) for i in range(batch_size)], dim=0)
        H = torch.cat([self.H.unsqueeze(0) for i in range(batch_size)], dim=0)
        C = torch.cat([self.C.unsqueeze(0) for i in range(batch_size)], dim=0)
        M = torch.cat([self.M.unsqueeze(0) for i in range(batch_size)], dim=0)
        D = torch.cat([self.D.unsqueeze(0) for i in range(batch_size)], dim=0)
        S = torch.cat([self.s.T@self.s for i in range(batch_size)], dim=0).reshape(-1, 2, 2)
        return R, H, C, M, D, S
    
    def solve_ricatti_ode(self, time):
        """
        Backward difference numerical solution of Ricatti ODE
        """

        time_rev = torch.flip(time, [0])
        L = len(time)
        res = [self.R]
        D_inv = torch.linalg.inv(self.D)

        for i in range(L-1):
            S = res[-1]
            delta = time_rev[i] - time_rev[i+1]
            S_new = ( 2.0 * (self.H.T) @ S - S @ self.M @ D_inv @ self.M.T @ S + self.C) * delta + S

            res.append(S_new)
        res = res[::-1]
        return torch.stack(res)
        
    def calculate_value(self, time, space):
        """
        Analytical representation of value function given the current state = (time, space) of the system
        """

        N = len(time)
        u = torch.zeros(N)

        for i in range(N):
            t0 = time[i]
            dt = 0.001 
            t_grid = torch.arange(t0, self.T+dt, dt)
            L = len(t_grid)

            S = self.solve_ricatti_ode(t_grid) 
            x = space[i]

            u[i] = x @ S[0] @ x.T

            for j in range(L):
                
                temp = torch.reshape(self.s, (2,1)) @ torch.reshape(self.s, (1,2))    
                u[i] += torch.trace(temp @ S[j]) * dt

        return u
    
    def calculate_control(self, time, space):
        """
        Analytical representation of optimal control given the current state = (time, space) of the system
        """

        N = len(time)
        a_star = torch.zeros(N, 1, 2)

        for i in range(N):
            t0 = time[i]

            dt = 0.001 
            t_grid = torch.arange(t0, self.T+dt, dt)
            S = self.solve_ricatti_ode(t_grid)

            a = -1.0 * torch.linalg.inv(self.D) @ self.M.T @ S[0] @ space[i].T

            a_star[i] = a.T

        return a_star.squeeze(1)
    
    def mc_high_dim(self, time, space, N, N_mc, a1 = False):
        """
        Value function equals the expected value of optimisation objective J wrt optimal control and system current state
        Expected value is calculated via Monte Carlo simulations, where N_mc trajectories are simultaneously sampled at each t = 1...N
        """

        D_inv = torch.linalg.inv(self.D)
        s = self.s.unsqueeze(2)
        
        t0 = time[0]
        t_grid = torch.linspace(t0, self.T, N+1)
        dt = (self.T-t0)/N

        #shape X and a = Nmc, 2, 1 (column as in theory)
        X = torch.cat([space[0][0].unsqueeze(0) for i in range(N_mc)], dim=0)
        X = X.unsqueeze(2)

        z = torch.normal(mean = 0, std = 1, size = [N, N_mc, 1, 1], dtype = torch.float32)
        
        J = torch.zeros(N_mc, 1, 1)
        a = torch.ones(N_mc, 2, 1, dtype=torch.float32)
        
        S = self.solve_ricatti_ode(t_grid)

        for i in range(N): #loop over time

            if not a1:        
                a = -1.0 * (D_inv @ self.M.T @ S[i] @ X)
                
            J += (X.reshape(N_mc,1,2) @ self.C @ X + a.reshape(N_mc,1,2) @ self.D @ a) * dt

            X_new = X + (self.H @ X + self.M @ a) * dt + s * z[i] * np.sqrt(dt)
            X = X_new

        J += X.reshape(N_mc,1,2) @ self.R @ X
        return J
    
    def error_calculation(self, time, space, time_arr, mc_arr):
        """
        Relative error between analytical value function and value function calculated with MC for a give system state
        """

        e = []
        n1, n2 = len(time_arr), len(mc_arr)
        
        val_analyt = self.calculate_value(time, space)
             
        for i in range(n1):
            for j in range(n2):
                J_arr = self.mc_high_dim(time, space, N = time_arr[i], N_mc = mc_arr[j])
                val_mc1 = torch.mean(J_arr)
                e.append( np.abs(val_analyt.item() - val_mc1.item()) / np.abs(val_analyt.item()) )
        
        return e
    
    def data_processing(self, batch_size, type):
        space = torch.rand(batch_size, 1, 2, dtype=torch.float32) * 6 - 3
        time = torch.rand(batch_size, dtype = torch.float32) * self.T

        if type == 'Control':
            val = self.calculate_control(time, space)
        elif type == 'Value':
            val = self.calculate_value(time, space)

        val1 = ( val - val.mean() ) / val.std()

        space1 = ( space - space.mean() ) / space.std()
        time1 = ( time - time.mean() ) / time.std()

        # Validation
        space_val = torch.rand(batch_size, 1, 2, dtype=torch.float32) * 6 - 3
        time_val = torch.rand(batch_size, dtype = torch.float32) * self.T


        if type == 'Control':
            val_val = self.calculate_control(time_val, space_val)
        elif type == 'Value':
            val_val = self.calculate_value(time_val, space_val)

        val_val = ( val_val - val.mean() ) / val.std()

        space_val = ( space_val- space.mean() ) / space.std()
        time_val = ( time_val - time.mean() ) / time.std()

        val1 = val1.unsqueeze(1)
        val_val = val_val.unsqueeze(1)

        time1 = time1.unsqueeze(1)
        space1 = space1.squeeze(1)

        time_val = time_val.unsqueeze(1)
        space_val = space_val.squeeze(1)

        return time1, space1, time_val, space_val, val1, val_val 