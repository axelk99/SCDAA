import torch
import numpy as np

class LQR:
    def __init__(self, H, M, sigma, C, D, T, R):
        self.H = H
        self.M = M
        self.s = sigma
        self.C = C
        self.D = D
        self.T = T
        self.R = R

    def solve_ricatti_ode(self, time):
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
    
    def monte_carlo_v1(self, time, space, N, N_mc):
        
        C = torch.cat([self.C.unsqueeze(0) for i in range(N_mc)], dim=0)
        D = torch.cat([self.D.unsqueeze(0) for i in range(N_mc)], dim=0)
        D_inv = torch.linalg.inv(self.D)
        D_inv = torch.cat([D_inv.unsqueeze(0) for i in range(N_mc)], dim=0)
        R = torch.cat([self.R.unsqueeze(0) for i in range(N_mc)], dim=0)
        H = torch.cat([self.H.unsqueeze(0) for i in range(N_mc)], dim=0)
        M = torch.cat([self.M.unsqueeze(0) for i in range(N_mc)], dim=0)
        M_T = torch.cat([self.M.T.unsqueeze(0) for i in range(N_mc)], dim=0)
        s = torch.cat([self.s.unsqueeze(0) for i in range(N_mc)], dim=0).unsqueeze(2)
        
        t0 = time[0]
        t_grid = torch.linspace(t0, self.T, N+1)
        dt = (self.T-t0)/N
        

        X = torch.cat([space[0][0].unsqueeze(0) for i in range(N_mc)], dim=0)
        X = X.unsqueeze(2)
        
        z = torch.normal(mean = 0, std = 1, size = [N, N_mc, 1, 1], dtype = torch.float64)

        S = self.solve_ricatti_ode(t_grid)
        #S = torch.cat([S for i in range(N_mc)], dim=0).reshape(N_mc, S.shape[0], S.shape[1], S.shape[2])
        
        J = torch.zeros(N_mc, 1, 1)
        
        for i in range(N):
            #shape X and a = Nmc, 2, 1 (column as in theory)
            Si = torch.cat([S[i].unsqueeze(0) for k in range(N_mc)], dim=0)

            a = -1.0 * (D_inv @ M_T @ Si @ X)

            J += (X.reshape(N_mc,1,2) @ C @ X + a.reshape(N_mc,1,2) @ D @ a) * dt

            X_new = X + (H @ X + M @ a) * dt + s * z[i] * np.sqrt(dt)
            X = X_new

        J += X.reshape(N_mc,1,2) @ R @ X
        return J
