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
            S_new = ( 2.0 * (self.H.T) @ S - S @ self.M @ D_inv @ self.M @ S + self.C) * delta + S

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
