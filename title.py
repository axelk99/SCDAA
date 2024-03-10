import torch
import numpy as np

class LQR:
    def __init__(self, H, M, x, sigma, C, D, T, R):
        self.H = H
        self.M = M
        self.x = x
        self.s = sigma
        self.C = C
        self.D = D
        self.T = T
        self.R = R

    def solve_ricatti_ode(self, time):
        time_rev = torch.flip(time, [0])
        L = len(time)
        res = [self.R]
        D_inv = np.linalg.inv(self.D)

        for i in range(L-1):
            S = res[-1]
            delta = time_rev[i] - time_rev[i+1]
            if i == 0:
                print(delta)
            S_new = ( (self.H.T) @ S - S @ self.M @ D_inv @ self.M @ S + self.C + S ) * delta + S

            res.append(S_new)

        return torch.stack(res)
        
    def calculate_value(self, time, space):
        N = len(time)
        u = torch.zeros(N)

        dt = (time[-1] - time[0]) / (N-1)

        S = self.solve_ricatti_ode(time) 

        for i in range(N):
            x = space[i]

            u[i] = x @ S[i] @ x.T

            for j in range(N-i):
                
                temp = torch.reshape(self.s, (2,1)) @ torch.reshape(self.s, (1,2))    
                u[i] += torch.trace(temp @ S[i+j]) * dt

        return u
    
    def calculate_control(self, time, space):
        N = len(time)
        sol = self.solve_ricatti_ode(time) 
        a_star = [-1.0 * torch.linalg.inv(self.D) @ self.M.T @ sol[i] @ space[i].T for i in range(N)]

        return a_star
