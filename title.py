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

    # def solve_ricatti_ode(self, time):
    #     time_rev = torch.flip(time, [0])
    #     L = len(time)
    #     res = [self.R]
    #     D_inv = np.linalg.inv(self.D)

    #     for i in range(L-1):
    #         S = res[-1]
    #         delta = time_rev[i] - time_rev[i+1]
    #         if i == 0:
    #             print(delta)
    #         S_new = ( (self.H.T) @ S - S @ self.M @ D_inv @ self.M @ S + self.C + S ) * delta + S

    #         res.append(S_new)
    #     return res
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
        #torch.from_numpy(res)
        return torch.from_numpy(res)
        
    def calculate_value(self, time, space):
        value = torch.zeros(len(space))
        
        S = self.solve_ricatti_ode(time) 

        for i in range(len(space) - 1):
            x = space[i, :, :]

            value = torch.matmul(torch.matmul(x.T, S), x)

            integral = 0

            for j in range(len(time) - 1):
                delta_t = time[j+1] - time[j]
                integral += torch.trace(torch.matmul(torch.matmul(self.s, self.s.T), S)) * delta_t


            value[i] = value + integral

        return value
    
    def calculate_control(self, time, space):
        N = len(time)
        sol = self.solve_ricatti_ode(time) 
        a_star = [torch.linalg.inv(self.D) @ self.M.T @ sol[i] for i in range(N)]
        
        #control = torch.zeros(len(space), 2)

        #S = self.solve_ricatti_ode(time) 
        #D_inv = np.linalg.inv(self.D)
        #D_M = torch.matmul(D_inv, self.M.T)
        #D_M_S = torch.matmul(D_M, S)
        #for i in range(len(space)):
        #    x = space[i]
        #    control[i,:] = -torch.matmul(D_M_S, x)

        return a_star
