
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

    def riccati_equation(self, time, value_T):
        