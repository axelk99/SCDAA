
class LQR:
    def __init__(self, H, M, x, sigma, C, D, T):
        self.H = H
        self.M = M
        self.x = x
        self.sigma = sigma
        self.C = C
        self.D = D
        self.T = T

    def riccati_equation(self, time, value_T):
        