
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

    def ricatti_equation(self, time):
        time_rev = time[::-1]
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
        return res
        