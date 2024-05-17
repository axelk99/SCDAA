**Linear quadratic regulator** <br />
<br />
We consider

$$dX_s = [H X_s + M \alpha_s] ds + \sigma dW_s, \ s \in [t, T],\ X_t = x$$
Our aim is to minimize
$$J^{\alpha}(t,x) := E^{t,x} [\int_{t}^{T}{X_s^T C X_s + \alpha_s^T D \alpha_s  }ds\ +X_T^T R X_T] $$
where C ≥ 0, R ≥ 0 and D > 0, are given and deterministic and we will assume 2×2.<br />
<br />

The value function is $u(t,x) := \underset{\alpha}{inf} J^{\alpha}(t,x)$. We know how to solve the Bellman PDE to obtain that 
$$u(t,x) = x^T S(t) x + \int_{t}^{T}{tr(\sigma \sigma^T S(r))dr}$$
where S is the solution of the Riccati ODE:
$$S^{'}(r) = -2H^TS(r) + S(r)MD^{-1}M^TS(r)-C, \ r \in [t, T], \ S(T) = R $$
Note that solution takes values in the space of 2x2 matrices. The optimal Markov control is 
$$a(t,x) = -D^{-1}M^TS(t)x$$
**Exercise 1.1**<br />
Solve Riccati ODE, calculate value function and optimal control for a give system state using analytical expressions <br />
<br /> **Exercise 1.2**<br />
Run a Monte Carlo simulation of the system with the optimal control and ensure that the solution is converging to the optimal value function obtained in Exercise 1.1.<br />
<br /> **Exercise 2.1**<br />
Create a neural network that approximates the value function and obtain the weights by minimising MSE cost functional given the dataset $u(t_i, x_i), \ i = 1...n$ from Exercise 1.1. <br />
<br /> **Exercise 2.2**<br />
The same as Exercise 2.1 but for optimal control.<br />
<br /> **Exercise 3**<br />
Implement the Deep Galerkin method for the linearisation of the Bellman PDE resulting from taking the constant control $\alpha = (1, 1)^T$ regardless of the state of the system. The deep Galerkin method replaces $u$ by a neural network approximation $u(·, ·; \theta)$, chooses random points from the problem domain $(t^{(i)}, x^{(i))}$ with $i = 1, . . . , N$ and them aims to minimize:

$$R(\theta) := R_{eqn}(\theta) + R_{boundary}(\theta) = \frac{1}{N} \sum_{i=1}^{N}|\partial u (t^{(i)}, x^{(i)};\theta)  + \frac{1}{2} tr(\sigma \sigma^T \partial_{xx} u(t^{(i)}, x^{(i)};\theta) + $$

$$ + (\partial_{x} u(t^{(i)}, x^{(i)};\theta))^T H x^{(i)} + (\partial_{x} u(t^{(i)}, x^{(i)};\theta))^T M \alpha + (x^{(i)})^T C x^{(i)} + \alpha^T D \alpha|^2 + $$

$$ + \frac{1}{N} \sum_{i=1}^{N}| u(T, x^{(i)}; \theta) - (x^{(i)})^T R x^{(i)} | ^ 2 $$
