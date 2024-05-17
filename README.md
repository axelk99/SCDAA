All results can be found in the Jupyter notebook main.ipynb. 
The exercise are clearly labeled and code to get generate plots and results is there.

The LQR class containing functions for exercise 1.1. and 1.2. are in MC.py.

The neural networks are defined in NN.py.

Functions for plots are in plot_class.py.

Let d = 1. Consider now the continuous-time interpolation of the Euler-Maruyama scheme:
$$dY_t^\lambda=-h\left(Y_{k_\lambda\left(t\right)}^\lambda\right)dt+\sqrt{\frac{2}{\beta}}dW_t$$
Prove that $\lim\underset{\lambda\rightarrow0}{W_2}\left(\mu_t,\nu_t^\lambda\right)=0$, where $\mu_t=Law\left(X_t\right)$ and $\nu_t^\lambda=Law\left(Z_t\right)$.
