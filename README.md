**Linear quadratic regulator** <br />
<br />
We consider

$$dX_s = [H X_s + M \alpha_s] ds + \sigma dW_s, \ s \in [t, T],\ X_t = x$$
<br />
Our aim is to minimize
$$J^{\alpha}(t,x) := E^{t,x} [\int_{t}^{T}{X_s^T C X_s + \alpha_s^T D \alpha_s  }ds\ +X_T^T R X_T] $$
J (t,x):=E (Xs CXs +α Dαs)ds+XT RXT t
,
where4 C ≥ 0, R ≥ 0 and D > 0, are given and deterministic and we will assume 2×2.
