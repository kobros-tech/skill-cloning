# Mathematical Formulation for the ICLR Manuscript

The ICLR manuscript should explicitly define the independent solve accuracy used by the reuse gate.

Let the target solve-probe set be
\[
D_T^{\mathrm{solve}}=\{(x_j,y_j)\}_{j=1}^{m}.
\]
For a frozen candidate skill $s_i$ with parameters $\theta_i$ and prediction function $f(\cdot;\theta_i)$, define
\[
\boxed{
A(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\mathbf{1}\!\left[\left|f(x_j;\theta_i)-y_j\right|\leq\epsilon\right]
}
\]
where $\mathbf{1}[\cdot]$ is the indicator function, $m=64$, and $\epsilon=0.5$ in the current implementation. Thus $A(T,s_i)\in[0,1]$ is the empirical fraction of independent solve-probe examples predicted within the declared tolerance.

The reuse gate requires both frozen compatibility and independent solve accuracy:
\[
P(T\mid s_i)\geq 0.90
\quad\text{and}\quad
A(T,s_i)\geq0.85.
\]

This definition is preferable to describing $A(T,s_i)$ only in prose because it makes the controller fully reproducible and distinguishes the solve check from the compatibility score. The compatibility score measures frozen mean-squared error transformed as $P(T\mid s_i)=\exp[-\operatorname{MSE}(T,s_i)/60]$; $A(T,s_i)$ instead measures the fraction of predictions satisfying an absolute-error tolerance.

## Interpretation

$A(T,s_i)$ is an empirical diagnostic, not a learned probability model. It should not be described as the probability that skill $s_i$ solves task $T$. The notation $A(T,s_i)$ denotes the observed solve accuracy on the independently generated probe set.
