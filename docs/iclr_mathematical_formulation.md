# Mathematical Formulation for the ICLR Manuscript

The ICLR manuscript should explicitly define the independent solve accuracy used by the reuse gate.

Let the target solve-probe set be

$$
D_T^{\mathrm{solve}}=\{(x_j,y_j)\}_{j=1}^{m}.
$$

For a frozen candidate skill $s_i$ with parameters $\theta_i$ and prediction function $f(\cdot;\theta_i)$, define

$$
\boxed{
A(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\mathbf{1}\!\left[\left|f(x_j;\theta_i)-y_j\right|\leq\epsilon\right]
}
$$

where $\mathbf{1}[\cdot]$ is the indicator function, $m=64$, and $\epsilon=0.5$ in the current implementation. Thus $A(T,s_i)\in[0,1]$ is the empirical fraction of independent solve-probe examples predicted within the declared tolerance.

The reuse gate requires both frozen compatibility and independent solve accuracy:

$$
P(T\mid s_i)\geq0.90
\quad\text{and}\quad
A(T,s_i)\geq0.85.
$$

The compatibility score is defined from frozen mean-squared error as

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\left(f(x_j;\theta_i)-y_j\right)^2
$$

and

$$
P(T\mid s_i)=\exp\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

The two quantities measure different properties. $P(T\mid s_i)$ is a continuous similarity score based on frozen MSE, whereas $A(T,s_i)$ is an independent empirical solve-accuracy check based on an absolute-error tolerance. Therefore $A(T,s_i)$ should not be described as the probability that skill $s_i$ solves task $T$.

## Controller rule

Let $\tau_{\mathrm{solve}}=0.90$ and $\tau_{\mathrm{clone}}=0.15$. If $s^*$ is the highest-scoring previously acquired skill, the controller uses

$$
\mathrm{action}(T,s^*)=
\begin{cases}
\mathrm{reuse} & \text{if } P(T\mid s^*)\geq\tau_{\mathrm{solve}} \text{ and } A(T,s^*)\geq0.85,\\
\mathrm{clone} & \text{if } P(T\mid s^*)\geq\tau_{\mathrm{clone}} \text{ and the reuse condition is not satisfied},\\
\mathrm{scratch} & \text{otherwise.}
\end{cases}
$$

When cloning is selected, the target starts from the parent's parameters:

$$
\theta_T^{(0)}=\theta_{s^*}.
$$

This formulation makes the decision rule and the independent solve check explicit and reproducible.

## GitHub rendering conventions

All display equations in this document use `$$` delimiters on separate lines. Multiline environments must remain completely inside the display block, and row breaks use `\\`. Greek symbols and mathematical operators are written with LaTeX commands rather than raw Unicode symbols. These conventions follow the repository's mathematical notation guide.
