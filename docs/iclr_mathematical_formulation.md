# Mathematical Formulation for the ICLR Manuscript

This document records the mathematical notation used by the ICLR manuscript. Display equations use `$$ ... $$` delimiters on separate lines, following the repository's math-notation conventions.

Let the incoming target task be $T$ and let a previously acquired skill $s_i$ have parameters $\theta_i$ and prediction function $f(\cdot;\theta_i)$.

## Independent solve accuracy

Let the independent solve-probe set be

$$
D_T^{\mathrm{solve}}=\{(x_j,y_j)\}_{j=1}^{m}.
$$

The independent target-solve accuracy is

$$
\boxed{
A(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\mathbf{1}\!\left[
\left|f(x_j;\theta_i)-y_j\right|\leq\epsilon
\right]
}
$$

where $\mathbf{1}[\cdot]$ is the indicator function. In the current implementation, $m=64$ and $\epsilon=0.5$, so $A(T,s_i)\in[0,1]$ is the empirical fraction of independent solve-probe examples whose predictions fall within the declared absolute-error tolerance.

The reuse gate requires both frozen compatibility and independent solve accuracy:

$$
P(T\mid s_i)\geq0.90
\quad\text{and}\quad
A(T,s_i)\geq0.85.
$$

## Compatibility score

For the compatibility probe set $D_T^{\mathrm{probe}}=\{(x_j,y_j)\}_{j=1}^{m}$, define

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\left(f(x_j;\theta_i)-y_j\right)^2.
$$

The frozen compatibility score is

$$
P(T\mid s_i)=\exp\!\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

Thus $P(T\mid s_i)$ is a continuous similarity score derived from frozen mean-squared error, whereas $A(T,s_i)$ is an independent empirical solve check based on an absolute-error tolerance. Neither quantity should be interpreted as a calibrated probability that $s_i$ can solve $T$.

## Controller rule

Let $s_{\mathrm{best}}$ denote the highest-scoring previously acquired skill. Let

$$
\tau_{\mathrm{solve}}=0.90,
\qquad
\tau_{\mathrm{clone}}=0.15.
$$

The controller selects the acquisition action according to

$$
a(T,s_{\mathrm{best}})=
\begin{cases}
\mathrm{reuse} & \text{if } P(T\mid s_{\mathrm{best}})\geq\tau_{\mathrm{solve}} \text{ and } A(T,s_{\mathrm{best}})\geq0.85,\\
\mathrm{clone} & \text{if } P(T\mid s_{\mathrm{best}})\geq\tau_{\mathrm{clone}} \text{ and the reuse condition is not satisfied},\\
\mathrm{scratch} & \text{otherwise.}
\end{cases}
$$

When cloning is selected, the target is initialized from the parent's parameters:

$$
\theta_T^{(0)}=\theta_{s_{\mathrm{best}}}.
$$

For scratch learning, $\theta_T^{(0)}$ is independently initialized.

## Acquisition objective

For target training examples $\{(x_j,y_j)\}_{j=1}^{n}$, the adapted parameters minimize mean squared error:

$$
\theta_T^*=\mathop{\mathrm{argmin}}_{\theta}
\frac{1}{n}\sum_{j=1}^{n}
\left(f(x_j;\theta)-y_j\right)^2.
$$

## Notation conventions

- Use `$$ ... $$` for display equations and `$ ... $` for inline mathematics.
- Use braces for every multi-character superscript or subscript, for example `$s_i$`, `$\theta_i$`, and `$\tau_{\mathrm{solve}}$`.
- Keep multiline environments completely inside the display block.
- Avoid unsupported macros such as `\operatorname` and `\operatorname*`.
- Avoid a literal `*` superscript in controller notation; use `$s_{\mathrm{best}}$` instead.
- Use simple LaTeX commands such as `\mathrm{}` and `\mathop{}` for mathematical labels and operators.
