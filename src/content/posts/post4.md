---
title: '拉格朗日对偶性'
pubdate: 2026-03-01 
abstract: '处理最优化问题的一种常见手段'
category: '数学方法'
---

在求解约束最优化问题中，可以利用拉格朗日对偶性将原始问题转化为对偶问题来解决。

对于原始问题： $f(x), c_i(x), h_j(x)$ 是定义在 $\textbf{R}^n$ 上的连续可微函数。考虑最优化问题

$$
\begin{align*}
\min_{x \in \textbf{R}^n} \quad& f(x) \\
s.t. \quad & c_i(x) \leq 0, \quad i = 1, 2, \cdots,k \\
& h_j(x) = 0 \quad j = 1, 2, 3, \cdots, l 
\end{align*}
$$

这个就是最优化的原始问题，为了解决这个问题，我们引入拉格朗日函数：

$$
L(x, \alpha, \beta) = f(x) + \sum_{i = 1}^k \alpha_ic_i(x) + \sum_{j = 1}^l\beta_jh_j(x)
$$

其中 $\alpha_i \geq 0$

考虑关于 $x$ 的函数 $\theta_P(x) = \max_{\alpha, \beta: \alpha_i\geq 0} L(x, \alpha, \beta)$

对于不符合不等式约束的 $x$ ，取对应的 $\alpha_i \to +\infty$ 那么 $\theta_P$的取值就趋于 $+\infty$。同样的，对于不满足等式约束的 $x$ 可以通过对应的 $\beta_j$ 的取值，使 $\theta_P$ 的取值趋于 $+\infty$。

因此可以得到：

$$
\theta_P(x) = \begin{cases}
f(x) \quad x 满足约束条件\\
+\infty \quad \text{else}
\end{cases}
$$

因此原始问题与 $\min_x \theta_P(x)$ 是等价的

## 对偶问题

定义：

$$
\theta_D(\alpha, \beta) = \min_x L(x, \alpha, \beta)
$$

$L(x, \alpha, \beta)$ 关于 $\alpha$ 或 $\beta$是一个仿射函数，而 $L(x, \alpha, \beta)$ 就可以看作一个函数集，而仿射函数集的下确界就是一个凹函数，

问题：

$$
\begin{align*}
\max \quad &\theta_D(\alpha, \beta)\\
s.t. \quad &\alpha_i \geq0
\end{align*}
$$ 

称之为原始问题的对偶问题，实际上可以转化为一个凸优化问题


## 对偶问题解的联系
定义原始问题与对偶问题的最优值  

$$ 
p^* = \min_x\theta_P(x), d^* = \max_{\alpha, \beta: \alpha_i \geq0}\theta_D(\alpha, \beta)
$$

**弱对偶性**：当二者都存在时 $d^* \leq p^*$

**强对偶性**：若 $x^{*}, \alpha^{*}, \beta^{*}$ 分别是原始问题与对偶问题的可行解且 $p^{*} = d^{*}$时， 那么我们的得到的结果就是最优解

**定理**：若 $f(x)$ 与 $c_i(x)$ 均为凸函数，且 $h_j(x)$ 均为仿射函数（即线性的等式约束），那么对偶问题与原始问题的最优值是一致的，并且最优解是等价的。

**KKT条件**：若 $f(x)$ 与 $c_i(x)$ 均为凸函数，且 $h_j(x)$ 均为仿射函数（即线性的等式约束），那么 $x^{*}, \alpha^{*}, \beta^{*}$ 分别是原始问题与对偶问题最优解的充分必要条件是：

$$
\begin{align*}
\nabla_x L(x^{*}, \alpha^{*}, \beta^{*}) = 0\\
\alpha_i^{*}c_i(x^{*}) = 0\\
c_i(x^{*}) \leq0\\
\alpha_i^{*} \geq0\\
h_j(x^{*}) = 0
\end{align*}
$$
