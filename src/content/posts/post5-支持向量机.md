---
title: '支持向量机一：线性可分问题'
pubdate: 2026-03-03
category: '机器学习'
---

支持向量机（SVM）是一种二类分类模型，通过定义在特征空间上的间隔最大分类器从而有别于感知机模型。

## 线性可分支持向量机与硬间隔最大化

给定一个特征空间上的训练集：

$$
T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\}
$$

其中 $x_i \in \textbf{R}^n$ ， $y_i \in \{+1, -1\}$。

若训练集是线性可分的，我们就可以找到一个分离超平面将所有的实例点分到不同的类去。在之前的感知机的学习中，我们可以通过迭代调整的方式得到最终的分离超平面，但是对于不同的初值，得到的结果是不唯一的。我们通过定义函数间隔与几何间隔的方式来找出最好的分离平面。

### 函数间隔与几何间隔

对于一个分离超平面 $0 = \omega \cdot x + b$， $|\omega \cdot x_i + b|$ 是实例 $x_i$ 到分离平面的距离。一般来说，距离越大说明分类的可信度越高，因此函数间隔就可以定义为：

$$
\hat{\gamma}_i = y_i(\omega \cdot x_i + b)
$$

而超平面关于训练集 $T$ 的函数间隔就定义为 $\hat{\gamma} = \min \hat{\gamma}_i$ ，通过所有实例到超平面距离的最小值来衡量该分类器的好坏

对于距离进行规范化即 $\gamma = \dfrac{\hat{\gamma}}{||\omega||}$ ，就得到了几何间隔，避免了因为系数的等比例放大导致的尺度不统一。

### 间隔最大化

前面讲了，支持向量机模型不同于感知机的最大特点就在于选择了几何间隔最大化的超分离平面，可以将我们的分类问题，转化为如下的优化问题：

$$
\begin{align*}
\max_{\omega, b} \quad & \gamma\\
s.t. \quad & y_i\frac{\omega \cdot x_i + b}{||\omega||} \geq \gamma, \quad i = 1, 2, \cdots, N
\end{align*}
$$

由于 $\hat{\gamma} = ||\omega||\gamma$，并且函数间隔的实际取值并不影响优化结果，不妨取 $\hat{\gamma} = 1$ ，那么原优化问题可以转化为：

$$
\begin{align*}
\max_{\omega, b} \quad & \frac{1}{||\omega||}\\
s.t. \quad & y_i(\omega \cdot x_i + b) \geq 1, \quad i = 1, 2, \cdots, N
\end{align*}
$$

也等价于

$$
\begin{align*}
\min_{\omega, b} \quad & \frac{1}{2}||\omega||^2\\
s.t. \quad & y_i(\omega \cdot x_i + b) \geq 1, \quad i = 1, 2, \cdots, N
\end{align*}
$$

这样，我们就将原问题转化为了一个凸优化问题。求解得到 $\omega^*, b^*$。

### 对偶问题的优化

应用拉格朗日对偶性来解决最优化问题，由于原始问题是凸优化问题，因此可以通过对偶问题来得到原始问题的最优解。

首先给出拉格朗日函数：

$$
L(\omega, b, \alpha) = \frac{1}{2}||\omega||^2 + \sum_{i = 1}^N\alpha_i - \sum_{i = 1}^N\alpha_iy_i(\omega \cdot x_i + b)
$$

其中拉格朗日乘子 $\alpha_i \geq 0$

原始问题的对偶问题即为；

$$
\begin{align*}
\max_{\alpha}\min_{\omega, b} \quad & L(\omega, b, \alpha)\\
s.t. \quad & \alpha_i \geq0
\end{align*}
$$

首先求 $\min_{\omega, b} L(\omega, b, \alpha)$。通过求梯度为零的点，得到：

$$
\begin{split}
\omega - \sum_{i = 1}^N\alpha_iy_ix_i = 0\\
-\sum_{i = 1}^N\alpha_iy_i = 0
\end{split}
$$

代回可以得到：

$$
\begin{split}
\min_{\omega, b} L(\omega, b, \alpha) & = \frac{1}{2}\sum_{i = 1}^N\sum_{j = 1}^N\alpha_i\alpha_jy_iy_jx_i \cdot x_j + \sum_{i = 1}^N\alpha_i - \sum_{i = 1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i \cdot x_j\\
& = \sum_{i = 1}^N\alpha_i - \frac{1}{2}\sum_{i = 1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i \cdot x_j
\end{split}
$$

则对偶问题被转化成：

$$
\begin{split}
\min_{\alpha} \quad & \frac{1}{2}\sum_{i = 1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i \cdot x_j - \sum_{i = 1}^N\alpha_i\\
s.t. \quad & \sum_{i = 1}^N\alpha_iy_i = 0\\
& \alpha_i \geq 0, \quad i = 1, 2, \cdots, N 
\end{split}
$$

处理该优化问题，得到对偶问题的可行解 $\alpha^*$ ，对于原始问题的可行解：

$$
\omega^* = \sum_{i = 1}^N\alpha_iy_ix_i
$$

由于 $\alpha^*$ 必然不是 $0$ （显然最优解存在 $\alpha_j > 0$，由KTKT条件 $y_j(\omega^* \cdot x_j + b^*) - 1 = 0$，可以得到：

$$
b^* = y_j - \sum_{i = 1}^N\alpha_iy_ix_i \cdot x_j
$$

### 支持向量与间隔

我们发现，在模型训练的过程中，最终只有 $\alpha_i > 0$ 的实例点决定了分离超平面。在可视化图形中，这些间隔点是距离分离超平面最近的几个实例点，称为**支持向量**，而由经过支持向量且平行于分离超平面的两个超平面之间构成的空间就称之为该模型的间隔。

## 软间隔最大化

以上构建的模型建立在训练集线性可分的基础上，但线性可分实际上是一个很强的前提，因此考虑引入一个对于间隔的松弛变量 $\zeta$，对于两个类别中的极端个例，允许出现在间隔中，甚至被误分类。

引入松弛变量，原始问题就变为：

$$
\begin{split}
\min \quad & \frac{1}{2}||\omega||^2 + C\sum_{i = 1}^N\zeta_i\\
\text{s.t.} \quad & y_i(\omega\cdot x_i + b) \geq 1 - \zeta_i, \quad i = 1, 2, \cdots, N \\
& \zeta_i \geq 0, i = 1, 2, \cdots, N
\end{split}
$$

在这个问题中 $C$ 是惩罚参数，即模型允许存在误分类，但是目标函数中会对误分类进行惩罚，最终通过最优化达到平衡。

### 对偶问题

该问题求对偶问题的方式基本上是一致的，首先得到原始的对偶问题：

$$
\begin{split}
\max_{\alpha, \beta} \min_{\omega, b, \zeta} \quad & \frac{1}{2}||\omega||^2 + C\sum_{i = 1}^N\zeta_i + \sum_{i = 1}^N \alpha_i(1 - \zeta_i) - \sum_{i = 1}^N\alpha_iy_i(\omega \cdot x_i + b) - \sum_{i = 1}^N \beta_i\zeta_i\\
\text{s.t.} \quad & \alpha_i \geq 0, \beta_i \geq 0
\end{split}
$$

首先求极小值：

$$
\begin{split}
\nabla_\omega L & = \omega - \sum_{i = 1}^N \alpha_iy_ix_i = 0\\
\nabla_b L & = -\sum_{i = 1}^N\alpha_iy_i = 0\\
\nabla_{\zeta_i} L & = C - \alpha_i  - \beta_i= 0
\end{split}
$$

代回可以将原问题转化为：

$$
\begin{split}
\min_{\alpha} \quad & \frac{1}{2}\sum_{i = 1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i \cdot x_j - \sum_{i = 1}^N\alpha_i\\
\text{s.t.} \quad & \sum_{i = 1}^N\alpha_iy_i = 0\\
& C =  \alpha_i + \beta_i\\
& \alpha_i \geq 0, \beta_i \geq 0, i = 1, 2, \cdots, N
\end{split}
$$

通过等式约束消去 $\beta_i$ 来约束变量可以得到

$$
\begin{split}
\min_{\alpha} \quad & \frac{1}{2}\sum_{i = 1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i \cdot x_j - \sum_{i = 1}^N\alpha_i\\
\text{s.t.} \quad & \sum_{i = 1}^N\alpha_iy_i = 0\\
& 0 \leq \alpha_i \leq C
\end{split}
$$

通过解这个最优化问题，得到 $\alpha^*$ ，以线性可分问题类似的方式可以得到 $\omega^*, b^*$