---
title: 'EM算法'
pubdate: 2026-03-12
category: '机器学习'
--- 

## 引入

在我们的概率模型中，有时观察及所要，但很多时候模型中还隐含了很多的潜在变量。对于这一类情况，我们就不能简单的直接使用极大似然等估计方法，而EM算法就是适用于含有隐变量问题的极大似然估计法。

我们以掷硬币问题为例，假设现在有两枚硬币记为 $A$ 和 $B$ ，两枚硬币正面朝上的概率分别记为 $\theta_A, \theta_B$ 。现在我们要这两个参数的值，显然我们需要进行掷硬币实验，如果我们对于两枚硬币各进行 $n$ 轮掷硬币实验，每轮实验进行 $k$ 次，A硬币第 $i$ 轮实验结果记为 $\{a_i^{(1)}, a_i^{(2)}, \cdots, a_i^{(n)}\}$ ，通过极大似然估计法，可以很容易得到:

$$
\theta_A = \frac{\sum_{i = 1}^n\sum_{j = 1}^kI(a_i^{(k)} = 1)}{nk}
$$

但是考虑这样一种情况，将两枚硬币的实验混在一起，共进行 $k$ 次 $n$ 轮掷硬币实验， 得到 $\{x_i^{(1)}, x_i^{(2)}, \cdots, x_i^{(n)}\}$ 这样我们就不能直接使用极大似然法进行估计了。这里我们就要引入一个隐变量 $Z$ ，代表了所使用的硬币序列 $\{z_1, z_2, \cdots, z_k\}$ ,其中使用A硬币时 $Z = 1$ ，否则 $Z = 0$ 。

为了解决这一个问题，我们考虑一个地带的方法来进行求解，首先我们为两个参数假定一个初值 $\theta_A^1, \theta_B^1$ ，这样我们就可以得出在这样的初值假设下两枚硬币被使用的概率 

$$
\begin{split}
P_A^1 = \frac{(\theta_A^1)^{N_i} (1 - \theta_A^1)^{N_i}}{(\theta_A^1)^{N_i} (1 - \theta_A^1)^{N_i} + (\theta_B^1)^{N_i} (1 - \theta_B^1)^{N_i}}\\
P_B^1 = \frac{(\theta_B^1)^{N_i} (1 - \theta_B^1)^{N_i}}{(\theta_A^1)^{N_i} (1 - \theta_A^1)^{N_i} + (\theta_B^1)^{N_i} (1 - \theta_B^1)^{N_i}}
\end{split}
$$

其中 $N_i$ 表示第 $i$ 轮实验中硬币正面朝上的次数。，对于 $k$ 个概率值求均值，得到了这个概率之后，我们可以进一步用这个概率，估计得到新的参数值，重复这个过程直到参数收敛。

通过这个例子，我们来总结EM算法的过程：

对于观测变量 $Y$ ，隐变量 $Z$ 联合分布 $P(Y , Z|\theta) $ ，条件分布 $P(Z|Y, \theta)$

* 选择参数的初值 $\theta^{(0)}$
* E步：记 $\theta^{(i)}$ 为第 $i$ 次迭代的估计值， 在第 $i + 1$ 次迭代的E步计算：

$$
\begin{split}
Q(\theta, \theta^{(i)}) & = E_Z[\log P(Y, Z |\theta)|Y, \theta^{(i)}]\\
& = \sum_Z \log P(Y, Z|\theta)P(Z|Y, \theta^{(i)})
\end{split}
$$

* M步：求使 $Q(\theta, \theta^{(i)})$ 极大化的 $\theta$ ，则：
  
$$
\theta^{(i + 1)} = \arg\max_\theta Q(\theta, \theta^{(i)})
$$
* 重复这两步直到结果收敛。

请注意，对于任意的初值，EM算法都可以得到收敛的结果，但是算法是对于初值是敏感的，因为EM算法的优化对象不是凸函数，会收敛到局部极值点。

## 算法推导

对于一个含有隐变量的概率模型，我们要通过极大化对数似然函数的方式来得到参数的估计值，即：

$$
\begin{split}
\theta^* & = \arg \max_\theta L(\theta)\\
&  =\arg \max_\theta  \log P(Y | \theta)\\
& = \arg\max_\theta \log\sum_Z P(Y, Z|\theta)\\
& = \arg\max_\theta \log\sum_Z P(Y | Z, \theta) P(Z|\theta)
\end{split}\
$$

这个优化问题的难点就在于 $Z$ 的分布是完全未知且要求一个和的对数。前面通过迭代的方式来求这个极大值，现在我们来说明迭代的方法是有效的。

假设第 $i$ 次迭代得到的结果是 $\theta^{(i)}$ ，那么：

$$
L(\theta) - L(\theta^{(i)}) = \log \left(\sum_Z P(Y|Z, \theta)P(Z|\theta)\right) - \log P(Y|\theta^{(i)})
$$

利用凹函数的Jensen不等式放缩得到：

$$
\begin{split}
L(\theta) - L(\theta^{(i)}) & = \log \left(\sum_ZP(Z|Y, \theta^{(i)})\frac{ P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta^{(i)})}\right) - \log P(Y|\theta^{(i)})\\
& \geq \sum_ZP(Z|Y, \theta^{(i)})\log \left(\frac{ P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta^{(i)})}\right) - \log P(Y|\theta^{(i)})\\
& = \sum_ZP(Z|Y, \theta^{(i)})\log \left(\frac{ P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta^{(i)}) P(Y|\theta^{(i)})}\right)
\end{split}
$$

令 $B(\theta, \theta^{(i)}) = L(\theta^{(i)}) + \sum_ZP(Z|Y, \theta^{(i)})\log \left(\frac{ P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta^{(i)}) P(Y|\theta^{(i)})}\right)$ ，则 $B(\theta^{(i)}, \theta)$ 是对数似然函数的下界。为了实现对数似然函数的最大化，可以等价的转化为 $B(\theta, \theta^{(i)})$ 的最大化，即：

$$
\begin{split}
\theta^{(i+1)}  & = \arg\max_\theta B(\theta, \theta^{(i)})\\
& = \arg\max_\theta \left[L(\theta^{(i)}) + \sum_ZP(Z|Y, \theta^{(i)})\log \left(\frac{ P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta^{(i)}) P(Y|\theta^{(i)})}\right) \right]\\
& = \arg\max_\theta \sum_ZP(Z|Y,\theta^{(i)})\log\left(P(Y|Z, \theta) P(Z|\theta)\right)\\
& = \arg\max_\theta \sum_Z P(Z|Y, \theta^{(i)})\log P(Y,Z|\theta)\\
& = \arg\max_\theta Q(\theta, \theta^{(i)})
\end{split}
$$

不再详细对于算法的收敛性进行证明，通过前人的探索，可以知道的是，对于任意的参数初值，算法最终一定会收敛，但是EM算法仍然是初值敏感的，对于不同的初值会得到不同的收敛结果。为了得到更好的估计结果，通常可以对于初值进行随机采样，从所有的结果中选择最优项。

