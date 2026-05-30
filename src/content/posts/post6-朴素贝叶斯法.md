---
title: '朴素贝叶斯法'
pubdate: 2026-03-08
category: '机器学习'
---

在介绍朴素贝叶斯方法之前，首先回顾一下贝叶斯定理。通过全概率公式可以得到：

$$
p(B_k | A) =  \frac{p(A | B_k)\,p(B_k)}{p(A)} =  \frac{p(A | B_k)\,p(B_k)}{\sum_{i = 1}^np(A|B_i)\,p(B_i)}
$$

其中 $P(B_k|A)$ 称为后验概率， $P(B_k)$ 称为先验概率， $P(A|B_k)$ 称为似然函数， $\{B_k\}$ 是相互独立的。

贝叶斯公式实际上简单地揭示了**我们是如何通过观察群体中某一特征**来帮助我们进行分类的。在某一类别中某一特征出现的概率越高，自然的也认为携带该特征的样本更有可能属于这一类别。

## 朴素贝叶斯法

特征向量 $x \in \textbf{R}^n$ ， 输出空间为 $\{c_1, c_2, \cdots, c_K\}$。希望通过训练集进行学习，对于给定的特征向量进行分类工作。

源自贝叶斯公式的要求，我们假设特征向量 $X$ 的 $n$ 个分量是独立的， 且分量 $X^{(j)}$ 可以取到的值有 $s_j$ 个。有独立性假设有：

$$
\begin{split}
P(X = x|Y = c_k) = & P(X^{(1)} = x^{(1)}, \cdots, X^{(n)} = x^{(n)} | Y = c_k)\\
= & \prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k)
\end{split}
$$

对于给定的输入 $x$，通过贝叶斯公式计算后验概率：

$$
\begin{split}
P(Y = c_k | X = x) = & \frac{P(X = x|Y = c_k) \cdot P(Y = c_k)}{\sum_k P(X = x|Y=c_k) \cdot P(Y = c_k)}\\
= & \frac{\prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k) \cdot P(Y = c_k)}{\sum_k P(Y = c_k) \cdot \prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k)}
\end{split}
$$

将后验概率最大化，最大化的结果就是分类器的分类结果，可表示为：

$$
y = \argmax_{c_k} \frac{\prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k) \cdot P(Y = c_k)}{\sum_k P(Y = c_k) \cdot \prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k)}
$$

由于分母 $\sum_k P(Y = c_k) \cdot \prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k)$ 对于不同的 $c_k$是一样的，因此优化问题可以简化为：

$$
y = \argmax_{c_k} \prod_{j = 1}^n P(X^{(j)} = x^{(j)} | Y = c_k) \cdot P(Y = c_k)
$$

## 优化问题的处理

### 参数估计

对于 $P(Y = c_k), P(X^{(j)} = x^{(j)} | Y = c_k)$ 利用极大似然法进行参数估计。

对于训练集 $T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\}$ ，以 $P(Y = c_k)$为例。为了估计 $P(Y)$，可以得到对数似然函数：

$$
\begin{split}
\ln L(P) = & \ln\prod_{i = 1}^NP(Y = y_i)\\
= & \ln\prod_{k = 1}^KP(Y = c_k)^{\sum_{i = 1}^NI(y_i = c_k)}\\
= & \sum_{k = 1}^K \sum_{i = 1}^N I(y_i = c_k)\ln P(Y = c_k)
\end{split}
$$

由于有约束条件 $\sum_{k = 1}^KP(Y = c_k) = 1$，利用拉格朗日乘数法可以得到

$$
P(Y = c_k) = \frac{\sum_{i = 1}^NI(y_i = c_k)}{\lambda}
$$

再反向利用约束条件：

$$
 \sum_{k = 1}^K\frac{\sum_{i = 1}^NI(y_i = c_k)}{\lambda} = \sum_{k = 1}^K P(Y = c_k)= 1
$$

就得到 $\lambda = \sum_{k = 1}^K\sum_{i  = 1}^N I(y_i = c_k) = N$ ，因此最大似然条件下得到的参数估计结果为

$$
P(Y = c_k) = \frac{\sum_{i  = 1}^N I(y_i = c_k) }{N}
$$

同理有 

$$
P(X^{(j)} = a_{jl} | Y = c_k) = \frac{\sum_{i = 1}^NI(x_i^{(j)} = a_{jl}, y_i = c_k)}{\sum_{i = 1}^NI(y_i = c_k)}
$$

朴素贝叶斯方法是一个非常简单直观的分类器构建的方法，但是使用范围局限在小模型中，并且特征的独立前提要求较强，很难在实际问题中有很好的适用性。