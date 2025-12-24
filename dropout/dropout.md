---
title: "Dropout"
description: "dropout"
publishDate: "3 July 2022"
tags: ["tech/ml"]
---

> dropout是2012年Hinton在其论文《Improving neural networks by preventing co-adaptation of feature detectors》中提出的，用于防止过拟合的发生。

## Dropout的原理
Dropout一般用在全连接层之后，它的思路很简单：每次在前向传播的时候，让某个神经元的激活值以一定的概率$p$停止工作(置为$0$)。

为什么Dropout可以防止过拟合呢？
Dropout提供了一种有效地近似组合指数级的不同经典网络架构的方法：
- 多个模型的平均基本上都可以提高模型的性能，例如ensemble方法，但是训练多个深度学习模型成本是高昂的。Dropout近似的模拟了组合指数多个模型的情况，但是共享所有的参数。
- 训练时，每次前向传播都会采样出一个子网络，但是测试时，求这些模型的平均值(期望)是不可能的。因此，需要尽量让期望值是相同的，于是有下面两个方式来让dropout函数的输出的期望在训练和测试时是相同的。

## Dropout的细节
训练时与测试时dropout有什么区别呢？你肯定会想到训练时候drop，测试时候不能drop。这个回答只对了一部分，因为dropout会改变输入的分布，即期望值变小了，导致测试时候的输入分布与训练时候的输入分布不一致，因此需要对输入**进行缩放**。

首先，输出的logits如果一定概率扔掉，如果把dropout看做一个函数，那么这个函数的期望值肯定是变小了，称为$E(f_{drop(logit)}) = p \cdot logit$，**注意其中$p$是保留下来的概率**(与pytorch等函数实现时候，$p$经常表示的是drop的概率，下面的$p$也是相同的含义。)那$logits$可以用下面的$logits'$代替，这样dropout的输出的期望值保持不变[^1]：
$$
\begin{split}\begin{aligned}
logits' =
\begin{cases}
    0 & \text{ 其他情况 } 1-p \\
    \frac{logits}{p} & \text{ 留存概率为p}
\end{cases}
\end{aligned}\end{split}
$$

为什么要维持期望值不变呢？dropout相当于在每层计算时注入了噪音，那么关键的挑战就是如何注入这种噪声。 一种想法是以一种无偏向（unbiased）的方式注入噪声。 这样在固定住其他层时，每一层的期望值等于没有噪音时的值。

[^1]: [李沐的动手深度学习: 4.6. 暂退法Dropout](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html)

最后，dropout的简单实现：
```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return np.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```