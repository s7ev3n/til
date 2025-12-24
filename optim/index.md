---
title: "Optimizers"
description: "optimizer notes"
publishDate: "1 May 2022"
updatedDate: "3 July 2024"
tags: ["tech/ml"]
draft: false
---

> 本文是对常见优化器的总结。可以按照下面的路径来学习：GD->SGD->移动指数平均->动量SGD->RMSprop->Adam。

## Gradient Descent
梯度下降 (Gradient Descent)算法可以用下面的公式表示：
$$
\theta=\theta -  \eta \cdot \nabla_{\theta}J\left(\theta\right)
$$		
其中，$\theta$ 表示算法的参数， $\eta$是梯度下降的步长(也称为学习率)，$J$ 表示损失函数，$\nabla_{\theta}J\left(\theta\right)$ 表示损失函数对参数 $\theta$ 的偏导数。

**梯度下降需要对整个训练集进行一次反向传播计算梯度后再进行参数更新**，这个操作在深度学习是非常昂贵和不现实的。

## Stochastic Gradient Descent
随机梯度下降 (Stochastic Gradient Descent，SGD)在使用梯度更新参数时刚好相反，是另一个极端，每次只使用整个训练集中的一个样本，这样带来的问题是：参数更新间具有很大的方差，即下降的方向会有很大的波动。因此，出现了折中的方法：mini-batch SGD，即采样一个batch来更新梯度，也是深度学习中最常见的优化方法了。

SGD是很多经典网络的优化方法，他们反而没有采用类似Adam等看起来更新颖的优化方法，原因可能是SGD还是非常有效可以找到更好的极值点的。**SGD一旦陷入了鞍点(局部极值点)，就很难再逃离鞍点了**。

## SGD with Momentum
所谓动量(Momentum)，就是在梯度的更新项可以保留更多的历史梯度信息，如果历史梯度在某个方向上下降很快，那么即使这一次的梯度较小，依然可以以较大的梯度进行这次更新，所谓积累了动量。

动量就是以指数移动平均的方式更新梯度：
$$
v_t  = \gamma v_{t-1} + \nabla_{\theta}J\left(\theta\right)  \\
	  \theta = \theta - \eta \cdot v_t
$$		
其中,$v_t$ 就是动量值。

:::note
假设有$n$个数据$[\theta_1, \theta_2, …, \theta_n]$，移动指数平均(Exponential Moving Average)为：
$$
v_t = \alpha\cdot v_{t-1} + (1-\alpha)\cdot \theta_t
$$
其中$v_{t}$表示前t个数据的移动指数平均值，$\alpha$为加权权重(一般为0.9-0.999)。

EMA可以**近似**看成过去$1/(1−\alpha)$个时刻v值的平均，注意是近似。
:::

:::important
注意：这里更新的形式和指数移动平均稍有不同，是李沐视频[^1]中的更新方式，动量用泄漏平均值（leaky average）来计算，参考2采用这种方式；而吴恩达课程[^2]中，使用标准的移动指数平均，并对比这两种做了一定的解释。
:::
[^1]: [李沐：优化算法](https://zh-v2.d2l.ai/chapter_optimization/index.html)
[^2]: [吴恩达：2.6动量梯度下降法](https://www.bilibili.com/video/BV1FT4y1E74V?p=66)

## AdaGrad
SGD with Momentum优化算法是对SGD在梯度上进行了优化，是梯度更新的方差更小。但是对于所有的参数更新，采用相同的学习率。
AdaGrad试图解决稀疏特征的参数更新，通过的是自动调整学习率。

只有在这些不常见的特征出现时，与其相关的参数才会得到有意义的更新。如果采用相同的学习率，常见特征的参数相当迅速地收敛到最佳值，而对于不常见的特征，没有足够的观测保证更新到最佳。一种解决方法是记录特征的出现，允许对每一个参数动态地更新它的学习率。
具体来说，AdaGrad算法使用 $s_t$ 表示过去累加的梯度的方差，并除以均方差来调整学习率 $\eta$ 来更新参数：

$$
\begin{split}\begin{aligned}
    \mathbf{s}_t & = \mathbf{s}_{t-1} +  \nabla_{\theta}J\left(\theta\right)^2 \\
    \mathbf{\theta} & = \mathbf{\theta} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \nabla_{\theta}J\left(\theta\right)
\end{aligned}\end{split}
$$

我们可以发现，AdaGrad算法中$s_t$可能会越来越来大， $\frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}}$越来越小，最后导致该参数的更新几乎停止。

## RMSprop
Hinton在某个课程上提出了RMSprop这个改进，针对AdaGrad可能出现的累计梯度方差爆炸的问题，使用移动指数平均来更新累加梯度方差：
$$
\begin{split}\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \nabla_{\theta}J\left(\theta\right)^2 \\
    \mathbf{\theta} & \leftarrow \mathbf{\theta} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \nabla_{\theta}J\left(\theta\right)
\end{aligned}\end{split}
$$

## ADAM
> ADAM不是二阶优化器，二阶优化算法需要使用Hessian矩阵，ADAM中使用的是二阶矩！  

:::note
二阶矩指的是随机变量的平方的期望值，即$E[\mathbf{X^2}]$
:::

最后到了ADAM算法[^4]，是对前面优化算法的有点的集成。ADAM算法的优缺点：
- 1.收敛速度更快
- 2.对于稀疏数据具有优势，因为可以自适应学习率
- 3.可能不收敛,可能错过全局最优解

关键组成部分：
- 1.使用移动指数平均来更新动量和梯度方差(梯度的二阶矩)：
$$
\begin{split}\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \nabla_{\theta}J\left(\theta\right), \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \nabla_{\theta}J\left(\theta\right)^2
\end{aligned}\end{split}
$$			
其中，常见设置是$\beta_1=0.9$和$\beta_2=0.999$，也就是说，方差的估计比动量的估计移动得远远更慢。另外，如果设置$v_0=0, s_0=0$，在初始阶段，带来的动量和梯度的估计会有非常大的偏差，会**引出一个技巧 Bia Correction**[^3]。
$$
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}
$$				
其中， $t$表示$t$次幂。

[^3]: [吴恩达视频：偏差修正](https://www.bilibili.com/video/BV1FT4y1E74V?p=65)

- 2.更新梯度

$$
\nabla_{\theta}J\left(\theta\right)' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}
$$

其中的注意的是，下面的均方差计算，$\epsilon$并没有放在开根号计算中，与是RMPSprop的不同，据说这样实际的表现更好一丢丢。另外， $'$不是导数，而是对梯度的估计的意思。

- 3.更新参数
$$
\theta \leftarrow \theta - \mathbf{g}_t'
$$	

代码：		
```python
def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

[^4]: [Adam那么棒，为什么还对SGD念念不忘 (3)—— 优化算法的选择与使用策略](https://zhuanlan.zhihu.com/p/32338983)
