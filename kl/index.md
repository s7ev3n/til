---
title: "KL Divergence"
description: 'some more detail about kl'
publishDate: 03 January 2026
tags: [tech/gems]
draft: false
---

## Intro

在博客中的一些地方，我们提到过KL Divergence这一定义，它衡量两个概率分布$P$和$Q$之间差异的非对称度（可以理解为距离），它具有方向性(!):

$$
D_{KL}(P||Q)=\sum_{i=0}^{n}p(x)\frac{p(x)}{q(x)}=-\sum_{i=0}^{n}p(x)\frac{q(x)}{p(x)}
$$

但是我们之前就停止在那里了，我们现在继续下去。

## Symbol
首先，从分布的名称$P$和$Q$开始，公式中分布$P$代表的是什么分布？分布$Q$代表的是什么分布？

$P$被定义为“真实分布”（The true distribution），而$Q$被定义为 “模型分布” 或“近似分布”（The model/approximating distribution），即$P$是one-hot真值，$Q$是模型预测。

惯例用$P$来表示真实分布的原因可以这样记$P$是单词Probability（概率）的首字母，在统计学中，$p(x)$通常用来描述自然界中数据的真实生成概率，或者是理论上的基准分布（Ground Truth，因为它是我们最关心的最基本概念，所以默认使用第一个字母$P$来表示。

## One step into KL

仔细观察一下上面KL定义的公式，是以$P$分布为权重的加权平均，用概率论的语言来说，它可以用期望来表达：

$$
D_{KL}(P(X)||Q(X))=\sum_{i=0}^{n}p(x)\log \frac{p(x)}{q(x)}=\mathbb{E}_{x\sim P(X)}[\log\frac{p(x)}{q(x)}]
$$

理解KL散度可以从另外的角度来看：
- 视角由$P$决定：我们在计算差异时，是站在$P$的角度去“采样”数据点，然后衡量$P$和$Q$在这些点上的差异
- 前后的主次关系：$P$是基准，是标准；$Q$是用来拟合$P$的对象

$P$写在前面，表示这个误差是 “从$P$的视角看出去，$Q$跟我差多远” 

## KL Directions

接下来我们就可以看下KL的方向不同了，这其实在[VAE的文章](https://www.s7ev3n.xyz/posts/vae/#forward-kl-vs-reverse-kl)有过介绍，但是那是嵌在VAE的语境下的，下面可以更多的来看下不同方向的KL的特性。

### Forward KL

正向KL

别名：Maximum Likelihood Estimation (MLE，最大似然估计)

含义：在真实分布$P$发生的地方，去衡量$Q$预测的概率有多准

数学行为：$\textbf{E}_{x\sim P}​[logP(x)−logQ(x)]$，如果在某个$x$处，$P(x)>0$（真实数据存在），但$Q(x)→0$（模型认为概率是0），那么$logQ(x)→−∞$，导致整个 KL 散度变成无穷大。

特性：Zero-Avoiding（零避免）。为了防止无穷大的惩罚，$Q$绝对不敢在$P$有概率的地方等于0。$Q$必须“覆盖”住$P$所有的高概率区域。

结果：**Mode Covering（模式覆盖）**。如果$P$是多峰分布（比如有两个分开的山峰），而 $Q$是单峰高斯分布。Forward KL会迫使$Q$变得很宽，试图同时罩住两个山峰。后果：$Q$的方差会变得很大，虽然覆盖了$P$，但在任何一个具体的峰值上，$Q$的概率密度都很低（变得很模糊）。

### Reverse KL
反向KL，需要注意的是，$P$和$Q$的含义没有变化！

别名：Variational Inference (VI，变分推断)

含义：在模型分布$Q$认为会发生的地方，去衡量$Q$和$P$的差异

数学行为：$\textbf{E}_{x\sim Q}​[logQ(x)−logP(x)]$。积分是在$x \sim Q$上进行的，如果在某个$x$处，$P(x)=0$（真实数据不存在），但$Q(x)>0$（模型编造了数据），$logP(x)→−∞$，散度变无穷大。如果$P(x)>0$，但$Q(x)=0$，这一项在积分中乘以$Q(x)$后就是0，没有惩罚！

特性：Zero-Forcing（零强制）。$Q$不敢在$P$没有概率的地方生成概率（不敢乱编造）。但是，$Q$可以心安理得地忽略$P$的某些部分（只要$Q$自己在那边也是 0）

结果：Mode Seeking（模式寻找）。同样面对多峰分布$P$和单峰高斯$Q$。Reverse KL会选择$P$中的一个山峰，然后紧紧地包裹住它。对于其他$P$的山峰，只要$Q$在那里不分配概率，就不会受到惩罚。后果：$Q$的方差会很小，拟合得非常尖锐和精准，但可能会漏掉$P$的其他模式。