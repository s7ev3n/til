---
title: Loss functions
description: different loss functions
publishDate: "1 Jan 2023"
updatedDate: "4 Jan 2025"
tags: ["tech/ml"]
---

> 这里总结下各种Loss Functions，持续更新中。

## Classification Loss Functions
分类Loss是最常见的Loss Function，几乎都是从熵的概念发展出来的。

### Cross Entropy
在介绍各种分类损失函数之前，我们先从交叉熵的定义开始。可以从两个角度来理解交叉熵：1）信息熵；2）极大似然估计。

**先从信息熵理解交叉熵？**

熵是信息量的表达或者说度量，可以用$h(x) = -log(p(x))$来计算一个随机事件发生的熵，从公式可知，概率越小的随机变量发生了($X=x_{i}$)产生的熵越大，信息量越大。注，$log$以2为底，此时信息用bit(比特)量化。

信息熵用下面公式进行表达：
$$
H[P] = \sum_i - P(i) \log P(i)
$$
其中，$P$是概率质量函数，从公式可以得知，**信息熵是对于所有可能随机变量$X$的取值的期望值，即所有可能发生事件所带来的信息量的期望**。

相对熵，也叫KL散度(KL divergence)，它是一种距离的度量，描述两个分布或随机变量的统计距离。统计距离量化了两个统计对象之间的距离，统计对象可以是两个随机变量，两个概率分布或者样本，或者一个独立样本点和一个点群之间的距离，或者更加广泛的样本点。

$$
DD_{KL}(P||Q)=\sum_{i=0}^{n}p(x)\frac{p(x)}{q(x)}=-\sum_{i=0}^{n}p(x)\frac{q(x)}{p(x)}
$$

将$\log \frac{p(x_{i})}{q(x_{i})}$展开成$\log p(x_{i}) - \log q(x_{i})$带入上面的公式，可以推导得到信息熵、相对熵和交叉熵之间的关系：
$$
D_{KL}(p \| q) = CE(p, q) - H(p)
$$

可以得到交叉熵的公式定义：
$$
CE=-p(x_i)log(q_i)
$$

:::tip
KL散度可以理解为从p角度来看，$q$事件和$p$事件的差异性，这点需要搞清楚 $D_{KL}(p \| q)$ 的方向，因为它不是对称的，意味着一个分布$P$到另一个分布$Q$的距离不等于$Q$到$P$的。

在进行Knowledge Distillation Loss的计算时，$p_{student}$和$p_{teacher}$ 要分清楚，答案是$D_{KL}(p_{teacher} \| p_{student})$ ，即从Teacher角度看，Teacher和Student的差异性。
:::

**从最大似然理解交叉熵**

假设我们是在进行分类任务，分类任务的服从[类别分布](https://zhangzhenhu.github.io/blog/glm/source/%E6%A6%82%E7%8E%87%E5%9F%BA%E7%A1%80/content.html#id20)(categorical distribution)：
$$
P(X) = \prod_{k=1}^{K} p_i^{y_k}, \quad \sum_{k=1}^K p_k = 1
$$

其中，$y_k$ 为类别标签，真值为1，其余为0；$P(x)$ 就是似然函数，**最大化似然函数就最小化负对数似然函数**，即损失函数为：
$$
min: - \sum_{k=1}^K y_k log(p_k)
$$
此时，很自然交叉熵的损失函数就出来了。

### Binary Cross-Entropy Loss
BCE又称为Sigmoid Cross-Entropy Loss，适用于**二分类任务**，每个样本只有两个类别：正例和负例，具体的公式：
$$
BCE=-\frac{1}{N}\sum_{i=1}^{N}[y_i log(p_i) + (1-y_i)log(1-p_i)]
$$
其中，$y_i$是真实标签，即$0$或者$1$，$p_i$是模型预测为$1$的概率（$p_i$是使用Sigmoid函数计算），$N$是样本数量。当标签为$1$时，该样本贡献的损失是$-log(p_i)$，当标签为$0$时，该样本贡献的损失是$-log(1-p_i)$。

简单的代码实现如下，需要注意的是`targets`，在BCE中已经是one-hot向量了：
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bce(inputs, targets, epsilon = 1e-15):
    '''
    BCE: -1/N*(yi*log(pi)+(1-yi)*log(1-pi))

    inputs: (N, ), 模型输出的logits
    targets: (N, )，标签，值为0或1
    epsilon: 防止 log(0) 导致的数值问题
    '''
    probs = sigmoid(inputs)
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))

    return loss
```

### Categorical Cross-Entropy Loss
Categorical Cross-Entropy Loss，又被称为Softmax Cross-Entropy Loss，适用于**单标签多分类**任务，即有多个（$>2$）类别，并且某样本只属于某一个类别。

公式表达如下，对所有的类别进行了求和，虽然除了只有标签为$1$的类别才能计算，其余的类别均为$0$，并没有贡献Loss：
$$
CE=-\frac{1}{N}\sum_{j=1}^{N}\sum^{C}_{i} y_i log(p_i)
$$

代码实现中，注意输入的`targets`是类别的序号，但是在计算时候应该变为one-hot向量：
```python
def softmax(x):
    '''
    Args:
    x: (N, C)，俗称logits，即每个类别的分数，C表示类别的数量

    Returns:
    ret: (N, C)，表示在每个类别上的概率
    '''
    # softmax计算先减去最大值，保持数值溢出，维持数值计算稳定
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def ce(inputs, targets, epsilon = 1e-15):
    '''
    CE = -1/N*log(pi)

    inputs: (N, C), 模型输出的logits
    targets: (N, )，标签，值为类别的序号
    epsilon: 防止 log(0) 导致的数值问题
    '''
    probs = softmax(inputs)
    probs = np.clip(probs, epsilon, 1 - epsilon)
    n_cls = inputs.shape[1]
    # targets本应该变成一个one-hot向量(N, C)和probs直接相乘
    # 这里采用的切片的方法probs[np.arange(n_cls), targets]
    # 得到一个(N,)的向量，其中是每个样本在标签处的概率
    loss = -np.mean(np.log(probs[np.arange(n_cls), targets]))
    return loss
```

### Focal Loss
Focal Loss是计算机视觉中用于处理分类问题中类别不平的情况，即如果一个样本被模型高概率预测为正确，那么它对loss的贡献应该很小，而一个样本如果被模型预测错误，那么它对loss的贡献应该更大，即使模型更关注难样本。**Focal Loss使用Sigmoid函数，可以说是BCE Loss的改进**。

论文中的公式如下：
$$
FL(p_t)=-\alpha_t(1-p_t)^{\gamma}log(p_t)
$$
但是上面公式中的$p_t$是对BCE的简化，看起来更简洁，但是不方便代码实现，写成BCE的形式更方便写代码：
$$
FL(p_i) = \sum_{i}^{N} -\alpha \cdot y_i \cdot (1 - p_i)^\gamma \cdot \log(p_i) - (1 - \alpha) \cdot (1 - y_i) \cdot p_i^\gamma \cdot \log(1 - p_i)
$$

:::important
目标检测的分类任务是一个多分类的Loss，但是对Focal Loss的描述是对BCE二分类的改进，这是怎么回事？

实际上，可以把每个类别的预测看作一个独立的二分类问题（即one-vs-rest方式），这种情况下Focal Loss就是在每个类别上对二分类交叉熵进行了调制，从而重点关注难分样本。多类别问题被拆解成了多个二分类问题。

代码实现上，Focal Loss像是BCE和CCE的混合：即输入的target是one-hot，和CCE一致；然后对样本属于每个类别分别做BCE，再求和。
:::

代码实现参考[^1] ：
```python
def py_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    inputs = inputs.float() # (N, C) N表示样本数量
    targets = targets.float() # (N, C) 表示N个样本的类别，是one-hot
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t的写法和前面的BCE一致
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # alpha_t也是和BCE的类似写法
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()
```

下面贴出了参考[^1]中的`FocalLoss`实现，其中需要注意的是`F.one_hot`中需要创建`num_classes+1`的长度，把类别的需要放到对的位置。
<details>
<summary><code>FocalLoss</code>实现：</summary>

```python
class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, (N, C)
            target (torch.Tensor): The learning label of the prediction, (N,), 类别编号
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                # target要变成one-hot，注意这里的num_classes+1
                # 这样把类别编号放在对的位置
                # 后面再取到num_classes就是把背景或ignore类别放进去
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
```
</details>

[^1]: [mmdet.gaussian_focal_loss](https://mmdetection.readthedocs.io/en/v2.10.0/_modules/mmdet/models/losses/gaussian_focal_loss.html)
## Ranking Loss Functions

[^2]: [Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss](https://gombru.github.io/2019/04/03/ranking_loss/)