---
title: "SMART Reinforment Learning Finetuning"
description: "Reinforment Learning Finetuning of a AI Planner"
publishDate: 25 November 2025
tags: [tech/project, tech/adas]
draft: true
---

## Introduction
端到端(E2E)是近年来智能驾驶高频出现的名词，标志着智能驾驶技术架构逐渐向着全模块数据驱动方向发展，于是今年从感知转向端到端。这篇小结是对AI Planner（所谓二阶段端到端）和对应的强化学习微调的回顾，更多的是技术导向的，自己仍然是“Planning问题”的小白，缺少Planning在实际系统中问题的经验和思考。

### AI Planning

对Planning和AI Planning问题的一些观察和粗浅理解：
- Planning从技术上可以理解为有约束的优化问题；Planning问题也可以是一个搜索问题
- Planning模块中存在大量人工定义的规则去处理复杂的道路情况，这导致模块无法处理现实驾驶中复杂的、无穷尽的路况
- Planning vs Prediction：Planning更强调的是交互和博弈场景，Prediction更多的是时序预测
- Multi-modal in natrual。多模，这里不是视觉感知中的多种传感器数据的多模态，而是指输出多条可能的轨迹。可以理解为，在某个规划场景下，有多条可行的轨迹；可以想象成这是一个多峰的分布
- 大多数AI Planner现在都是模仿学习，无法从错误中恢复，存在所谓的covariance shift，而这对于AI Planning来说非常致命
- 感知模块是超大量数据压缩和理解的过程，可以形成一个抽象的向量化世界，给到Planning模型，信息或错误会不断传递给Planning
- AI Planner模型相比于感知模型其实非常小(几M到100M之间)，它只需要所谓的抽象的向量化世界即可，因此比较小的模型就可以学习到Planning数据中的模式
- AI Planner对训练数据的分布非常敏感，规模越大的人类驾驶数据，分布越难均衡，且直行样本占据绝大多数，交互和博弈的样本是非常少量的；现在很多智驾公司会雇佣司机专项去大规模(10万小时以上)采集驾驶数据
- AI Planner有Scaling Law吗？MotionLM的[Scaling Laws of Motion Forecasting and Planning Report](https://arxiv.org/html/2506.08228v1)验证了Scaling特性，使用了惊人的超过50万小时的数据，但是模型最大也不超过118M

[给自动驾驶感知工程师的规划速成课](https://zhuanlan.zhihu.com/p/706193528)

注意区分Trajectory和Path

### Tech choices

在项目中，共有三个AI Planner，可以称为Rregression-based Path Planner, Model-based Trajectory Planner, NTP-based Trajectory Planner，这三种Planner对规划问题的建模方式不同。

#### Rregression-based Path Planner


#### Model-based Trajectory Planner
Tree based AI Planner:
- 采用MuZero思想的model-based AI Planner：可以理解成预测action template中的动作，显示的建模state transition model p(s'|s,a)，实现任意长度的动作预测，并转化成轨迹点；训练过程中采用teacher forcing；
- 它把planning问题假设为一个搜索问题，那么model-based tree search planner确实是一个好方法
- 优点：良好的多模态预测，符合动力学的轨迹输出，可以进行强化学习训练
- 缺点：covariance shift严重，需要大量的augmentation

#### NTP-based Trajectory Planner
即基于SMART的Planner，更确切的说SMART是一个交通模型(Traffic Model)


#### Self Play RL
Self Play RL，初步了解了纯RL的Planning方法，尤其是Apple的纯强化学习项目，引出了gpudrive和self play
- RL Self play看起来可以实现几十亿次交互，应该是有充分的博弈和交互行为的
- RL是一个perfect information game下可以，但是自动驾驶中，场景的噪音非常大，单纯的self play RL我觉得也很难泛化到真实场景中
- 纯RL的行为需要靠Reward来约束，RL非常擅长hack某些reward，模型的行为虽然高效实现reward，但是非常不像人的行为

## Project development

### RLFT AI Path Planner
- RL尝试
  - 模型：出Path的AI Planner
  - 模拟器：手动实现，输入下一时刻的pose，对所有的动静态目标进行变换
  - 数据：在4帧数据上
  - RL框架：stablebaseline3
  - 动作空间：连续动作
    - 由于AI Planer只能输出Path，即waypoint的x和y，为了获取pose，
  - Policy：使用多元高斯讲连续动作作为均值，采用后做归一化
    - 加载预训练的参数
    - actor和critic都Random初始化
    - 增大actor和critic参数大小
  - Reward：
    - Apple self play的复杂Reward
    - Approach goal, collide, off-road
    - Goal Reach, collide, off-road
  - 指标
    - 主要看Reward(reach goal)的曲线，value的曲线和policy loss
  - 结论
    - 训练大都在5-6M step，即一个晚上左右
    - 采用预训练权重的实验，Reward曲线都比较差，没有持续上升的趋势，当时觉得是不是连续动作更难收敛
    - 采用随机初始化的实验，可以收敛，部分实验可以收敛，说明连续动作的AI Planner是没有问题的
    - 增大actor和critic的参数大小是有益的
    - 增大buffer size和batch size是有效的
  - 复盘
    - stablebaseline3对PPO的算法进行了非常大量的封装，几乎很难做一些自定义的修改，并且需要熟读doc，知道它抽象出来的接口对应PPO的哪个环节
    - stablebaseline3中所有的rollout数据存储在cpu，训练需要cpu到gpu的不断拷贝，训练效率较差

### SMART Pretraining
IL阶段
- 确定了SMART作为基础模型，它采用了Transformer模型和NTP，并使用700万帧训练，16卡A100训练
- 然后，使用CatK对上面的预训练进行了微调，可视化结果并不理想
### SMART RLFT
- SMART RL 
- 模型：
  - SMART预训练模型，但是只训练actor和critic
  - 全量模型参数参与训练
- 模拟器：Drive RL（定制版gpudrive）
- 数据：真实的1000帧数据
- RL框架：clean rl
- 动作空间：离散的轨迹词表
  - 虽说SMART输出的是离散词表中的序号，但是每个序号内包含一个0.5s的连续轨迹
  - 这里动作只取0.5s轨迹的第一个点
- Policy
  - 加载预训练actor(即SMART的next token prediction head)
  - 增大actor和critic，从头初始化
  - 增加了Goal Encode的信息，因为Goal的信息一直没有给到模型
    - 这个时候理解了SMART是交通模型，它构建了所有agent的行为的联合概率
    - 而Planning是一个条件概率，根据给定的信息，例如导航信息、红绿灯、变道等等，输出行为，且在这个条件信息下，依然是多模态的行为
- Reward
  - Goal Reach, collide, off-road
- 指标
  - perc reach goal，reach goal的比率
- 结论
  - 全量模型参与训练相比只训练actor和critic显示出可行性，但是由于会减少buffer size和mini batchsize，以及卡的数量有限，所以后续并没有继续
  - Ego Goal Encode是有效的方式，显著增加reach goal
  - 最有效的是：增大buffer size和mini batchsize
    - 这与CARL这篇文章的结论非常的一致
  - 由于encode goal使得actor的训练从头开始，所以reach goal虽高，但是动作非常不连续
  - clip frac一直比较高，在0.5以上，很多的参数更新都被截断了，调整lr可以让clip frac在0.1-0.3正常范围内，但是训练结果更差
- 复盘
  - 这是向着正式SMART RLFT迈进的重要一步，虽然还是很粗糙，但是连接起来：NTP模型+高效模拟器+成功PPO训练，对理解PPO，理解AI Planning有了重要的进步
### SMART RLFT 2.0
有非常多的地方可以改进，有些改进已经实现了，但是没有实验
- 将ego goal encode的actor head进行预训练，类比LLM，相当于SMART预训练是base model，ego goal encode相当于SFT，然后再用RL进行微调
  - 这样做可以避免RL从头开始训练
- 在policy loss中，增加base model和当前训练model的kl散度，避免训练得到的动作太远离base model
- 执行完一整个token的0.5s轨迹。这一步，有些类似于action chunking，执行完一个token并获得0.5s的reward
- 尝试GRPO(DAPO)对整条轨迹进行采样
- 继续扩大训练的规模，增大数据集规模，增大buffer size，增大mini batchsize

### Summary & Takeaways

- 我觉得Tree-based Planner有太多的所谓augmentation了，不对，纯RL也不对
- 坚信Transformer和NextTokenPrediction仍然是现阶段最有效的建模范式
- 遵循语言模型的发展，需要很重的预训练与RL的微调，E2E也走在这条路上，即超大的规模的模仿学习预训练(克隆人类的驾驶行为)，下个阶段就是通过RL把好的行为的概率拉升
- 所以，纯RL(包括self play)不是解决Planning的路径，IL->RL才是，当然是很重的IL
  - 当然，很重要的问题是RL是单步的，只能执行一步动作，而AI Planning出的是轨迹，通常是8s


## Reflections and Open Questions
我质疑，从人类驾驶数据中到底学到的是什么？人类驾驶的数据真的是专家轨迹吗？很多人类的驾驶行为完全不是最优行为，生活中经常骂的那些sb司机还是很多的
      - 如果diffusion拟合的是人类的驾驶数据的分布，如果质疑人类的驾驶数据，那么diffusion的拟合数据分布的方式可行吗？
      - 应该如何使用人类的驾驶数据？？？？
        - offline RL的训练一个value funciton是一个好主意吗？
    - 现在的所谓一阶段端到端模型，其实是进入了大模型时代
      - 以前很难想象可以带着视觉数据训练50万小时，这是惊人的开销，但是现在目前工业界都是这么做的，都是千张卡起的任务
      - 另外，很多的数据不是单纯的人类数据，而是优秀的司机专项去采集的10万小时起的数据

[Waymo的博客](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)对新范式的智能驾驶技术架构做了非常好的总结，值得借鉴。