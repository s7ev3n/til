---
title: Distributed Training Handbook
description: distributed training
publishDate: "1 Dec 2025"
updatedDate: "1 Dec 2025"
tags: ["tech/ml"]
draft: true
---

## Background

### Glossary
#### `reduce` and `map`

`reduce`和`map`操作都是函数式编程语言Lisp中的基础原语。经典的MapReduce

`reduce`，通常翻译为归约，指的是将一组数据，通过某种运算规则，“浓缩”成一个数值。
我一直对这个`reduce`的概念感到陌生，没法把它的含义和英文单词联系起来。

在串行(单线程)的语境下，也有归约的概念，例如`tf.math.reduce_mean()`，即对张量的某个维度进行求平均规约，但是现在基本简化为了`mean`操作。

在并行的语境下，`reduce`操作指的是多个CPU或GPU参与运算，这其中就会涉及到跨设备的通信，常见的操作见[集体通信操作](#collective-operations)。


`map`，通常翻译为映射，是对某个集合内的所有元素应用同一个操作，得到经过操作后的新的集合，所以本质上，`map`操作和一对一的函数映射一致。


#### rank, local_rank, world_size, host, device

- `rank`和`local_rank`
`rank`表示进程序号，通常是从0开始的连续整数值。`rank`是一个抽象概念，是用于进程间通讯的逻辑唯一标识或地址，它屏蔽了底层细节（例如进程运行在哪台机器），只需要`rank`即可实现不同进程的通讯。在并行训练的语境下，`rank`一般指的是系统全局的某个“进程-GPU对”，一般情况下**每个进程独占一块GPU**。`rank=0`通常被称为主进程（Master/Root），负责初始化、分配任务和汇总结果。

> 为什么叫`rank`？是因为沿袭了MPI(Message Passing Interface)即分布式进程间消息传递标准其中的术语。

> 为什么进程和GPU是一对一的关系？ 没有硬性规定进程和GPU是一对一的关系，而是工程上最省事、最省内存、最省通信的做法。
每个GPU同时只能被一个CUDA Context“独占”驱动，一个进程内部所有线程共享同一个CUDA Context。

| 映射方案         | 工程代价          | 实际结果                        |
| ------------ | -------------------| ----------------------------- |
| **一进程多 GPU** | 1. 进程内要自己调度多张卡，手动管理 `cudaSetDevice`；<br>2. 每次通信要先把数据 `gather` 到同一块卡再发，额外拷贝；<br>3. NCCL 集合通信（AllReduce）要求每张卡一个独立“rank”，同进程里多张卡只能当“子 rank”，框架代码爆炸；<br>4. 单进程地址空间增大，显存碎片、FD、线程数都翻倍。 | 代码复杂、易错、性能差，没人愿意写。            |
| **多进程一 GPU** | 1. GPU 同时只能跑一个 CUDA context，第二进程就得阻塞等上下文切换；<br>2. 显存被重复申请，两份模型参数、两份梯度，瞬间 OOM；<br>3. NCCL 通信时两个进程抢同一条 PCIe/NVLink，带宽竞争。                                                           | 性能反而下降，显存直接炸。                 |
| **一进程一 GPU** | 1. 代码最简单：每个进程永远 `cudaSetDevice(local_rank)`，一辈子不用切卡；<br>2. NCCL 集合通信天然要求“rank 数 == GPU 数”，1:1 直接满足；<br>3. 显存、计算、通信上下文完全隔离，没有竞争；<br>4. 失败隔离好，一张卡挂只死一个进程，重启代价最小。                   | 框架层实现最瘦、用户代码最少、性能最好，于是成了事实标准。 |

> 什么是CUDA Context？它是GPU设备执行CUDA程序所需的所有状态信息和资源管理的总和，可以把它看成进程在GPU的“地址空间+执行环境”。当一个主机端的进程(例如，执行的Python训练脚本)第一次调用CUDA API的时候，会自动为每个它将使用的GPU设备创建一个CUDA Context，假设有两个GPU设备，它将创建两个独立的CUDA Context。

> CUDA Context占用多大的显存？context 页表、kernel 代码段、常量段、printf 缓冲等，每进程每卡 300–800 MB 起步，是的，空的进程就要占据将近1GB的显存。

主要包括：
1. 设备状态（Device State）：

- 计算内核（Kernels）： 所有准备好在 GPU 上执行的函数代码。
- 运行时配置： 用于配置 Grid 和 Block 大小的启动参数。

2. 内存管理（Memory Management）：
- 设备内存空间： 维护设备（GPU）上分配的所有全局内存（Global Memory）地址和大小信息。
- 页锁定内存（Pinned/Host Memory）： 记录 Host 端用于与 GPU 高速传输数据的内存块。

3. 资源管理（Resource Management）：
- 流（Streams）： 用于管理和调度异步操作（如数据传输和 Kernel 执行）的队列。
- 事件（Events）： 用于同步和计时操作的标记。


`local_rank`是进程所在的物理节点(Node)的进程唯一标识，它的主要作用和`rank`不同，是用于节点(Node)内的资源配分，即`cudaSetDevice(local_rank)`


`world_size`就是所有可见的GPU的总数，可能在一些求平均的操作中用到。

- `host`和`device`

`host` = 主机端 = CPU + 内存
`device` = 设备端 = GPU + 显存

### Collective Operations
分布式训练会涉及到跨节点、跨卡的通讯，使用的是NCCL库作为通信后端，并定义了相关的[集体通信操作](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)，在此处对这些操作进行简单的翻译和搬运。

#### `AllReduce`

#### `Reduce`

#### `Broadcast`

#### `AllGather`

#### `Gather`

#### `Scatter`

#### `ReduceScatter`

#### `AlltoAll`


### Memory Profiling
显存的占用最主要的四部分：模型权重(Model weigts)、模型梯度(Model gradients)、优化器状态(Optimizer states)和激活值(Activations，用于计算梯度)。还有一些其他的显存占用，例如，CUDA Context，一些Buffer，以及碎片，下文讨论中忽略这些显存占用。

[Memory usage in transformer](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=profiling_the_memory_usage)

### Mixed precision training

混合精度训练不省显存！混合精度训练不省显存！混合精度训练不省显存！甚至相比FP32的全精度的梯度累计(accumulate gradients)训练，每个参数还要多4 bytes。

全精度FP32训练（一个FP32参数4字节）: $m_{params}=4 * N$，$m_{grad}=4 * N$，$m_{optim}=(4+4)*N$

混合精度训练（BF16）：$m_{params}=2 * N$，$m_{grad}=2 * N$，$m_{param\_fp32}=4 * N$，$m_{optim}=(4+4)*N$

出于训练稳定性的原因还会再存储FP32的权重：$m_{param\_fp32}=4 * N$，被称为master weights.

即使不能省显存，混合精度训练还是有很大的优势是在前向和后向的训练中，1）低精度的操作更快，2）前向时降低激活值的显存占用


## Sharding
Deepspeed ZeRO or PyTorch FSDP


## Parallelism Methods

### Data Parallelism

### Tensor Parallelism

### Pipeline Parallelism