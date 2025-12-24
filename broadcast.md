---
title: "Broadcasting"
description: "use broadcasting"
publishDate: "3 July 2022"
tags: ["tech/gems"]
---

## 广播机制

Numpy存在广播机制允许不同形状的数组进行算术运算，其核心目的是在不显式复制数据的情况下，将较小或维度较低的数组扩展为与较大数组兼容的形状，从而执行高效的计算。对应地，torch也有广播机制。

### 广播机制的规则
为了使得不同形状的数组可以进行算术运算，通过两个步骤实现：**对齐维度**和**扩展维度**。

- **对齐维度**
若数组维度不同，将较小维度数组的**左侧**补1，直到维度数一致。重要的事情说三遍：左侧左侧左侧补1。例如，形状为`(3,)`的一维数组与形状为`(2,3)`的二维数组运算时，前者被补全为`(1,3)`。

- **扩展维度**
当两个不同形状的数组在维度上补全后，还需要扩展维度的长度使得两者可以在对应的维度上匹配，但是需要**只有长度为1的维度可以扩展**！例如上面的例子中，被补全后的`(1,3)`数组，还需要在`1`的维度上扩展到`2`，两者才能计算。

### 增加轴的方法

#### `Numpy`增加维度
- 使用`np.newaxis`或者`None`或`...`：
```python
arr = np.array([1, 2, 3])  # shape: (3,)
arr_new = arr[np.newaxis, :]  # shape: (1, 3)
arr_new = arr[np.newaxis, ...] # shape: (1, 3)
arr_new_col = arr[:, np.newaxis]  # shape: (3, 1)
arr_new2 = arr[None, :] # shape: (1, 3)
```
- 使用`np.expand_dims`
```python
arr_expanded = np.expand_dims(arr, axis=0)  # shape: (1, 3)
arr_expanded_col = np.expand_dims(arr, axis=1)  # shape: (3, 1)
```

#### `PyTorch`增加维度
- 使用`tensor.unsqueeze`
```python
import torch
tensor = torch.tensor([1, 2, 3])  # shape: [3]
tensor_unsqueezed = tensor.unsqueeze(0)  # shape: [1, 3]
tensor_unsqueezed_col = tensor.unsqueeze(1)  # shape: [3, 1]
```
- 使用`None`或`...`
```python
tensor = torch.tensor([1, 2, 3])  # shape: [3]
tensor_new = tensor[None, :]  # shape: [1, 3]
tensor_new = tensor[None, ...]  # shape: [1, 3]
```

### 使用广播机制
上面的规则，一般都是广播机制自动的匹配机制。但是使用广播机制的精髓是**创造机会**来达到“不显式复制数据的情况下，从而执行高效的计算”。

例如，我们有两个三维点集: `points1: (m, 3)`和`points2: (n, 3)`，现在我们要计算`points1`所有点到`points2`点集中最近的点的下标。这里就可以使用广播机制来实现：
- 首先，扩充`points1`的轴：`points1[:, None, :]`，之后的形状是`(m,1,3)`
- 然后，扩充`points2`的轴：`points2[None, ...]`，之后的形状是`(1,n,3)`
- 此时两者相减触发广播机制，`points1`会在1的维度扩充到`(m,n,3)`，`points2`会扩充到`(m,n,3)`，达到的效果是所有的m个点会和所有的n个点相减
- 最后，`np.argmin(np.sum(diff**2, axis=2), axis=1)`

## 广播机制的原理
广播机制在内部实现中，并没有真正复制数据，而是通过虚拟扩展（Virtual Expansion）的方式处理不同形状的数组运算。相比在某一个维度复制数据，在绝大多数场景下广播机制是高效且内存友好的选择。

所谓**虚拟扩展**是指通过调整张量的`shape`和`stride`，模拟数据复制后的行为，而非物理上的复制数据。
