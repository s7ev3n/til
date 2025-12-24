---
title: "Beam Search"
description: "beam serach"
publishDate: "1 May 2025"
tags: ["tech/gems"]
---

Beam Search，中文又称为集束搜索，可以理解为贪心搜索和BFS搜索的中间状态，即树搜索时选择top k个更有希望的节点继续搜索。
它的工作流程大致如下：

1. 从所有可能的初始选项中，选出概率最高的`k`个选项作为最初的`k`个候选序列。
2. 后续步骤：
   1. 针对上一步保留的每一个候选序列，计算所有可能的下一个选项，并生成新的、更长的候选序列
   2. 在所有这些新生成的候选序列中，根据整体的评分（通常是累积概率），再次选出最高的`k`个。
3. 循环往复：重复上一步，直到达到结束标志（如生成了终止符或达到预设的最大长度）
4. 最终选择：从最终保留的 k 个候选序列中，选出得分最高的那一个作为最终结果

大体代码：
```python
def beam_search_decoder(model, start_sequence, beam_width, max_length):
    """
    使用 Beam Search 算法进行序列解码。

    Args:
        model: 一个能够根据当前序列预测下一个词概率的模型。
               它应该有一个 predict_next(sequence) 方法。
        start_sequence (list): 初始序列，通常只包含一个起始符，例如 ['<start>']。
        beam_width (int): 束宽 (k)，即每一步保留的候选项数量。
        max_length (int): 生成序列的最大长度。

    Returns:
        list: 得分最高的最佳序列。
    """
    
    # 1. 初始化
    # beams 是一个列表，存储元组 (sequence, score)
    # 初始时，只有起始序列，其对数概率得分为 0.0
    beams = [(start_sequence, 0.0)]

    # 2. 迭代解码
    # 循环直到达到最大长度
    for _ in range(max_length):
        all_candidates = []

        # 3. 扩展 (Expansion)
        # 遍历当前的所有候选序列 (beams)
        for seq, score in beams:
            # 如果序列已结束，则将其视为一个最终候选，保留它但不进行扩展
            if seq[-1] == '<end>':
                all_candidates.append((seq, score))
                continue

            # 使用模型预测下一步的所有可能输出及其概率
            # `model.predict_next` 返回一个 (token, probability) 的列表
            next_word_predictions = model.predict_next(seq)

            # 为每个可能的下一个词创建新的候选序列
            for word, prob in next_word_predictions:
                # 使用 log 概率，将乘法转换为加法，避免数值下溢
                new_score = score + math.log(prob)
                new_seq = seq + [word]
                all_candidates.append((new_seq, new_score))

        # 4. 剪枝 (Pruning)
        # 根据得分对所有候选序列进行降序排序
        # key=lambda x: x[1] 表示按元组的第二个元素（即 score）排序
        ordered_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        
        # 选择得分最高的 beam_width 个序列作为新的 beams
        beams = ordered_candidates[:beam_width]

        # 5. 检查终止条件
        # 如果所有候选序列都已生成结束符，则可以提前停止
        if all(seq[-1] == '<end>' for seq, score in beams):
            break
            
    # 6. 返回结果
    # 最终，得分最高的序列（列表中的第一个）即为最佳结果
    # 我们返回序列本身，忽略其得分
    best_sequence, best_score = beams[0]
    return best_sequence

# --- 模型和参数的伪定义 ---

class PseudoModel:
    def predict_next(self, sequence):
        # 这是一个伪实现。在真实场景中，这里会调用神经网络（如Transformer, LSTM）
        # 并返回一个(词, 概率)对的列表，通常是softmax层的输出结果。
        # 例如: [('你好', 0.8), ('世界', 0.15), ...]
        print(f"  预测 '{' '.join(sequence)}' 的下一个词...")
        # 返回一些虚拟的预测结果
        if sequence[-1] == '<start>':
            return [('我', 0.6), ('你', 0.4)]
        elif sequence[-1] == '我':
            return [('爱', 0.7), ('是', 0.3)]
        elif sequence[-1] == '你':
            return [('好', 0.9), ('是', 0.1)]
        elif sequence[-1] == '爱':
             return [('Python', 0.9), ('<end>', 0.1)]
        else:
            return [('<end>', 1.0)] 
```

返回的beams是一个list，元素是一个(sequence, score)元组：
```python
[
  ( [<token1>, <token2>, ..., <end>],  -0.85 ),  # 这是 beams[0]，得分最高的序列
  ( [<token_a>, <token_b>, ..., <end>], -1.02 ),  # 这是 beams[1]，得分第二高的序列
  ( [<token_x>, <token_y>, ..., <end>], -1.15 ),  # 这是 beams[2]，得分第三高的序列
  ...
]
```

执行这个代码，可能得到的结果：
```text
# # 1. 定义模型和参数
# my_model = PseudoModel()
# k = 2
# start_seq = ['<start>']
# max_len = 5

# # 2. 执行 Beam Search
# result = beam_search_decoder(my_model, start_seq, k, max_len)

# # 3. 打印结果
# print(f"\n最佳序列: {result}") 
# # 预期输出 (可能会是): ['<start>', '我', '爱', 'Python', '<end>']
```