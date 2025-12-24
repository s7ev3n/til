---
title: Monte Carlo Tree Search
description: mcts
publishDate: "4 May 2025"
updatedDate: "4 May 2025"
tags: ["tech/gems"]
---

## Monte Carlo Tree Search

蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）是一种用于在某些类型的决策过程中（尤其是在具有巨大搜索空间的问题中，如棋类游戏）寻找最优决策的启发式搜索算法，对比之下DFS和BFS就是暴力搜索。MCTS被用在AlphaGo, MuZero中，近期的LLM-Reasoning模型也有使用。

### 概述
使用MCTS解决某一个问题（例如下棋）就是建立求解这个问题的一颗搜索树的过程，从一个给定的状态出发开始，环境会在达到某个终止状态时给出奖励，然后回溯路径上的节点，并更新节点的一些特征值。

以下棋为例，你处在某一个棋局状态之中，应该怎么走呢？如果之前见过一样的或类似的棋局状态，就知道下面走哪一步胜率最高，如果不知道，那就在头脑中模拟，头脑中模拟就是MCTS过程，**没有真实的下棋子，没有改变任何的棋局状态**，通过MCTS后，就知道最好的下一步，这时候才真正下棋子，到达新的棋局状态，然后在新的棋局状态开始新一轮MCTS，只要见过海量的棋局状态，经过海量的MCTS，就可以知道最好的下一步。它的逻辑可以用伪代码表示：

```python
tree = MCTS()
board = new_board()
while True:
    # 输入你的下棋的棋子
    row_col = input("enter row,col: ")
    row, col = map(int, row_col.split(","))
    index = 3 * (row - 1) + (col - 1)
    if board.tup[index] is not None:
        raise RuntimeError("Invalid move")
    # 下棋（棋盘改变状态）
    board = board.make_move(index)
    print(board.to_pretty_string())
    # 如果棋盘到终止状态，游戏结束
    if board.terminal:
        break
    # 使用mcts的四步骤进行迭代
    for _ in range(mcts_iterations):
        tree.run_mcts(board)
    # mcts过后，我们就知道了最优的下一步，选择执行下一步，更新棋盘
    board = tree.choose(board)
```

### MCTS算法流程
MCTS的迭代核心有四个步骤：选择(Selection) $\rightarrow$ 扩展(Expansion) $\rightarrow$ 模拟(Simulation/Rollout) $\rightarrow$ 回溯(Backpropagation/Update)。
:::tip
在看其他资料的时候，有些资料的表示并不一致，造成了理解的困难：
- 未完全扩展节点，指的是该节点存在子节点，但是该节点还有未探索的可行子节点没有在作为节点添加。实际上，这和扩展的方式有关，常见的实现会一次性添加所有的可行节点，就不会出现这种情况了
- 叶子节点。叶子节点的定义按照尝试理解为没有子节点的节点，但是在MCTS的语境下，“未完全扩展节点”也属于叶子节点，即不是完全扩展节点或终止节点都算是叶子节点。这种说法显然有很大的歧义
- UCT的计算在不同的扩展方式下也不同。如果扩展阶段一次添加一个节点，那么UCT计算发生在完全扩展节点，因为在未完全展开节点，会结束选择阶段，进入扩展阶段。如果一次扩展所有节点，UCT计算每到一个节点都会发生，未探索的节点（一定是叶子节点）UCT值无穷大，因此每次都会选择未探索的节点

扩展方式的不同会造成选择阶段一些行为的不同，本文全部按照“**扩展阶段一次性添加所有的可行节点**”来理解。
:::
如下面的流程图所示：
<div class="mermaid">
graph LR
    B["根节点S<sub>0</sub>"] --> C{"当前节点是<br/>叶子节点?"};
    C -- 否 --> D["**选择(Selection)** <br/>当前节点 = 当前节点中UCB(S<sub>i</sub>)最大的子节点"];
    D --> C;
    C -- 是 --> E{"当前节点的<br/>n<sub>i</sub>=0?"};
    E -- 是 --> F["**模拟(Rollout)**"];
    E -- 否 --> G["**扩展(Expansion)** <br/>为当前节点一次性扩展所有可用动作作为子节点"];
    G --> H["随机选择一个子节点"];
    H --> I["**模拟(Rollout)**"];
    J["**反向传播(Backpropagation)** <br/>用模拟结果更新自当前叶节点至根S<sub>0</sub>路径上各节点的N、Q值"];
    F --> J;
    I --> J;
    J -.-> B;
</div>

也可以用如下的代码来理解：
```python
def run_mcts(self, node):
    "Make the tree one layer better. (Train for one iteration.)"
    path = self._select(node)
    leaf = path[-1]
    self._expand(leaf)
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)

```

#### 节点
每个节点需要记录三个基本信息：
- 当前节点的状态，例如棋盘的棋局
- 该节点被访问的次数$n_i$
- 累积评分值$v_i$，是平均奖励值，即获得的总奖励值除以$n_i$

> 累积评分值$v_i$是MCTS迭代之后，用于最终决定执行动作的依据，选择$v_i$最大的动作执行。是MCTS外部的游戏棋局选择执行下一步的依据。作为对比，UCB值不需要存储，只发生在MCTS的选择阶段，用于推进树的向下搜索。

#### 选择 Selection
选择(Selection)阶段的目标是找到下一个节点来进行扩展(Expansion)。我们默认扩展阶段会一次性添加所有的可行动作作为子节点，所以选择阶段结束一定是发生在叶子节点。在选择阶段搜索过程中，可能遇到下面的情况：

1. 该节点是非叶子节点，依据UCB值(Upper Confidence Bounds)选择最大值的节点往下搜索，由于$n_i=0$时，UCB的值为无穷大，此时随机选择一个即可
2. 该节点是一个叶子节点，此时选择阶段结束。叶子节点就是没有子节点的节点，它可能有两种情况，这个节点被“探索”过，就是被模拟过，所以有过统计信息，例如被访问次数不为0，另一种情况是这个节点是扩展阶段添加的，但是还没有进行过模拟。

可以参考下面的代码理解：
```python
self.Q = defaultdict(int)  # total reward of each node
self.N = defaultdict(int)  # total visit count for each node
self.children = dict()  # 记录所有扩展过的节点，可以通过children.keys()来获得扩展过的子节点
# self.children[node]是当前节点的所有子节点
        
def _select(self, node):
    "Find an unexplored descendent of `node`"
    path = []
    while True:
        path.append(node)
        # node not in self.children 表示这个节点从未被扩展过
        # not self.children[node] 为True是表示当前节点是叶子节点
        if node not in self.children or not self.children[node]:
            # node is either unexplored or terminal
            return path
        # 从当前子节点中排出掉已经扩展过的节点就是未扩展的节点
        unexplored = self.children[node] - self.children.keys()
        if unexplored:
            # pop操作是随机选择一个未扩展子节点
            n = unexplored.pop()
            path.append(n)
            return path
        # 所有的子节点都被扩展过（都被self.children记录过），就UCT选择
        node = self._uct_select(node)  # descend a layer deeper
```

再来说一下UCT函数，它平衡了探索和利用：
$$ 
\text{UCT} = \frac{w_i}{n_i} + C \sqrt{\frac{\ln N}{n_i}} 
$$
其中，$n_i$是当前节点的访问次数，由于$n_i$是分母，所以一个新添加的节点的UCB的值是无穷大；$N$是当前节点父节点的访问次数，$C$是一个常数，$w_i$是当前节点获得的奖励。

```python
def _uct_select(self, node):
    "Select a child of node, balancing exploration & exploitation"

    # All children of node should already be expanded:
    assert all(n in self.children for n in self.children[node])

    log_N_vertex = math.log(self.N[node])

    def uct(n):
        "Upper confidence bound for trees"
        return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
            log_N_vertex / self.N[n]
        )

    return max(self.children[node], key=uct)
```

#### 扩展 Expansion
扩展阶段的输入是选择阶段“选中”的未完全扩展的节点，扩展阶段的目标是从当前节点的状态下的可用的、未使用的动作列表中选择一个动作，并作为一个新节点添加到当前节点下作为子节点。

> 有些说法会说，在扩展阶段添加一个未被探索的动作作为子节点，但是在实际实现中，这里选择一次性添加所有的可行动作:

```python
def _expand(self, node):
    "Update the `children` dict with the children of `node`"
    if node in self.children:
        return  # already expanded
    # add **all unexplored children** of `node` to the tree
    self.children[node] = node.find_children()
```

#### 模拟 Simulation Rollout
模拟阶段从当前给定的节点出发，使用默认策略从合法动作中选择一个动作，推进更新到下一个状态，直到快速的玩完游戏获得奖励，例如获胜、输掉或者平局等。**需要注意的是模拟阶段并不添加任何新的节点，它只是想要从当前阶段快速的玩完游戏获得结果**，所以只要random快速做出决策即可：

```python
def _simulate(self, node):
    "Returns the reward for a random simulation (to completion) of `node`"
    invert_reward = True
    while True:
        if node.is_terminal():
            reward = node.reward()
            return 1 - reward if invert_reward else reward
        node = node.find_random_child()
        invert_reward = not invert_reward

```

#### 反向传播 Backpropagate
反向传播阶段就是游戏到了终点，得到了奖励，把奖励值加到经过的每一个节点上，即更新每个节点的总奖励值。因为模拟阶段并没有创建新的节点，所以奖励值将首先加到模拟开始的节点上，然后沿着选择的路径直到根节点：

```python
def _backpropagate(self, path, reward):
    "Send the reward back up to the ancestors of the leaf"
    for node in reversed(path):
        self.N[node] += 1
        self.Q[node] += reward
        reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa
```

#### MCTS实现
代码的实现参考一份极简的开源实现[^1]
[^1]: [A minimal implementation of Monte Carlo tree search (MCTS)](https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1):

```python
from abc import ABC, abstractmethod
from collections import defaultdict
import math

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        # add all unexplored children of `node` to the tree
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
```
### MCTS可视化
针对一个路径和问题，vibe coding了一个MCTS树展开的可视化过程，可以先安装一个`flask`环境，运行后端代码，运行单步迭代详情。

<details>
<summary>MCTS可视化后端代码</summary>

```python
import math
import random
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests 
import json
import logging

# --- Game Definition ---
class Game:
    TARGET_NUMBER = 10
    POSSIBLE_ACTIONS = [1, 2, 3]
    MAX_MOVES = 7

    @staticmethod
    def get_initial_state():
        return (0, 0)  # (current_sum, moves_taken)

    @staticmethod
    def perform_action(state, action):
        current_sum, moves_taken = state
        new_sum = current_sum + action
        return (new_sum, moves_taken + 1)

    @staticmethod
    def get_legal_actions(state):
        current_sum, moves_taken = state
        app.logger.debug(f"Game.get_legal_actions called with state: {state}. TARGET: {Game.TARGET_NUMBER}, MAX_MOVES: {Game.MAX_MOVES}, CLASS POSSIBLE_ACTIONS: {Game.POSSIBLE_ACTIONS}")
        if current_sum >= Game.TARGET_NUMBER or moves_taken >= Game.MAX_MOVES:
            app.logger.debug(f"Game.get_legal_actions for state {state} returning [] because terminal condition met: sum_check: {current_sum >= Game.TARGET_NUMBER}, moves_check: {moves_taken >= Game.MAX_MOVES}")
            return []
        legal_actions_copy = list(Game.POSSIBLE_ACTIONS)
        app.logger.debug(f"Game.get_legal_actions for state {state} returning: {legal_actions_copy}")
        return legal_actions_copy


    @staticmethod
    def is_terminal(state):
        current_sum, moves_taken = state
        return current_sum >= Game.TARGET_NUMBER or moves_taken >= Game.MAX_MOVES

    @staticmethod
    def get_reward(state):
        current_sum, moves_taken = state
        if current_sum == Game.TARGET_NUMBER:
            return 1.0
        elif current_sum > Game.TARGET_NUMBER:
            return -1.0
        elif moves_taken >= Game.MAX_MOVES and current_sum != Game.TARGET_NUMBER:
            return -0.5
        return 0.0

# --- MCTS Node ---
class MCTSNode:
    _id_counter = 0 

    def __init__(self, state, parent=None, action_that_led_to_state=None):
        self.id = MCTSNode._id_counter
        MCTSNode._id_counter += 1
        self.state = state
        self.parent = parent 
        self.action_that_led_to_state = action_that_led_to_state
        self.children = [] 
        self.visits = 0
        self.value = 0.0
        self.untried_actions = Game.get_legal_actions(self.state) 

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return Game.is_terminal(self.state)

    def ucb_score(self, exploration_constant=1.414):
        if self.visits == 0:
            return float('inf')
        exploitation_term = self.value / self.visits
        if self.parent is None or self.parent.visits == 0: 
            return exploitation_term 
        exploration_term = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation_term + exploration_term

    def add_child(self, action, child_state):
        child_node = MCTSNode(child_state, parent=self, action_that_led_to_state=action)
        self.children.append(child_node)
        if action in self.untried_actions: 
            self.untried_actions.remove(action)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward
    
    def to_dict_simple(self): 
        return {
            "id": self.id,
            "state": self.state,
            "action_from_parent": self.action_that_led_to_state,
            "visits": self.visits,
            "value": round(self.value, 2)
        }

    def __repr__(self):
        return (f"Node(ID:{self.id}, S:{self.state}, A:{self.action_that_led_to_state}, "
                f"V:{self.value:.2f}, N:{self.visits}, Children:{len(self.children)})")

# --- MCTS Algorithm ---
class MCTS:
    _game_path_node_id_counter = 0

    def __init__(self, initial_game_state_tuple):
        MCTSNode._id_counter = 0 
        self.root = MCTSNode(initial_game_state_tuple)
        self.iteration_count = 0 
        
        MCTS._game_path_node_id_counter = 0 
        self.game_path_history = [{
            "state": initial_game_state_tuple, 
            "action_from_parent": None, 
            "id": f"game_path_{MCTS._game_path_node_id_counter}"
        }]
        MCTS._game_path_node_id_counter += 1


    def _select_promising_node_with_path(self, node):
        current_node = node
        selection_path = [current_node.to_dict_simple()]
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node, selection_path 
            else:
                if not current_node.children: 
                    return current_node, selection_path 
                current_node = max(current_node.children, key=lambda n: n.ucb_score())
                selection_path.append(current_node.to_dict_simple())
        return current_node, selection_path

    def _expand_node(self, node):
        if not node.untried_actions: 
            return None, None 
        action = random.choice(node.untried_actions)
        next_state_tuple = Game.perform_action(node.state, action)
        new_child_node = node.add_child(action, next_state_tuple)
        return new_child_node, action

    def _simulate_random_rollout_with_path(self, node):
        current_rollout_state = node.state
        rollout_path = [] 
        moves = 0
        while not Game.is_terminal(current_rollout_state) and moves < Game.MAX_MOVES * 2 :
            legal_actions = Game.get_legal_actions(current_rollout_state)
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            rollout_path.append({"from_state": current_rollout_state, "action_taken": action})
            current_rollout_state = Game.perform_action(current_rollout_state, action)
            moves += 1
        reward = Game.get_reward(current_rollout_state)
        rollout_path.append({"terminal_state": current_rollout_state, "reward_obtained": reward})
        return reward, rollout_path

    def _backpropagate_with_path(self, node, reward):
        temp_node = node
        backpropagation_path = []
        while temp_node is not None:
            temp_node.update(reward)
            backpropagation_path.append({
                "node_id": temp_node.id,
                "state": temp_node.state,
                "updated_visits": temp_node.visits,
                "updated_value": round(temp_node.value, 2)
            })
            temp_node = temp_node.parent
        return backpropagation_path

    def run_one_iteration_detailed(self):
        self.iteration_count += 1 
        iteration_details = {"iteration_number": self.iteration_count}
        promising_node, selection_path = self._select_promising_node_with_path(self.root)
        iteration_details["selection"] = {
            "path": selection_path, "selected_node_id": promising_node.id,
            "selected_node_state": promising_node.state, "is_terminal": promising_node.is_terminal(),
            "is_fully_expanded": promising_node.is_fully_expanded()
        }
        node_for_rollout = promising_node
        expanded_child_info = None 
        if not promising_node.is_terminal() and promising_node.untried_actions:
            expanded_child, expanded_action = self._expand_node(promising_node)
            if expanded_child:
                node_for_rollout = expanded_child
                expanded_child_info = {"action": expanded_action, "child_id": expanded_child.id, "child_state": expanded_child.state}
                iteration_details["expansion"] = {
                    "parent_node_id": promising_node.id, "action_taken": expanded_action,
                    "new_child_node_id": expanded_child.id, "new_child_state": expanded_child.state
                }
            else: iteration_details["expansion"] = {"error": "Expansion failed"}
        else: iteration_details["expansion"] = {"message": "No expansion (node terminal or fully expanded)."}
        
        app.logger.debug(f"Iter {self.iteration_count}: Selection ended at Node ID {promising_node.id}. Expansion: {expanded_child_info if expanded_child_info else iteration_details['expansion'].get('message', 'N/A')}")

        simulation_reward, rollout_path = self._simulate_random_rollout_with_path(node_for_rollout)
        iteration_details["simulation"] = {
            "start_node_id": node_for_rollout.id, "start_node_state": node_for_rollout.state,
            "rollout_path": rollout_path, "reward": simulation_reward
        }
        app.logger.debug(f"Iter {self.iteration_count}: Simulation from Node ID {node_for_rollout.id} yielded reward {simulation_reward}")

        backpropagation_path = self._backpropagate_with_path(node_for_rollout, simulation_reward)
        iteration_details["backpropagation"] = {
            "start_node_id": node_for_rollout.id, "reward_propagated": simulation_reward,
            "updated_path": backpropagation_path
        }
        app.logger.debug(f"Iter {self.iteration_count}: Backpropagation complete. Root visits: {self.root.visits}, Root value: {self.root.value:.2f}")
        return iteration_details

    def get_best_action(self):
        if not self.root or not self.root.children:
            return None
        best_child_node = max(self.root.children, key=lambda node: node.visits)
        if best_child_node.visits == 0 and len(self.root.children) > 0:
            app.logger.warning(f"Best child (action: {best_child_node.action_that_led_to_state}) has 0 visits. MCTS might need more iterations from current root.")
        return best_child_node.action_that_led_to_state

    def advance_tree(self, action):
        if not self.root:
            app.logger.error("Cannot advance tree: MCTS root is None.")
            return False, "MCTS树未初始化。"
        try:
            action_val = int(action)
        except ValueError:
            app.logger.error(f"Invalid action type for advance_tree: {action}. Expected int.")
            return False, f"提供的行动 '{action}' 类型无效。"

        found_child = None
        for child in self.root.children:
            if child.action_that_led_to_state == action_val:
                found_child = child
                break
        
        if found_child:
            self.root = found_child
            self.root.parent = None 
            self.root.children = [] 
            self.root.untried_actions = Game.get_legal_actions(self.root.state) 
            
            new_path_node_id = f"game_path_{MCTS._game_path_node_id_counter}"
            MCTS._game_path_node_id_counter += 1
            self.game_path_history.append({
                "state": self.root.state,
                "action_from_parent": action_val, 
                "id": new_path_node_id
            })
            app.logger.info(f"Tree advanced. New root is Node ID: {self.root.id}, State: {self.root.state}, Untried Actions: {len(self.root.untried_actions)}, Children: {len(self.root.children)}")
            return True, "树已成功推进到新状态，并已重置其子节点和未尝试行动列表。"
        else:
            app.logger.warning(f"Action {action_val} not found among children of current root (ID: {self.root.id}). Children actions: {[c.action_that_led_to_state for c in self.root.children]}")
            return False, f"行动 {action_val} 不是当前根节点的有效子行动。"

    def get_game_path_tree_data(self):
        if not self.game_path_history: return {}
        history_root_data = self.game_path_history[0]
        display_tree_root = {
            "id": history_root_data["id"], "state": history_root_data["state"],
            "action_from_parent": history_root_data.get("action_from_parent"),
            "is_current_game_node": len(self.game_path_history) == 1,
            "value": "N/A", "visits": "N/A", "avg_value": "N/A", "ucb_score_from_parent": "N/A",
            "is_terminal": Game.is_terminal(history_root_data["state"]),
            "is_fully_expanded": "N/A", "displayed_children": []
        }
        current_parent_in_display_tree = display_tree_root
        for i in range(1, len(self.game_path_history)):
            history_node_data = self.game_path_history[i]
            is_current = (i == len(self.game_path_history) - 1)
            child_display_node = {
                "id": history_node_data["id"], "state": history_node_data["state"],
                "action_from_parent": history_node_data.get("action_from_parent"),
                "is_current_game_node": is_current,
                "value": "N/A", "visits": "N/A", "avg_value": "N/A", "ucb_score_from_parent": "N/A",
                "is_terminal": Game.is_terminal(history_node_data["state"]),
                "is_fully_expanded": "N/A", "displayed_children": []
            }
            current_parent_in_display_tree["displayed_children"].append(child_display_node)
            current_parent_in_display_tree = child_display_node 
        return display_tree_root

    def _get_node_data_recursive(self, node, current_depth, max_depth):
        if node is None or current_depth > max_depth: return None
        children_data_list = []
        if current_depth < max_depth:
            for child_node in node.children:
                child_data = self._get_node_data_recursive(child_node, current_depth + 1, max_depth)
                if child_data: children_data_list.append(child_data)
        return {
            "id": node.id, "state": node.state,
            "action_from_parent": node.action_that_led_to_state,
            "visits": node.visits, "value": f"{node.value:.2f}",
            "avg_value": f"{(node.value / node.visits) if node.visits > 0 else 'N/A'}",
            "ucb_score_from_parent": f"{node.ucb_score():.2f}" if node.parent else "N/A (Root)",
            "is_terminal": node.is_terminal(), "is_fully_expanded": node.is_fully_expanded(),
            "untried_actions_count": len(node.untried_actions),
            "children_count": len(node.children), "displayed_children": children_data_list
        }

    def get_tree_visualization_data(self, max_display_depth=2):
        if self.root is None: return {}
        return self._get_node_data_recursive(self.root, 0, max_display_depth)

# --- Flask Application ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG) 
app.logger.setLevel(logging.DEBUG)

mcts_algorithm = None
current_game_settings = {} 

GEMINI_API_KEY = "" 
GEMINI_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini_api(prompt_text):
    api_url = f"{GEMINI_API_URL_BASE}?key={GEMINI_API_KEY}"
    chat_history = [{"role": "user", "parts": [{"text": prompt_text}]}]
    payload = {"contents": chat_history}
    app.logger.info(f"Calling Gemini API with prompt (first 200 chars): {prompt_text[:200]}...")
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
        response.raise_for_status()
        result = response.json()
        app.logger.debug(f"Gemini API raw response: {result}")
        if (result.get("candidates") and
            result["candidates"][0].get("content") and
            result["candidates"][0]["content"].get("parts") and
            result["candidates"][0]["content"]["parts"][0].get("text")):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            app.logger.error(f"Unexpected Gemini API response structure: {result}")
            return "AI未能按预期格式响应。"
    except requests.exceptions.Timeout:
        app.logger.error("Gemini API request timed out.")
        raise Exception("AI 服务请求超时")
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Gemini API request failed: {e}")
        raise Exception(f"AI 服务请求失败: {str(e)}")
    except Exception as e:
        app.logger.error(f"Error processing Gemini response or other unexpected error: {e}")
        raise Exception(f"处理AI响应时出错: {str(e)}")

@app.route('/init_game', methods=['POST'])
def init_game_endpoint():
    global mcts_algorithm, current_game_settings
    try:
        data = request.json
        Game.TARGET_NUMBER = int(data.get('target', 10))
        Game.MAX_MOVES = int(data.get('max_moves', 7))
        Game.POSSIBLE_ACTIONS = [int(a) for a in data.get('actions', [1, 2, 3])]
        current_game_settings = {
            "target": Game.TARGET_NUMBER, "max_moves": Game.MAX_MOVES, "actions": Game.POSSIBLE_ACTIONS
        }
        app.logger.info(f"Game initialized with settings: {current_game_settings}")
        initial_state = Game.get_initial_state()
        mcts_algorithm = MCTS(initial_state) 
        app.logger.info(f"Initial MCTS root: ID {mcts_algorithm.root.id}, State {mcts_algorithm.root.state}, Untried: {len(mcts_algorithm.root.untried_actions)}, Children: {len(mcts_algorithm.root.children)}")
        return jsonify({
            "message": "游戏初始化成功。", "settings": current_game_settings,
            "initial_state": initial_state,
            "is_initial_terminal": Game.is_terminal(initial_state), 
            "tree_root": mcts_algorithm.get_tree_visualization_data(max_display_depth=2),
            "game_path_tree": mcts_algorithm.get_game_path_tree_data() 
        }), 200
    except Exception as e:
        app.logger.error(f"Initialization failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"初始化失败: {str(e)}"}), 400

@app.route('/run_mcts_iterations', methods=['POST']) 
def run_mcts_iterations_endpoint():
    global mcts_algorithm
    if not mcts_algorithm: return jsonify({"error": "游戏未初始化。"}), 400
    app.logger.info(f"Entering /run_mcts_iterations. Root ID: {mcts_algorithm.root.id}, State: {mcts_algorithm.root.state}, Untried: {len(mcts_algorithm.root.untried_actions)}, Children: {len(mcts_algorithm.root.children)}")
    try:
        data = request.json
        num_iterations = int(data.get('iterations', 1))
        if num_iterations <= 0: return jsonify({"error": "迭代次数必须为正数。"}), 400
        for i in range(num_iterations):
            app.logger.debug(f"--- Batch Iteration {i+1}/{num_iterations} for Root ID {mcts_algorithm.root.id} ---")
            promising_node, _ = mcts_algorithm._select_promising_node_with_path(mcts_algorithm.root)
            node_for_rollout = promising_node
            if not promising_node.is_terminal() and promising_node.untried_actions:
                expanded_child, expanded_action = mcts_algorithm._expand_node(promising_node)
                if expanded_child: 
                    node_for_rollout = expanded_child
                    app.logger.debug(f"Batch Iter {i+1}: Expanded Node ID {promising_node.id} with action {expanded_action} to new child ID {expanded_child.id}")
                else: 
                    app.logger.warning(f"Batch Iter {i+1}: Expansion failed for Node ID {promising_node.id} despite having untried_actions.")
            else:
                 app.logger.debug(f"Batch Iter {i+1}: No expansion for Node ID {promising_node.id} (Terminal: {promising_node.is_terminal()}, Untried: {len(promising_node.untried_actions)})")

            simulation_reward, _ = mcts_algorithm._simulate_random_rollout_with_path(node_for_rollout)
            mcts_algorithm._backpropagate_with_path(node_for_rollout, simulation_reward)
            mcts_algorithm.iteration_count +=1 
        app.logger.info(f"{num_iterations} MCTS batch iterations completed. Total user iterations: {mcts_algorithm.iteration_count}")
        app.logger.info(f"After batch iterations. Root ID: {mcts_algorithm.root.id}, State: {mcts_algorithm.root.state}, Untried: {len(mcts_algorithm.root.untried_actions)}, Children: {len(mcts_algorithm.root.children)}")
        return jsonify({
            "message": f"{num_iterations} 次批量 MCTS 迭代完成。总用户迭代次数: {mcts_algorithm.iteration_count}",
            "tree_root": mcts_algorithm.get_tree_visualization_data(max_display_depth=3),
            "root_stats": {"visits": mcts_algorithm.root.visits, "value": f"{mcts_algorithm.root.value:.2f}"}
        }), 200
    except Exception as e:
        app.logger.error(f"Error during MCTS batch iterations: {str(e)}", exc_info=True)
        return jsonify({"error": f"MCTS 批量迭代过程中出错: {str(e)}"}), 500

@app.route('/run_single_iteration_detailed', methods=['POST'])
def run_single_iteration_detailed_endpoint():
    global mcts_algorithm
    if not mcts_algorithm: return jsonify({"error": "游戏未初始化。"}), 400
    app.logger.info(f"Entering /run_single_iteration_detailed. Root ID: {mcts_algorithm.root.id}, State: {mcts_algorithm.root.state}, Untried: {len(mcts_algorithm.root.untried_actions)}, Children: {len(mcts_algorithm.root.children)}")
    try:
        detailed_steps = mcts_algorithm.run_one_iteration_detailed() 
        app.logger.info(f"Detailed steps for user iteration {mcts_algorithm.iteration_count} completed.")
        app.logger.info(f"After single detailed iteration. Root ID: {mcts_algorithm.root.id}, State: {mcts_algorithm.root.state}, Untried: {len(mcts_algorithm.root.untried_actions)}, Children: {len(mcts_algorithm.root.children)}")
        return jsonify({
            "message": f"单步 MCTS 迭代 {mcts_algorithm.iteration_count} 完成。",
            "detailed_steps": detailed_steps,
            "tree_root": mcts_algorithm.get_tree_visualization_data(max_display_depth=3)
        }), 200
    except Exception as e:
        app.logger.error(f"Error during single detailed MCTS iteration: {str(e)}", exc_info=True)
        return jsonify({"error": f"MCTS 单步详细迭代过程中出错: {str(e)}"}), 500

@app.route('/get_best_move', methods=['GET'])
def get_best_move_endpoint():
    global mcts_algorithm
    if not mcts_algorithm or not mcts_algorithm.root: return jsonify({"error": "游戏未初始化或没有 MCTS 数据。"}), 400
    app.logger.info(f"Entering /get_best_move. Root ID: {mcts_algorithm.root.id}, State: {mcts_algorithm.root.state}, Untried: {len(mcts_algorithm.root.untried_actions)}, Children: {len(mcts_algorithm.root.children)}")
    
    if not mcts_algorithm.root.children:
        if Game.is_terminal(mcts_algorithm.root.state):
            message = "当前状态已是终止状态，无法获取下一步行动。"
        else:
            message = "根节点没有探索任何行动。请运行更多迭代或检查游戏状态。"
        app.logger.warning(f"/get_best_move: {message} for Root ID {mcts_algorithm.root.id}")
        return jsonify({
            "message": message, "best_action": None,
            "current_sum": mcts_algorithm.root.state[0] if mcts_algorithm.root.state else 'N/A',
            "moves_taken": mcts_algorithm.root.state[1] if mcts_algorithm.root.state and len(mcts_algorithm.root.state) > 1 else 'N/A',
            "details": "根节点没有子节点或已是终止状态。"
        }), 200

    best_action = mcts_algorithm.get_best_action()
    if best_action is None and mcts_algorithm.root.children: 
        app.logger.error("CRITICAL: get_best_action returned None even though root has children!")
        return jsonify({
            "message": "内部错误：无法在有子节点的情况下确定最佳行动。", "best_action": None,
            "current_sum": mcts_algorithm.root.state[0], "moves_taken": mcts_algorithm.root.state[1],
            "details": "内部逻辑错误。"
        }), 500

    children_info = []
    for child in mcts_algorithm.root.children:
        children_info.append({
            "action": child.action_that_led_to_state, "visits": child.visits, "value": f"{child.value:.2f}",
            "avg_value": f"{(child.value / child.visits) if child.visits > 0 else 'N/A'}"
        })
    app.logger.info(f"Best move suggested: {best_action} from Root ID {mcts_algorithm.root.id}")
    return jsonify({
        "message": "MCTS 建议的最佳行动。", "best_action": best_action,
        "current_sum": mcts_algorithm.root.state[0] if mcts_algorithm.root.state else 'N/A',
        "moves_taken": mcts_algorithm.root.state[1] if mcts_algorithm.root.state and len(mcts_algorithm.root.state) > 1 else 'N/A',
        "root_children_stats": sorted(children_info, key=lambda x: x["visits"], reverse=True)
    }), 200

@app.route('/apply_move', methods=['POST'])
def apply_move_endpoint():
    global mcts_algorithm
    if not mcts_algorithm:
        return jsonify({"error": "游戏未初始化。"}), 400
    try:
        data = request.json
        action_to_apply = data.get("action")
        if action_to_apply is None:
            return jsonify({"error": "请求中未提供行动。"}), 400
        try:
            action_to_apply_int = int(action_to_apply)
        except ValueError:
            return jsonify({"error": f"行动 '{action_to_apply}' 必须是有效的数字。"}), 400

        success, message = mcts_algorithm.advance_tree(action_to_apply_int)

        if success:
            new_root_state = mcts_algorithm.root.state
            is_terminal_now = Game.is_terminal(new_root_state)
            
            # Removed AUTO_ITERATIONS_AFTER_MOVE block
            app.logger.info(f"Auto-iterations after move have been REMOVED.")
            app.logger.info(f"State of new root (ID: {mcts_algorithm.root.id}) after advancing tree: Children: {len(mcts_algorithm.root.children)}, Untried: {len(mcts_algorithm.root.untried_actions)}")


            current_sum_after_move = new_root_state[0]
            moves_taken_after_move = new_root_state[1]
            app.logger.info(f"Move {action_to_apply_int} applied. New state: {new_root_state}, Terminal: {is_terminal_now}")
            return jsonify({
                "message": f"行动 {action_to_apply_int} 已执行。{message}",
                "new_root_state": new_root_state, "current_sum": current_sum_after_move,
                "moves_taken": moves_taken_after_move, "is_terminal": is_terminal_now,
                "tree_root": mcts_algorithm.get_tree_visualization_data(max_display_depth=2),
                "game_path_tree": mcts_algorithm.get_game_path_tree_data() 
            }), 200
        else:
            app.logger.warning(f"Failed to apply move {action_to_apply_int}: {message}")
            return jsonify({
                "error": message,
                "tree_root": mcts_algorithm.get_tree_visualization_data(max_display_depth=2),
                "game_path_tree": mcts_algorithm.get_game_path_tree_data()
                 }), 400
    except Exception as e:
        app.logger.error(f"Error in /apply_move: {str(e)}", exc_info=True)
        return jsonify({"error": f"应用行动时出错: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)

```

</details>

<iframe src="/mcts_frontend.html" width="100%" height="1200"></iframe>