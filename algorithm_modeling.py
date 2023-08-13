"""
问题建模: 将问题建模为网络拆解"游戏"
"""

from enum import Enum
from typing import List
from collections import deque
import random
from copy import deepcopy

import torch
from torch import Tensor
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.toolbox import (
    ProblemType,
    ProblemMetric,
    GraphTool,
)


class Action:
    class ActionType(Enum):
        DISCONNECT = 0
        CONNECT = 1

    def __init__(self, targetNodeIndex: int, actionType=ActionType.DISCONNECT):
        self.targetNodeIndex = targetNodeIndex
        self.actionType = actionType

    def __str__(self):
        return f"Action(targetNodeIndex={self.targetNodeIndex}, actionType={self.actionType})"


class State:
    def __init__(self, binaryInfo: List):
        # 0代表节点仍被连接, 1代表节点已经断连
        for ele in binaryInfo:
            if ele != 0 and ele != 1:
                raise ValueError("每个值都应该是0或1")
        self._binaryInfo = binaryInfo

    def __len__(self):
        return len(self._binaryInfo)

    def __getitem__(self, index):
        return self._binaryInfo[index]

    def __str__(self):
        return str(self._binaryInfo)

    @property
    def isTerminal(self):
        if 1 in self._binaryInfo:
            return False
        else:
            return True

    def exec_action(self, action: Action):
        if action.targetNodeIndex > self.__len__() - 1:
            raise ValueError("操作的节点索引超出")
        if action.actionType == Action.ActionType.DISCONNECT:
            # 行为是断连
            if self._binaryInfo[action.targetNodeIndex] == 0:
                # 目标节点已经断连
                raise ValueError("action多余，targetNode已被断连")
            self._binaryInfo[action.targetNodeIndex] = 0
        elif action.actionType == Action.ActionType.CONNECT:
            # 行为是连接
            if self._binaryInfo[action.targetNodeIndex] == 1:
                # 目标节点已经连接
                raise ValueError("action多余，targetNode已被连接")
            self._binaryInfo[action.targetNodeIndex] = 1

    def get_random_action(self, actionType: Action.ActionType):
        if actionType == Action.ActionType.DISCONNECT:
            available_targetNodeIndex = [
                index for index, value in enumerate(self._binaryInfo) if value == 1
            ]
            if available_targetNodeIndex:
                targetNodeIndex = random.choice(available_targetNodeIndex)
            else:
                raise ValueError("所有节点已经都被断连了，不存在可断连节点")
            return Action(targetNodeIndex=targetNodeIndex, actionType=actionType)
        elif actionType == Action.ActionType.CONNECT:
            raise ValueError("暂不支持CONNECT的动作类型")

    def get_best_action(
        self,
        values: Tensor,
        actionType: Action.ActionType = Action.ActionType.DISCONNECT,
    ):
        """获得Q最大的可行action"""
        if actionType == Action.ActionType.DISCONNECT:
            _, indices = torch.sort(values, descending=True)
            sorted_binaryInfo = [self._binaryInfo[index] for index in indices]
            if not self.isTerminal:
                index = sorted_binaryInfo.index(1)
                targetNodeIndex = indices[index].item()
            else:
                raise ValueError("无可选的action【因为所有的节点都已断连】")
            action = Action(targetNodeIndex=targetNodeIndex, actionType=actionType)
            return action
        elif actionType == Action.ActionType.CONNECT:
            raise ValueError("暂不支持CONNECT的动作类型")


def get_stateData(state: State, originData: Data, require_grad: bool = False):
    """获得状态的对应data[修改原始data的edge_index]"""
    # 复制一份原始data
    stateData = deepcopy(originData)
    # 得到原始图和原始节点
    graph: nx.Graph = to_networkx(originData)
    nodes = list(graph.nodes)
    # 按照state断连节点, 原地修改graph
    for nodeIndex in range(len(state)):
        # NOTE: 考虑到此时的graph是节点映射后的，因此其节点标签与state中的索引一一对应, 其实只需要按照state中的零值断连索引节点就可以
        if state[nodeIndex] == 0:
            GraphTool.disconnect_node(graph, nodes[nodeIndex])
    # 根据修改后的graph生成新的edge_index
    edge_index = GraphTool.get_edgeIndex_from_graph(graph)
    stateData.edge_index = edge_index
    # 确定是否需要记录梯度
    stateData.x.requires_grad = require_grad
    return stateData


class Game:
    def __init__(
        self,
        originGraph: nx.Graph,
        roundNum: int = 1_000,
        problemType: ProblemType = ProblemType.CN,
        verbose: bool = False,
    ):
        self.originGraph = originGraph
        self.roundNum = roundNum
        self.problemType = problemType
        self.verbose = verbose

        # 设置memory
        # NOTE: memorySize 应该跟图的大小相关，或者说一个data对应一个memorySize
        nodeNum = len(self.originGraph.nodes)
        self.memory = deque(maxlen=roundNum * nodeNum * nodeNum)

        # 初始化round
        self.round_index = 0
        self.round = self.get_round()

    def __str__(self):
        return f"Game(roundNum={self.roundNum}, nodeNum={len(self.originGraph.nodes)}, problem={self.problemType.name})"

    def play_step(self, action: Action):
        # 当前round进行一次
        reward, roundDone, score = self.round.play_step(action)
        # 判断game是否结束[所以round都走完]
        gameDone = False
        # 判断round是否结束，如果结束就开新的round
        if roundDone:
            if self.verbose:
                print(f"Round {self.round_index} Done...")
            if self.round_index == self.roundNum - 1:
                gameDone = True
            else:
                self.round_index += 1
                self.round = self.get_round()
        return reward, roundDone, gameDone, score

    def get_round(self):
        if self.verbose:
            print("roundIndex:", self.round_index)
            print("originGraph:", self.originGraph)
        return self.Round(self.originGraph, problemType=self.problemType)

    def remember(
        self,
        oldState: State,
        reward: float,
        action: Action,
        newState: State,
    ):
        self.memory.append(
            (oldState, reward, action, newState)
        )  # popleft if memorySize is reached

    class Round:
        def __init__(
            self, originGraph: nx.Graph, problemType: ProblemType = ProblemType.CN
        ):
            self.graph = deepcopy(originGraph)

            # 获得节点序列
            self.nodes = list(self.graph.nodes)
            # 设置指标
            self.metricFn = ProblemMetric.get(problemType)
            # 初始化状态
            self.state = State([1] * len(self.graph.nodes))
            # 获取原图连接性
            self.baseConnectivity = self.metricFn(self.graph)
            # 连接性记录
            self.connectivityRecord = []

        def _get_reward(self) -> tuple[float, float]:
            """计算reward

            Returns: (reward, ANC) tuple
            """
            # 计算当前时刻的ANC【包含当前状态在内所有状态连接性的均值】
            currentValue = np.mean(
                [
                    connectivity / self.baseConnectivity
                    for connectivity in self.connectivityRecord
                ]
            )
            # 计算前一时刻的ANC【不包含当前状态的所有状态连接性的均值】
            if len(self.connectivityRecord) == 1:
                lastValue = 1
            else:
                lastValue = np.mean(
                    [
                        connectivity / self.baseConnectivity
                        for connectivity in self.connectivityRecord[:-1]
                    ]
                )
            return lastValue - currentValue, currentValue

        def play_step(self, action: Action):
            if action.actionType == Action.ActionType.DISCONNECT:
                targetNodeIndex = action.targetNodeIndex

                # state执行action[判决action可行性, 更新state]
                self.state.exec_action(action)
                # 更新图数据
                GraphTool.disconnect_node(self.graph, self.nodes[targetNodeIndex])
                # 添加连接性记录
                self.connectivityRecord.append(self.metricFn(self.graph))
                # 计算奖励和当前总分
                reward, score = self._get_reward()
                # 判断游戏是否结束
                if self.state.isTerminal:
                    roundDone = True
                else:
                    roundDone = False
            else:
                raise ValueError("暂不支持connect动作")

            return reward, roundDone, score


if __name__ == "__main__":
    from torch_geometric.utils import to_networkx

    from data_processing import DatasetLoader

    dataset = DatasetLoader.load_synthetic_dataset(
        "SyntheticDataset-N1", problemType=ProblemType.CN
    )
    game = Game(
        to_networkx(dataset[0], to_undirected=True),
        roundNum=1_000,
        problemType=ProblemType.CN,
    )
    print(game)
