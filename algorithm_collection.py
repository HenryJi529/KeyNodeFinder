"""
针对关键节点识别问题的算法集合
"""

from typing import Callable, Union, List, Dict, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cache
from queue import Queue
from itertools import permutations
import random
from math import factorial
import argparse
from pathlib import Path

import networkx as nx
from tqdm.auto import tqdm
import torch
from torch import Tensor
from torch_geometric.data import Batch

from utils.toolbox import (
    ProblemType,
    ProblemMetric,
    GraphTool,
)
from utils.structures import Tree
from utils.common import timeit, colored_print
from utils.toolbox import ProblemType
from model_builder import NiceModel, Task
from model_handler import load_model_info
from algorithm_modeling import Action, State, get_stateData
from algorithm_criterion import AccumulatedNormalizedConnectivityCriterion


class _Algorithm(ABC):
    NAME = None

    @property
    def name(self):
        if isinstance(self.NAME, str):
            return self.NAME
        else:
            raise ValueError("Name must be a string")

    @abstractmethod
    def get_order_by_problemType(
        self, graph: nx.Graph, problemType: ProblemType
    ) -> List:
        pass

    def get_anc(self, graph: nx.Graph, problemType: ProblemType):
        criterion = AccumulatedNormalizedConnectivityCriterion(graph)
        order = self.get_order_by_problemType(graph, problemType)
        return criterion(order, problemType)


class _MetricBasedAlgorithm(_Algorithm):
    def _get_state_value(
        self,
        graph: nx.Graph,
        disconnectedNodeList: List,
        metric_fn: Callable[[nx.Graph], Union[int, float]],
    ):
        G = deepcopy(graph)
        for node in disconnectedNodeList:
            GraphTool.disconnect_node(G, node)
        return metric_fn(G)

    @abstractmethod
    def _get_order_by_metric(
        self, graph: nx.Graph, metric_fn: Callable[[nx.Graph], Union[int, float]]
    ) -> List:
        pass

    def get_order_by_problemType(
        self, graph: nx.Graph, problemType: ProblemType
    ) -> List:
        metricFn = ProblemMetric.get(problemType)
        return self._get_order_by_metric(graph, metricFn)


class _TopologyBasedAlgorithm(_Algorithm):
    @abstractmethod
    def _get_order_from_topology(self, graph: nx.Graph) -> List:
        pass

    def get_order_by_problemType(
        self, graph: nx.Graph, problemType: ProblemType
    ) -> List:
        return self._get_order_from_topology(graph)


class _RandomMixIn:
    def __init__(self, random: float):
        self.random = random

    def update_order(self, order: List):
        for _ in range(int(len(order) * self.random)):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        return order


class _AdaptiveMixIn(ABC):
    @abstractmethod
    def _get_valueDict_from_topology(self, graph: nx.Graph):
        pass

    def get_order_by_valueDict(self, graph: nx.Graph):
        G = graph.copy()
        pbar = tqdm(
            total=len(G.nodes),
            leave=False,
        )
        order = []
        while len(G.nodes) > 0:
            valueDict = self._get_valueDict_from_topology(G)
            node = max(valueDict, key=valueDict.get)
            order.append(node)
            pbar.update(1)
            G.remove_node(node)
        pbar.close()
        return order


class BruteForceAlgorithm(_Algorithm):
    """直接暴力搜索所有可能的拆除次序"""

    NAME = "BruteForce"

    class DisconnectedOrderIterator:
        def __init__(self, graph: nx.Graph):
            self.nodes = graph.nodes
            self.permutations = permutations(self.nodes)

        def __iter__(self):
            return self

        def __next__(self):
            try:
                next_permutation = next(self.permutations)
                return list(next_permutation)
            except StopIteration:
                raise StopIteration

    def get_order_by_problemType(
        self, graph: nx.Graph, problemType: ProblemType
    ) -> List:
        best_order, best_value = None, None
        criterion = AccumulatedNormalizedConnectivityCriterion(graph)
        iterator = iter(self.DisconnectedOrderIterator(graph))
        for current_order in tqdm(
            iterator,
            total=factorial(len(graph.nodes)),
            desc=f"{self.NAME}",
            leave=False,
        ):
            current_value = criterion(current_order, problemType)[0]
            if not best_order or best_value > current_value:
                best_order = current_order
                best_value = current_value
        return best_order


class NaiveAlgorithm(_MetricBasedAlgorithm):
    NAME = "Naive"

    def _get_order_by_metric(
        self, graph: nx.Graph, metric_fn: Callable[[nx.Graph], Union[int, float]]
    ) -> List:
        def get_nextStates(state: Tuple[bool]):
            nextStates = []
            for ind in range(len(state)):
                if state[ind] == True:
                    nextState = list(deepcopy(state))
                    nextState[ind] = False
                    nextStates.append(tuple(nextState))
            return nextStates

        def get_disconnectedNodeList(state: Tuple[bool], reversedNodeMapping: Dict):
            disconnectedNodeList = []
            for ind in range(len(state)):
                if state[ind] == False:
                    disconnectedNodeList.append(reversedNodeMapping[ind])
            return disconnectedNodeList

        def get_order(shortestPath: List[Tuple], reversedNodeMapping: Dict):
            mappedOrder = []
            for state in shortestPath[1:]:
                for ind in range(len(state)):
                    if state[ind] == False:
                        if ind not in mappedOrder:
                            mappedOrder.append(ind)
                            break
            realOrder = [reversedNodeMapping[item] for item in mappedOrder]
            return realOrder

        nodes = graph.nodes
        nodeMapping = {node: i for i, node in enumerate(nodes)}
        reversedNodeMapping = {value: key for key, value in nodeMapping.items()}
        valueDict = {}
        stateGraph = nx.DiGraph()

        # 状态图构建
        initialState = tuple([True] * len(nodes))
        terminalState = tuple([False] * len(nodes))
        q: Queue = Queue()
        q.put(initialState)
        pbar = tqdm(
            total=2 ** len(nodes),
            desc=f"{self.NAME}(calcState)",
            leave=False,
        )
        while True:
            state = q.get()
            if state == terminalState:
                pbar.update(1)
                pbar.close()
                break
            for nextState in get_nextStates(state):
                if nextState not in valueDict:
                    q.put(nextState)
                    valueDict[nextState] = self._get_state_value(
                        graph,
                        get_disconnectedNodeList(nextState, reversedNodeMapping),
                        metric_fn=metric_fn,
                    )
                    pbar.update(1)
                stateGraph.add_edge(state, nextState, weight=valueDict[nextState])

        # 计算状态图的最短路径
        shortestPath = nx.shortest_path(
            stateGraph, source=initialState, target=terminalState, weight="weight"
        )

        # 获取对应的关键节点排序
        order = get_order(shortestPath, reversedNodeMapping)

        return order


class GreedyAlgorithm(_MetricBasedAlgorithm):
    NAME = "Greedy"

    def _get_order_by_metric(
        self, graph: nx.Graph, metric_fn: Callable[[nx.Graph], Union[int, float]]
    ) -> List:
        order = []
        nodeList = list(graph.nodes)
        for _ in tqdm(
            range(graph.number_of_nodes()),
            desc=f"{self.NAME}",
            leave=False,
        ):
            currentChoice = {"node": None, "value": None}
            for node in nodeList:
                if node not in order:
                    value = self._get_state_value(graph, order + [node], metric_fn)
                    if not currentChoice["value"] or value < currentChoice["value"]:
                        currentChoice["node"] = node
                        currentChoice["value"] = value
            order.append(currentChoice["node"])
        return order


class GreedyAlgorithmV2(_MetricBasedAlgorithm):
    NAME = "Greedy"

    def __init__(self, greedy_factor: Union[float, int] = 1):
        self.greedy_factor = greedy_factor

    def _get_order_by_metric(
        self, graph: nx.Graph, metric_fn: Callable[[nx.Graph], Union[int, float]]
    ) -> List:
        def get_disconnected_nodes(tree: Tree, leafNode):
            return list(
                reversed(
                    [
                        treeNode.value
                        for treeNode in tree.find_branch(leafNode)
                        if treeNode.value != ""
                    ]
                )
            )

        def get_order(tree: Tree):
            return list(
                reversed(
                    [
                        treeNode.value
                        for treeNode in tree.find_branch(tree.find_leaves()[0])
                        if treeNode.value != ""
                    ]
                )
            )

        def cut_branch(tree: Tree, depth: int, greedy_factor: Union[int, float]):
            def get_end_index(greedy_factor):
                """计算贪心因子"""
                if type(greedy_factor) == int:
                    end_index = greedy_factor
                else:
                    end_index = (
                        int(len(leaves) * greedy_factor)
                        if int(len(leaves) * greedy_factor) >= 1
                        else 1
                    )
                return end_index

            leaves = tree.find_leaves()
            # 首先清除节点深度小的
            for leaf in leaves:
                if tree.get_leaf_depth(leaf) < depth:
                    tree.remove_branch(leaf)
            # 随机保留满足贪心因子的branch
            leaves = tree.find_leaves()
            random.shuffle(leaves)
            end_index = get_end_index(greedy_factor)
            for leaf in leaves[end_index:]:
                tree.remove_branch(leaf)

        tree = Tree()
        tree.add_node("")
        nodeList = list(graph.nodes)
        for depth in tqdm(
            range(len(nodeList)),
            desc=f"{self.NAME}",
            leave=False,
        ):
            # 扩展叶子
            bestValue = None
            for leafNode in tree.find_leaves():
                currentChoice = {"nodes": [], "value": None}

                disconnectedNodeList = get_disconnected_nodes(tree, leafNode)

                for node in nodeList:
                    if node not in disconnectedNodeList:
                        value = self._get_state_value(
                            graph, disconnectedNodeList + [node], metric_fn
                        )
                        if not currentChoice["value"] or value < currentChoice["value"]:
                            currentChoice["nodes"] = [node]
                            currentChoice["value"] = value
                        elif value == currentChoice["value"]:
                            currentChoice["nodes"].append(node)
                        else:
                            pass
                if not bestValue or currentChoice["value"] <= bestValue:
                    for node in currentChoice["nodes"]:
                        tree.add_node(node, parent=leafNode)
                else:
                    tree.remove_branch(leafNode)

            # 保持一定的贪心度，继续裁剪枝干
            cut_branch(tree, depth + 2, self.greedy_factor)

        # 获得任意一种最佳方案
        order = get_order(tree)
        return order


class PureRandomAlgorithm(_TopologyBasedAlgorithm):
    NAME = "Random"

    def _get_order_from_topology(self, graph: nx.Graph):
        nodeList = list(graph.nodes)
        order = random.sample(nodeList, len(nodeList))
        return order


class PageRankAlgorithm(_TopologyBasedAlgorithm):
    NAME = "PageRank"

    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        return nx.pagerank(graph, alpha=self.alpha)

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class RandomPageRankAlgorithm(PageRankAlgorithm, _RandomMixIn):
    NAME = "RandomPageRank"

    def __init__(self, alpha: float = 0.85, random: float = 0.2):
        PageRankAlgorithm.__init__(self, alpha)
        _RandomMixIn.__init__(self, random)

    def _get_order_from_topology(self, graph: nx.Graph):
        pagerankOrder = PageRankAlgorithm._get_order_from_topology(self, graph)
        return _RandomMixIn.update_order(self, pagerankOrder)


class HPRAAlgorithm(PageRankAlgorithm, _AdaptiveMixIn):
    NAME = "HPRA"

    def __init__(self, alpha: float = 0.85):
        PageRankAlgorithm.__init__(self, alpha)

    def _get_order_from_topology(self, graph: nx.Graph):
        return _AdaptiveMixIn.get_order_by_valueDict(self, graph)


class LeaderRankAlgorithm(_TopologyBasedAlgorithm):
    """LeaderRank: https://blog.csdn.net/DreamHome_S/article/details/79468188"""

    NAME = "LeaderRank"

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = GraphTool.leader_rank(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class CIAlgorithm(_TopologyBasedAlgorithm):
    NAME = "CI"

    def __init__(self, radius: int = 2):
        self.radius = radius

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        if self.radius == 0:
            return ValueError("radius必须为正整数")
        all_pairs_distances = dict(nx.all_pairs_shortest_path_length(graph))
        CI = {}
        for node in graph.nodes:
            CI[node] = (graph.degree(node) - 1) * sum(
                [
                    graph.degree(n) - 1
                    for n in all_pairs_distances[node]
                    if all_pairs_distances[node][n] == self.radius
                ]
            )
        return CI

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class HCIAAlgorithm(CIAlgorithm, _AdaptiveMixIn):
    def __init__(self, radius: float = 2):
        CIAlgorithm.__init__(self, radius)

    def _get_order_from_topology(self, graph: nx.Graph):
        return _AdaptiveMixIn.get_order_by_valueDict(self, graph)


class HDAlgorithm(_TopologyBasedAlgorithm):
    NAME = "HD"

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        return dict(graph.degree)

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class HDAAlgorithm(HDAlgorithm, _AdaptiveMixIn):
    NAME = "HDA"

    def _get_order_from_topology(self, graph: nx.Graph):
        return _AdaptiveMixIn.get_order_by_valueDict(self, graph)


class KCoreAlgorithm(_TopologyBasedAlgorithm):
    NAME = "KCore"

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        return nx.core_number(graph)

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class HITSAlgorithm(_TopologyBasedAlgorithm):
    NAME = "HITS"

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict, _ = nx.hits(graph, max_iter=100, normalized=True)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class BetweennessCentralityAlgorithm(_TopologyBasedAlgorithm):
    NAME = "BC"

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        return nx.centrality.betweenness_centrality(graph)

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class HBAAlgorithm(BetweennessCentralityAlgorithm, _AdaptiveMixIn):
    NAME = "HBA"

    def _get_order_from_topology(self, graph: nx.Graph):
        return _AdaptiveMixIn.get_order_by_valueDict(self, graph)


class EigenvectorCentralityAlgorithm(_TopologyBasedAlgorithm):
    NAME = "EC"

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        return nx.centrality.eigenvector_centrality(graph, max_iter=int(1e10))

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class HEAAlgorithm(EigenvectorCentralityAlgorithm, _AdaptiveMixIn):
    NAME = "HEA"

    def _get_order_from_topology(self, graph: nx.Graph):
        return _AdaptiveMixIn.get_order_by_valueDict(self, graph)


class ClosenessCentralityAlgorithm(_TopologyBasedAlgorithm):
    NAME = "CC"

    def _get_valueDict_from_topology(self, graph: nx.Graph):
        return nx.centrality.closeness_centrality(graph)

    def _get_order_from_topology(self, graph: nx.Graph):
        valueDict = self._get_valueDict_from_topology(graph)
        return sorted(valueDict, key=lambda key: valueDict[key], reverse=True)


class HCAAlgorithm(ClosenessCentralityAlgorithm, _AdaptiveMixIn):
    NAME = "HCA"

    def _get_order_from_topology(self, graph: nx.Graph):
        return _AdaptiveMixIn.get_order_by_valueDict(self, graph)


class ProposedAlgorithm(_Algorithm):
    NAME = "Proposed"

    def __init__(
        self,
        modelFilenameDict: Dict[ProblemType, str] = None,
        targetDir: Path = Path(__file__).parent / "models",
    ):
        self.modelDict = {}
        self.hyperparametersDict = {}
        self.evaluateResultDict = {}
        for problemType in list(ProblemType):
            if modelFilenameDict.get(problemType) is None:
                modelFilename = (
                    f"{NiceModel.__class__.__name__}_{problemType.name}_latest.pth"
                )
            else:
                modelFilename = modelFilenameDict.get(problemType)
            model, hyperparameters, evaluateResult = load_model_info(
                targetDir=targetDir,
                modelFilename=modelFilename,
            )
            self.modelDict[problemType] = model
            self.hyperparametersDict[problemType] = hyperparameters
            self.evaluateResultDict[problemType] = evaluateResult

    def get_order_by_problemType(
        self, graph: nx.Graph, problemType: ProblemType
    ) -> List:
        from data_processing import DatasetLoader  # NOTE: 放在这避免循环导入

        # 初始化data与state
        originData, node_mapping = DatasetLoader.load_single_graph(
            graph
        )  # node_mapping 用于最后的节点转化
        state = State([1 for _ in range(len(graph.nodes))])

        # 调用合适的模型
        model: NiceModel = self.modelDict[problemType]

        # 记录最佳断连顺序
        mapped_order = []
        model.eval()
        with torch.inference_mode():
            stateData = originData
            with torch.autocast(
                device_type=str(next(iter(model.parameters())).device),
                enabled=True,  # NOTE: 支持混合精度计算(可以在保持准确性的同时提高性能)
            ):
                for _ in tqdm(
                    range(graph.number_of_nodes()),
                    desc=f"{self.NAME}",
                    leave=False,
                ):
                    stateData = get_stateData(state, stateData)
                    values: Tensor = model(
                        Batch.from_data_list([stateData]), task=Task.VALUE
                    )
                    action: Action = state.get_best_action(values.squeeze())
                    state.exec_action(action)
                    mapped_order.append(action.targetNodeIndex)

        # 还原正确的节点标签
        reversed_node_mapping = {value: key for key, value in node_mapping.items()}
        real_order = [reversed_node_mapping[item] for item in mapped_order]
        return real_order


if __name__ == "__main__":

    @timeit
    def test_get_order(algorithm: _Algorithm, graph: nx.Graph):
        colored_print(f"当前算法: {algorithm.name}({algorithm.__class__.__name__})")
        orders = [
            algorithm.get_order_by_problemType(graph, problemType)
            for problemType in list(ProblemType)
        ]
        print(orders)
        for ind in range(len(ProblemType)):
            order = orders[ind]
            problemType = list(ProblemType)[ind]
            assert len(order) == len(
                graph.nodes
            ), f"{problemType.name}排序结果尺寸错误: {len(order)} != {len(graph.nodes)}"

    parser = argparse.ArgumentParser(description="Algorithm Test")
    parser.add_argument(
        "--node_num", required=False, type=int, default=7, help="图节点数"
    )
    args = parser.parse_args()
    nodeNum = args.node_num
    colored_print(f"node_num: {nodeNum}")

    # 创建测试图
    graph = nx.connected_watts_strogatz_graph(n=nodeNum, k=4, p=0.1)

    # BruteForceAlgorithm
    alogrithm_bruteforce = BruteForceAlgorithm()
    test_get_order(alogrithm_bruteforce, graph)

    # GreedyAlgorithm
    alogrithm_greedy1 = GreedyAlgorithm()
    test_get_order(alogrithm_greedy1, graph)

    # GreedyAlgorithmV2
    alogrithm_greedy2 = GreedyAlgorithmV2(2)
    test_get_order(alogrithm_greedy2, graph)

    # PureRandomAlgorithm
    algorithm_random = PureRandomAlgorithm()
    test_get_order(algorithm_random, graph)

    # NaiveAlgorithm
    algorithm_naive = NaiveAlgorithm()
    test_get_order(algorithm_naive, graph)

    # PageRankAlgorithm
    algorithm_pagerank = PageRankAlgorithm(0.85)
    test_get_order(algorithm_pagerank, graph)

    # LeaderRankAlgorithm
    algorithm_leaderrank = LeaderRankAlgorithm()
    test_get_order(algorithm_leaderrank, graph)

    # HITSAlgorithm
    algorithm_hits = HITSAlgorithm()
    test_get_order(algorithm_hits, graph)

    # CIAlgorithm
    algorithm_ci = CIAlgorithm()
    test_get_order(algorithm_ci, graph)

    # HDAAlgorithm
    algorithm_hda = HDAAlgorithm()
    test_get_order(algorithm_hda, graph)

    # BetweennessCentralityAlgorithm
    algorithm_bc = BetweennessCentralityAlgorithm()
    test_get_order(algorithm_bc, graph)

    # EigenvectorCentralityAlgorithm
    algorithm_ec = EigenvectorCentralityAlgorithm()
    test_get_order(algorithm_ec, graph)

    # ClosenessCentralityAlgorithm
    algorithm_cc = ClosenessCentralityAlgorithm()
    test_get_order(algorithm_cc, graph)

    # ProposedAlgorithm
    modelFilenameDict = {
        ProblemType.CN: "NiceModel_CN_test.pth",
        ProblemType.ND: "NiceModel_ND_test.pth",
    }
    alogrithm_proposed = ProposedAlgorithm(modelFilenameDict)
    test_get_order(alogrithm_proposed, graph)
