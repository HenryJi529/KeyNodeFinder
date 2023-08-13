import random
from typing import Union, List, Tuple, Dict
from enum import Enum
from pathlib import Path

import numpy as np
from pandas import DataFrame
import networkx as nx
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
import powerlaw


try:
    from utils.common import colored_print
    from utils.experiment import DEVICE
except:
    from common import colored_print
    from experiment import DEVICE

DATA_PATH = Path(__file__).parent.parent / "data"
SYNTHETIC_DATA_PATH = DATA_PATH / "synthetic"
REAL_DATA_PATH = DATA_PATH / "real"

DEFAULT_NODE_NUM_RANGES = [
    (30, 50),
    (50, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
]


class FeatureBuilder:
    FEATURE_NUM = 3

    @staticmethod
    def getFeatureArray(graph: nx.Graph):
        def get_sorted_values(dict1: Dict):
            return [
                value for _, value in sorted(dict1.items(), key=lambda item: item[0])
            ]

        hits = get_sorted_values(nx.hits(graph, max_iter=100, normalized=True)[0])
        degreeCentrality = get_sorted_values(nx.centrality.degree_centrality(graph))
        coreNumber = get_sorted_values(nx.core_number(graph))
        dataDict = {"DC": degreeCentrality, "HITS": hits, "K-Core": coreNumber}
        df = DataFrame(data=dataDict)
        return torch.tensor(df.values).contiguous()


class DataBatchDecomposer:
    @staticmethod
    def get_batchinfo_from_databatch(databatch: Batch) -> Tuple[int, List]:
        """获取data数与每个data中的节点数"""
        dataNum = (torch.max(databatch.batch) + 1).item()
        nodeNumList = [
            len(torch.where(databatch.batch == ind)[0]) for ind in range(dataNum)
        ]
        return dataNum, nodeNumList

    @staticmethod
    def get_featureArrayList_from_databatch(databatch: Batch):
        """获取每个data对应的featureArray构成的列表"""
        dataNum, _ = DataBatchDecomposer.get_batchinfo_from_databatch(databatch)
        featureArrayList = []
        for ind in range(dataNum):
            nodeIndex = torch.where(databatch.batch == ind)[0]
            featureArray = databatch.x[nodeIndex, :]
            featureArrayList.append(featureArray)
        return featureArrayList

    @staticmethod
    def get_logitsList_from_databatch(concatenated_logits: Tensor, databatch: Batch):
        dataNum, nodeNumList = DataBatchDecomposer.get_batchinfo_from_databatch(
            databatch
        )
        logitsList = []
        for ind in range(dataNum):
            if ind == 0:
                start = 0
            else:
                start = int(nodeNumList[ind - 1] * (nodeNumList[ind - 1] - 1) / 2)
            end = start + int(nodeNumList[ind] * (nodeNumList[ind] - 1) / 2) - 1
            logits = concatenated_logits[start : end + 1]
            logitsList.append(logits)

        return logitsList

    @staticmethod
    def get_differencesList_from_databatch(databatch: Batch, verbose=False):
        dataNum, nodeNumList = DataBatchDecomposer.get_batchinfo_from_databatch(
            databatch
        )
        if verbose:
            print(f"nodeNumList: {nodeNumList}")
        differencesList = []
        for ind in range(dataNum):
            nodeIndex = torch.where(databatch.batch == ind)[0]
            rank = databatch.y[nodeIndex]

            differences = RankingProcessor.generate_differences_from_rank(rank)
            differencesList.append(differences)
            if verbose:
                print(f"differences.shape: {differences.shape}")
                print(f"differences: {differences[:10]}")

        return differencesList


class GraphTool:
    @staticmethod
    def get_random_averageDegree(nodeNum: int):
        """获取偶数的averageDregee"""
        # 设置powerlaw分布的参数
        alpha = 3  # Powerlaw指数参数
        xmin = 1.8  # 最小截尾值

        # 创建Power_Law对象
        pl = powerlaw.Power_Law(xmin=xmin, parameters=[alpha])

        # 生成随机数
        while True:
            randomData = int(np.ceil(pl.generate_random(1)[0]))
            averageDegree = randomData if randomData % 2 == 0 else randomData + 1
            if averageDegree < nodeNum / 3:
                break
        return averageDegree

    @staticmethod
    def trust_rank(
        graph: nx.Graph,
        alpha: float = 0.85,
        seeds: list = None,
        max_iter: int = 100,
        tol: float = 1e-06,
        sorted: bool = False,
        verbose: bool = False,
    ):
        """针对无向图无自环的信任排名算法
        Args:
            graph (nx.Graph): networkx图对象
            alpha (float): 衰减因子
            max_iter (int): 最大迭代次数
            tol (float): 收敛阈值
        Returns:
            dict[nodeId, rankValue]: 所有节点的排名值
        """
        adjacencyMatrix = nx.adjacency_matrix(graph).todense()
        inverseDegreeMatrix = np.diag(1 / adjacencyMatrix.sum(axis=0))
        normalizedAdjacencyMatrix = adjacencyMatrix @ inverseDegreeMatrix

        if not seeds:
            v = np.array(list(nx.pagerank(graph).values())).reshape(-1, 1)
        else:
            nodeList = list(graph.nodes)
            v = np.zeros(graph.number_of_nodes())
            seeds = list(
                set(graph.nodes).intersection(set(seeds))
            )  # NOTE: 排除seeds参数中的未知节点
            v[[nodeList.index(seed) for seed in seeds]] = 1 / len(seeds)
            v = v.reshape(-1, 1)

        r = (np.ones(graph.number_of_nodes()) / graph.number_of_nodes()).reshape(-1, 1)
        for i in range(max_iter):
            # 计算下一轮迭代时节点的权重值
            r_next = alpha * normalizedAdjacencyMatrix @ r + (1 - alpha) * v
            # 判断是否已经收敛
            if np.abs(r_next - r).max() < tol:
                break
            # 没有收敛时，继续迭代
            r = r_next
            # 输出每一轮迭代后的权重值
            if verbose:
                print("Iteration {}: {}".format(i + 1, np.squeeze(r_next)))

        resultDict = {
            nodeId: rankValue for nodeId, rankValue in zip(graph.nodes, np.squeeze(r))
        }
        if sorted:
            resultDict = dict(
                sorted(resultDict.items(), key=lambda item: item[0], reverse=True)
            )
        return resultDict

    @staticmethod
    def leader_rank(
        graph: nx.Graph,
        tol: float = 1e-06,
    ):
        """LeaderRank
        算法步骤:
        1. 在网络中增加一个节点g(GroundNode)，将其与网络中的所有节点相连
        2. 给除了节点g之外的所有节点分配1单位的LR值
        3. 聚合节点信息: $s_i(t+1)=\sum_{j=0}^N \frac{a_{i j}}{k_j} s_j(t)$
        4. 将g的LR值平分给其他所有N个节点
        """
        nodeNum = graph.number_of_nodes()
        originNodes = list(graph.nodes)

        # 添加LeaderNode
        G = graph.copy()
        G.add_node(-1)
        for node in originNodes:
            G.add_edge(-1, node)
        degree = np.array(list(dict(G.degree).values()))
        adjacencyMatrix = nx.adjacency_matrix(G).todense()

        # LR初始化
        LR = np.ones(nodeNum + 1) / nodeNum
        LR[-1] = 0.0  # NOTE: 最后一个值刚好代表LeaderNode

        # # LR迭代更新
        while True:
            tempLR = LR
            LR = adjacencyMatrix @ (LR / degree)
            LR = LR / np.sum(LR)
            if np.sqrt(np.sum((tempLR - LR) * (tempLR - LR))) <= tol:
                break

        # 平分LeaderNode的LR值
        LR = LR + LR[-1] / nodeNum
        LR = LR[:-1]

        return {node: value for node, value in zip(originNodes, LR)}

    @staticmethod
    def sample_graph(G: nx.Graph, ratio: float = 0.1) -> nx.Graph:
        sampleNum = int(G.number_of_nodes() * ratio)

        nodes = list(G.nodes)
        sampled_nodes = random.sample(nodes, sampleNum)

        subgraph = G.subgraph(sampled_nodes)

        return subgraph

    @staticmethod
    def summarize_graph(graph: nx.Graph):
        resultDict = {}
        lcc = graph.subgraph(max(nx.connected_components(graph), key=len))
        resultDict["$|V|$"] = graph.number_of_nodes()
        resultDict["$|E|$"] = graph.number_of_edges()
        resultDict["$<k>$"] = np.mean([d for _, d in graph.degree()])
        resultDict["$<d>$"] = nx.average_shortest_path_length(lcc)
        resultDict["$diameter$"] = nx.diameter(lcc)
        resultDict["$ln|V|/ln<k>$"] = np.log(resultDict["$|V|$"]) / np.log(
            resultDict["$<k>$"]
        )
        return resultDict

    @staticmethod
    def read_graph(
        path: Path, delimiter: str = None, weighted: bool = None
    ) -> nx.Graph:
        def count_comma(path: Path):
            cnt = 0
            with open(path, "r") as file:
                for line in file:
                    cnt += line.count(",")
            return cnt

        def count_tab(path: Path):
            cnt = 0
            with open(path, "r") as file:
                for line in file:
                    cnt += line.count("\t")
            return cnt

        def isWeighted(path: Path, delimiter: str):
            with open(path, "r") as file:
                for line in file:
                    if len(line.split(delimiter)) == 2:
                        return False
            return True

        # 根据文件自动确定delimiter和weighted
        if delimiter is None:
            if count_comma(path) > 1:
                delimiter = ","
            elif count_tab(path) > 1:
                delimiter = "\t"
            else:
                delimiter = " "
        weighted = isWeighted(path, delimiter) if weighted is None else weighted

        if weighted:
            graph = nx.read_weighted_edgelist(path, delimiter=delimiter)
            return graph
            # df = pd.read_csv(path, sep=delimiter, names=["source", "target", "weight"])
            # return nx.from_pandas_edgelist(df, "source", "target", "weight") # alternative
        else:
            graph = nx.read_edgelist(path, delimiter=delimiter)
            return graph

    @staticmethod
    def disconnect_node(
        graph: nx.Graph, node: int, copied=False
    ) -> Union[None, nx.Graph]:
        """断开某节点的连边"""
        if copied:
            G = graph.copy()
        else:
            G = graph

        G.remove_edges_from(list(G.edges(node)))

        if copied:
            return G

    @staticmethod
    def get_edgeIndex_from_graph(
        graph: nx.Graph,
        return_node_mapping: bool = False,
        device: torch.device = DEVICE,
    ):
        # 获取图中的节点列表
        nodes = list(graph.nodes)

        # 创建映射字典，将原始节点映射到连续的节点索引
        node_mapping = {node: i for i, node in enumerate(nodes)}

        # 根据映射更新边列表
        edges = [
            (node_mapping[edge[0]], node_mapping[edge[1]]) for edge in graph.edges()
        ]

        # 转换为torch的edge_index
        edge_index = torch.tensor(edges).to(device).t().contiguous()

        if return_node_mapping:
            return edge_index, node_mapping
        else:
            return edge_index

    @staticmethod
    def get_normalizedAdjacencyMatrix_from_edgeIndex(
        edgeIndex: Tensor, maxNodeNum: int
    ):
        A = to_dense_adj(
            torch.cat([edgeIndex, edgeIndex.flip(0)], dim=1), max_num_nodes=maxNodeNum
        ).squeeze()  # NOTE: to_dense_adj是把输入的边索引当成有向的来处理
        simA = A + torch.eye(A.shape[1]).to(A.device)
        degree = torch.sum(simA, dim=0)
        return torch.diag(torch.rsqrt(degree)) @ simA @ torch.diag(torch.rsqrt(degree))

    @staticmethod
    def view_edges_from_edgeIndex(edgeIndex: Tensor):
        return DataFrame(edgeIndex.T, columns=["source", "target"])

    @staticmethod
    def test():
        def get_test_graph():
            graph = nx.Graph()
            graph.add_edges_from(
                [
                    (1, 1),  # NOTE: 可以考虑自环
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (5, 6),
                    (1, 4),
                ]
            )  # 生成的图结构为[1, 2, 3, 4], [5, 6]
            return graph

        graph = get_test_graph()
        GraphTool.disconnect_node(graph, 1)
        GraphTool.disconnect_node(graph, 6)
        edgeIndex, node_mapping = GraphTool.get_edgeIndex_from_graph(
            graph, return_node_mapping=True
        )
        colored_print("打印edgeIndex和对应node_mapping: ")
        print(edgeIndex)
        print(node_mapping)
        normalizedAdjacencyMatrix = (
            GraphTool.get_normalizedAdjacencyMatrix_from_edgeIndex(
                edgeIndex, maxNodeNum=len(graph.nodes)
            )
        )
        colored_print("打印归一化的邻接矩阵: ")
        print(normalizedAdjacencyMatrix)
        colored_print("打印leader_rank结果: ")
        print(GraphTool.leader_rank(graph))
        colored_print("打印BA无标度网络的结构信息: ")
        print(GraphTool.summarize_graph(nx.barabasi_albert_graph(1000, 6)))


class RankingProcessor:
    @staticmethod
    def get_rank_from_order(order: Tensor):
        """
        根据节点的order，生成从标号最小到最大的节点的排名
        """
        # NOTE: 根据torch.argsort(torch.argsort(torch.argsort(a))) == torch.argsort(a)，简化步骤
        # NOTE: 如果原本节点就是连续的(实际上edge_index也是这么要求的)，就没必要紧密
        # compacted_order = (
        #     torch.argsort(torch.argsort(order))
        # )  # # 两次argsort，将原本的排序转化为从0开始的连续整数节点排序
        # rank = torch.argsort(compacted_order)
        rank = torch.argsort(order)
        return rank

    @staticmethod
    def generate_differences_from_rank(rank: Tensor, normalize=True):
        """将一维度的rank转换为一维度的differences"""
        length = rank.shape[0]
        differenceNum = int(length * (length - 1) / 2)
        differences = torch.zeros(differenceNum).to(rank.device)
        index = 0
        for i in range(length - 1):
            for j in range(i + 1, length):
                differences[index] = rank[i] - rank[j]
                index += 1
        # NOTE: 是否转换为01格式(0: lt; 1:ge[实际上只会出现gt, gt表示前一个节点优于后一个节点])
        if normalize:
            differences = torch.where(differences > 0, torch.tensor(0), torch.tensor(1))

        return differences


class RankLossFunction:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.lossFn = BCEWithLogitsLoss()

    def __call__(self, concatenated_logits: Tensor, databatch: Batch):
        differencesList = DataBatchDecomposer.get_differencesList_from_databatch(
            databatch, verbose=self.verbose
        )
        concatenated_differences = torch.cat(differencesList)

        if self.verbose:
            print(f"concatenated_logits.shape: {concatenated_logits.shape}")
            print(f"concatenated_differences.shape: {concatenated_differences.shape}")

        # NOTE: 这里逻辑上需要分离，但出于梯度计算的方便，暂不分离
        loss = self.calc_loss(concatenated_differences, concatenated_logits.squeeze())

        if self.verbose:
            print("=" * 40)
        return loss

    def calc_loss(self, differences: Tensor, logits: Tensor) -> Tensor:
        differences = differences.to(torch.float32)
        loss = self.lossFn(logits, differences)
        return loss


class ProblemType(Enum):
    CN = 1
    ND = 2


class ProblemMetric:
    @classmethod
    def get_gcc_size(cls, graph: nx.Graph) -> int:
        return len(max(nx.connected_components(graph), key=len))

    @classmethod
    def get_pairwise_connectivity(cls, graph: nx.Graph) -> float:
        return sum(
            [
                len(subgraph) * (len(subgraph) - 1) / 2
                for subgraph in list(nx.connected_components(graph))
            ]
        )

    @classmethod
    def get(cls, problemType: ProblemType):
        if problemType == ProblemType.CN:
            # NOTE: 依据pairwise connectivity
            metricFn = cls.get_pairwise_connectivity
        elif problemType == ProblemType.ND:
            # NOTE: 依据最大连通子图的节点数
            metricFn = cls.get_gcc_size
        else:
            raise ValueError("ProblemType Error")
        return metricFn


if __name__ == "__main__":
    GraphTool.test()
