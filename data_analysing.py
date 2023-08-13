"""
图数据分析
"""

import copy
from collections import Counter
from typing import Dict, List, Union
from functools import cached_property, cache
from itertools import combinations
import math
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

from utils.toolbox import GraphTool

IMAGE_PATH = Path(__file__).parent / Path("images")


class GraphInfo:
    """
    处理带节点属性有权无向图(无边属性)的最大连通子图【一般不考虑其边的权重】
    """

    @staticmethod
    def test():
        G = nx.barabasi_albert_graph(10, 3)
        info = GraphInfo(G)
        assert all(
            [
                math.isclose(num1, num2)
                for num1, num2 in zip(
                    info.clusteringCoefficient.values(),
                    info.clusteringCoefficientSubset(info.nodes).values(),
                )
            ]
        ), "clusteringCoefficient failed..."
        print("DONE!")

    def __init__(self, originalGraph: nx.Graph):
        self.originalGraph = originalGraph
        # 提取原图中的最大连接子图
        self.graph: nx.Graph = originalGraph.subgraph(
            max(nx.connected_components(originalGraph), key=len)
        )

    @cached_property
    def isConnected(self):
        return nx.is_connected(self.originalGraph)

    @cached_property
    def numberOfNodes(self):
        return self.graph.number_of_nodes()

    @cached_property
    def numberOfEdges(self):
        return self.graph.number_of_edges()

    @cached_property
    def numberOfConnectedComponents(self):
        return nx.number_connected_components(self.originalGraph)

    def plot_graph(
        self,
        iterationsNum: int = 10,
        path: Union[str, Path] = IMAGE_PATH / "graph.pdf",
        save=False,
        withEdge=True,
        method: str = "DegreeCentralityMethod",
    ):
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis("off")

        if method == "DegreeCentralityMethod":
            # NOTE: 由于最大连通子图中的节点度>1，因为取对数只要+1就可以
            nodeSize = [np.log10(self.degree[i] + 1) * 10 for i in self.nodes]
            nodeColor = [self.degree[i] for i in self.nodes]
        elif method == "ClosenessCentralityMethod":
            nodeSize = [v * 50 for v in self.closenessCentrality.values()]
            nodeColor = [v for v in self.closenessCentrality.values()]
        elif method == "BetweennessCentralityMethod":
            nodeSize = [v * 50 for v in self.betweennessCentrality.values()]
            nodeColor = [v for v in self.betweennessCentrality.values()]
        elif method == "EigenvectorCentralityMethod":
            nodeSize = [
                np.sqrt(np.sqrt(v)) * 100 for v in self.eigenvectorCentrality.values()
            ]
            nodeColor = [v for v in self.eigenvectorCentrality.values()]
        else:
            pass

        options = {
            "pos": nx.spring_layout(
                self.unweightedGraph, iterations=iterationsNum, seed=0
            ),
            "node_size": nodeSize,
            "node_color": nodeColor,
            "cmap": plt.cm.cool,  # 设置节点colormap
            "edge_color": "gray",
            "with_labels": False,
            "width": 0.15,
        }
        if withEdge:
            options["edge_color"] = "gray"
            options["with_labels"] = False
            options["width"] = 0.15
            nx.draw(self.unweightedGraph, ax=ax, **options)
        else:
            nx.draw_networkx_nodes(self.unweightedGraph, ax=ax, **options)

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def unweightedGraph(self):
        U = nx.Graph()
        U.add_edges_from(self.graph.edges())  # NOTE: 会自动忽略边属性
        return U

    @cached_property
    def adjacencyMatrix(self):
        return (
            nx.adjacency_matrix(self.unweightedGraph, dtype=np.bool_)
            .todense()
            .astype(np.bool_)
        )

    @cached_property
    def degree(self):
        return dict(self.unweightedGraph.degree)

    @cached_property
    def nodes(self):
        return list(self.unweightedGraph.nodes)

    @cached_property
    def maxDegree(self):
        return max(self.degree.values())

    @cached_property
    def averageDegree(self):
        return np.mean([d for _, d in self.degree])

    @cached_property
    def degreeDistribution(self):
        return nx.degree_histogram(self.unweightedGraph)

    def plot_degreeDistribution(
        self,
        path: Union[str, Path] = IMAGE_PATH / "degreeDistribution.pdf",
        save=False,
        bar=False,
    ):
        fig, ax = plt.subplots(figsize=(20, 12))
        x = list(range(self.maxDegree + 1))
        y = [i / self.numberOfNodes for i in self.degreeDistribution]

        if bar:
            plt.bar(x, y)
        else:
            # NOTE: 去除零点
            xCopy = copy.deepcopy(x)
            yCopy = copy.deepcopy(y)
            for ind in range(self.maxDegree + 1):
                if self.degreeDistribution[ind] == 0:
                    # NOTE: 度为0的节点在连通图中不存在
                    if ind == 0:
                        pass
                    else:
                        xCopy[ind] = xCopy[ind - 1]
                        yCopy[ind] = yCopy[ind - 1]
            x = xCopy
            y = yCopy

            ax.plot(x, y, "ro-")
            ax.set_xscale("log")
            ax.set_yscale("log")
            xticks = [int(np.power(2, i)) for i in np.arange(np.log2(self.maxDegree))]
            yticks = [
                round(1 / 2 ** (-i), 4)
                for i in np.linspace(
                    np.log2(np.min(np.array(y)[np.array(y) > 0])),
                    np.log2(np.max(y)),
                    10,
                )
            ]
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([str(ind) for ind in xticks])
            ax.set_yticklabels([str(ind) for ind in yticks])

        ax.set_title(
            "度分布",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        ax.set_xlabel("$k$")
        ax.set_ylabel("$P(k)$")

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def shortestPathLengths(self):
        return dict(nx.all_pairs_shortest_path_length(self.unweightedGraph))

    def shortest_path_length(self, node1, node2):
        return nx.shortest_path_length(self.unweightedGraph, node1, node2)

    def shortest_path(self, node1, node2):
        return nx.shortest_path(self.unweightedGraph, source=node1, target=node2)

    def all_shortest_paths(self, node1, node2):
        return list(
            nx.all_shortest_paths(self.unweightedGraph, source=node1, target=node2)
        )

    @cached_property
    def diameter(self) -> int:
        return int(
            max(
                nx.eccentricity(
                    self.unweightedGraph, sp=self.shortestPathLengths
                ).values()
            )
        )
        # return nx.diameter(G) NOTE: 效率低

    @cached_property
    def averageShortestPathLengths(self):
        # NOTE: 这里考虑了到自身的距离，并不合理
        averageShortestPathLengths = {
            node: np.mean(list(spl.values()))
            for node, spl in self.shortestPathLengths.items()
        }
        return averageShortestPathLengths

    @cached_property
    def averageDistance(self):
        # NOTE: 相对图来说的平均距离
        return np.mean(self.averageShortestPathLengths.values())

    @cached_property
    def shortestPathLengthDistribution(self):
        distances = []
        for node1, spl in self.shortestPathLengths.items():
            for node2, distance in spl.items():
                if node1 != node2:
                    distances.append(distance)

        counter = Counter(distances)

        distance_count = np.zeros(self.diameter + 1)
        for key, value in counter.items():
            distance_count[key] = 100.0 * value / sum(counter.values())

        return distance_count

    def plot_shortestPathLengthDistribution(
        self,
        path: Union[str, Path] = IMAGE_PATH / "shortestPathLengthDistribution.pdf",
        save=False,
    ):
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(
            np.arange(self.diameter + 1, dtype=np.int64),
            height=self.shortestPathLengthDistribution,
        )
        ax.set_title(
            "最短路径长度分布",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        ax.set_xticks(np.arange(self.diameter + 1, dtype=np.int64))
        ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
        ax.set_ylabel("Frequency (%)", fontdict={"size": 22})

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    def efficiency(self, node1, node2):
        return nx.efficiency(self.unweightedGraph, node1, node2)

    @cached_property
    def localEfficiency(self):
        return nx.local_efficiency(self.unweightedGraph)

    @cached_property
    def globalEfficiency(self):
        return nx.global_efficiency(self.unweightedGraph)

    @cached_property
    def density(self):
        return nx.density(self.unweightedGraph)

    @cached_property
    def clusteringCoefficient(self) -> Dict:
        return nx.clustering(self.unweightedGraph)

    def clusteringCoefficientSubset(self, nodeList) -> Dict:
        clusteringCoefficient = {}
        for node in nodeList:
            neighborNodeList = self.all_neighbors(node)
            if len(neighborNodeList) == 1:
                clusteringCoefficient[node] = 0
            else:
                indexList = [self.nodes.index(node) for node in neighborNodeList]
                coef = self.adjacencyMatrix[indexList][:, indexList].sum()
                clusteringCoefficient[node] = coef / (
                    len(neighborNodeList) * (len(neighborNodeList) - 1)
                )

        return clusteringCoefficient

    @cached_property
    def averageClusteringCoefficient(self):
        return np.mean(list(self.clusteringCoefficient.values()))
        # return nx.average_clustering(self.unweightedGraph)

    @cached_property
    def globalClusteringCoefficient(self):
        return nx.transitivity(self.unweightedGraph)

    def plot_clusteringCoefficientHistogram(
        self,
        path: Union[str, Path] = IMAGE_PATH / "clusteringCoefficientHistogram.pdf",
        save=False,
    ):
        plt.figure(figsize=(15, 8))
        plt.hist(self.clusteringCoefficient.values(), bins=20)
        plt.title(
            "集聚系数直方图",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
        plt.ylabel("Counts", fontdict={"size": 20})

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def degreeCentrality(self) -> Dict:
        return nx.centrality.degree_centrality(self.unweightedGraph)

    def plot_degreeCentralityHistogram(
        self,
        path: Union[str, Path] = IMAGE_PATH / "degreeCentralityHistogram.pdf",
        save=False,
    ):
        plt.figure(figsize=(15, 8))
        plt.hist(self.degreeCentrality.values(), bins=25)
        plt.title("度中心性直方图", fontdict={"size": 35}, loc="center")
        plt.xlabel("Degree Centrality", fontdict={"size": 20})
        plt.ylabel("Counts", fontdict={"size": 20})

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def closenessCentrality(self):
        # NOTE: 也可以用节点到其他各点平均距离的倒数来计算
        return nx.centrality.closeness_centrality(self.unweightedGraph)

    def closenessCentralitySubset(self, nodeList: List):
        resultDict = {
            node: (self.numberOfNodes - 1)
            / sum(self.shortestPathLengths[node].values())
            for node in nodeList
        }
        return resultDict

    def plot_closenessCentralityHistogram(
        self,
        path: Union[str, Path] = IMAGE_PATH / "closenessCentralityHistogram.pdf",
        save=False,
    ):
        plt.figure(figsize=(15, 8))
        plt.hist(self.closenessCentrality.values(), bins=20)
        plt.title(
            "接近度中心性直方图",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        plt.xlabel("Closeness Centrality", fontdict={"size": 20})
        plt.ylabel("Counts", fontdict={"size": 20})

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def betweennessCentrality(self):
        return nx.centrality.betweenness_centrality(self.unweightedGraph)

    # NOTE: 理论上正确，但计算效率实在太低
    def betweennessCentralitySubsetBeta(self, nodeList: List):
        resultDict = {node: 0 for node in nodeList}
        allShortestPaths = {}
        # 计算介数
        node_combinations = combinations(self.nodes, 2)
        for node in nodeList:
            for node1, node2 in node_combinations:
                if node1 == node or node2 == node:
                    continue
                if f"{node1}-{node2}" not in allShortestPaths:
                    allShortestPaths[f"{node1}-{node2}"] = self.all_shortest_paths(
                        node1, node2
                    )
                currentShortestPaths = allShortestPaths[f"{node1}-{node2}"]
                resultDict[node] += sum(
                    1 for path in currentShortestPaths if node in path
                ) / len(currentShortestPaths)
        # 计算介数中心性
        resultDict = {
            node: 2 * value / self.numberOfNodes / (self.numberOfNodes - 1)
            for node, value in resultDict.items()
        }
        return resultDict

    @cached_property
    def edgeBetweennessCentrality(self):
        return nx.edge_betweenness_centrality(self.unweightedGraph)

    def plot_betweennessCentralityHistogram(
        self,
        path: Union[str, Path] = IMAGE_PATH / "betweennessCentralityHistogram.pdf",
        save=False,
    ):
        plt.figure(figsize=(15, 8))
        plt.hist(self.betweennessCentrality.values(), bins=20)
        plt.title(
            "介数中心性直方图",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        plt.xlabel("Betweenness Centrality", fontdict={"size": 20})
        plt.ylabel("Counts", fontdict={"size": 20})

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def eigenvectorCentrality(self):
        # NOTE: 根据特征向量中心性的计算方法，易得中心性大的节点总是与中心性大的节点相连
        return nx.centrality.eigenvector_centrality(self.unweightedGraph)

    def plot_eigenvectorCentralityHistogram(
        self,
        path: Union[str, Path] = IMAGE_PATH / "eigenvectorCentralityHistogram.pdf",
        save=False,
    ):
        plt.figure(figsize=(15, 8))
        plt.hist(self.eigenvectorCentrality.values(), bins=20)
        plt.title(
            "特征向量中心性直方图",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        plt.xlabel("Eigenvector Centrality", fontdict={"size": 20})
        plt.ylabel("Counts", fontdict={"size": 20})

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def degreeAssortativityCoefficient(self):
        return nx.degree_assortativity_coefficient(self.unweightedGraph)

    @cached_property
    def degreePearsonCorrelationCoefficient(self):
        """基于Pearson相关系数的度-度相关性"""
        return nx.degree_pearson_correlation_coefficient(self.unweightedGraph)

    @cache
    def all_neighbors(self, node: str):
        return list(nx.all_neighbors(self.unweightedGraph, node))

    @cached_property
    def averageNearestNeighborDegreeWithMatrix(self):
        """基于最近邻平均度值的度-度相关性(矩阵方法)"""
        A = nx.to_numpy_array(
            self.unweightedGraph, dtype=np.int64
        )  # A = nx.adjacency_matrix(self.unweightedGraph).todense()
        k_array = np.array([self.degree[node] for node in self.nodes])
        # k_array = A.sum(axis=1) # NOTE: 只有对角线无元素时，才是等价方法
        sorted_k = sorted(set(k_array))  # 获取所有可能的度值

        k_nn_i = A @ k_array / k_array

        isK = np.zeros((self.numberOfNodes, len(sorted_k)))
        for ind in range(len(sorted_k)):
            x_index = k_array == sorted_k[ind]
            isK[x_index, ind] = 1

        Knn = k_nn_i @ isK / np.array([self.degreeDistribution[k] for k in sorted_k])
        return sorted_k, Knn

    @cached_property
    def averageNearestNeighborDegree(self):
        """基于最近邻平均度值的度-度相关性"""
        k = set([self.degree[i] for i in self.nodes])  # 获取所有可能的度值
        sorted_k = sorted(k)

        k_nn_k = []
        for ki in sorted_k:
            if ki == 0:
                k_nn_k.append(0.0)
            else:
                c = 0
                k_nn_i = 0
                for i in self.nodes:
                    if self.degree[i] == ki:
                        k_nn_i += (
                            sum([self.degree[j] for j in self.all_neighbors(i)]) / ki
                        )
                        c += 1
                k_nn_k.append(k_nn_i / c)
        return sorted_k, k_nn_k

    def plot_averageNearestNeighborDegree(
        self, path: Union[str, Path] = IMAGE_PATH / "degreeCorrelation.pdf", save=False
    ):
        fig, ax = plt.subplots(figsize=(20, 12))

        sorted_k, Knn = self.averageNearestNeighborDegreeWithMatrix

        ax.plot(sorted_k, Knn, color="red")
        plt.title(
            "基于最近邻平均度值的度-度相关性",
            fontdict={
                "fontname": "Songti SC",
                "color": "darkred",
                "weight": "bold",
                "size": 30,
            },
            loc="center",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def hasBridges(self):
        return nx.has_bridges(self.unweightedGraph)

    @cached_property
    def bridges(self):
        return list(nx.bridges(self.unweightedGraph))

    @cached_property
    def localBridges(self):
        """
        The edges that are local bridges are saved in a list and their number is printed.
        In detaill, an edge joining two nodes $C$ and $D$ in a graph is a local bridge, if its endpoints $C$ and $D$ have no friends in common.
        """
        return list(
            nx.local_bridges(self.unweightedGraph, with_span=False)
        )  # NOTE: span就是被桥连接的节点的距离

    def plot_bridges(
        self,
        iterationsNum: int = 10,
        path: Union[str, Path] = IMAGE_PATH / "bridges.pdf",
        save=False,
    ):
        fig, ax = plt.subplots(figsize=(15, 8))
        options = {
            "pos": nx.spring_layout(
                self.unweightedGraph, iterations=iterationsNum, seed=0
            ),
            "node_size": 1,
            "width": 0.5,
        }
        nx.draw_networkx(
            self.unweightedGraph,
            ax=ax,
            edge_color="gray",
            node_color="blue",
            with_labels=False,
            **options,
        )
        nx.draw_networkx_edges(
            self.unweightedGraph,
            ax=ax,
            edgelist=self.localBridges,
            edge_color="green",
            **options,
        )  # green color for local bridges
        nx.draw_networkx_edges(
            self.unweightedGraph,
            ax=ax,
            edgelist=self.bridges,
            edge_color="r",
            **options,
        )  # red color for bridges
        plt.axis("off")

        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @cached_property
    def graphlet3(self):
        return nx.triangles(self.unweightedGraph)

    @cached_property
    def sumOfGraphlet3(self):
        return sum(list(self.graphlet3.values())) / 3

    @cached_property
    def averageOfGraphlet3(self):
        return np.mean(list(self.graphlet3.values()))

    @cached_property
    def medianOfGraphlet3(self):
        return np.median(list(self.graphlet3.values()))

    @cached_property
    def coreNumber(self):
        return nx.core_number(self.unweightedGraph)

    @cache
    def page_rank(self, alpha: float = 0.85):
        return nx.pagerank(self.unweightedGraph, alpha=alpha)

    @cached_property
    def spectrum(self):
        # laplacian_spectrum
        # adjacency_spectrum
        return nx.linalg.spectrum.normalized_laplacian_spectrum(self.unweightedGraph)

    @cached_property
    def hits(self):
        authority, hub = nx.hits(self.unweightedGraph, max_iter=100, normalized=True)
        return authority

    @cache
    def trust_rank(self, alpha: float = 0.85):
        return GraphTool.trust_rank(self.unweightedGraph, alpha=alpha)

    def identify_important_nodes(self, ratio: float = 0.05):
        def standardization(rankDict):
            meanValue = np.mean(list(rankDict.values()))
            stdValue = np.std(list(rankDict.values()))
            return {k: (v - meanValue) / stdValue for k, v in rankDict.items()}

        def get_top_x_percent_keys(dic: Dict, percent: float = 0.01) -> List:
            # 按照字典的值降序排序
            sorted_items = sorted(dic.items(), key=lambda x: x[1], reverse=True)
            # 获取前{percent}的key
            top_x_percent = int(len(sorted_items) * percent)
            top_items = sorted_items[:top_x_percent]
            keys = [item[0] for item in top_items]
            return keys

        resultDict = {node: 0 for node in self.nodes}

        clusteringCoefficientDict = self.clusteringCoefficient
        closenessCentralityDict = self.closenessCentrality
        coreNumberDict = self.coreNumber
        hitsDict = self.hits

        # 归一化[标准化]并累加
        for rankDict in [
            clusteringCoefficientDict,
            closenessCentralityDict,
            coreNumberDict,
            hitsDict,
        ]:
            rankDict = standardization(rankDict)
            for key in resultDict:
                resultDict[key] += rankDict[key]

        # 以{ratio*10}%为seeds作trustrank
        seeds = get_top_x_percent_keys(rankDict, ratio)
        resultDict = GraphTool.trust_rank(self.unweightedGraph, seeds=seeds)

        return resultDict


if __name__ == "__main__":
    GraphInfo.test()
