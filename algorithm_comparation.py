from typing import Tuple

import networkx as nx
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx
from matplotlib import pyplot as plt
import scienceplots as sp  # NOTE: 实际上已经导入matplotlib了

plt.rcParams["savefig.bbox"] = "tight"

from utils.toolbox import ProblemType, GraphTool, DEFAULT_NODE_NUM_RANGES
from utils.common import markers_generator
from data_synthesizing import GraphBA, GraphER, GraphSW
from algorithm_collection import _Algorithm
from algorithm_criterion import AccumulatedNormalizedConnectivityCriterion


def compare_algorithms_on_graph(algorithms: Tuple[_Algorithm], graph: nx.Graph):
    resultDict, recordDict = {}, {}
    criterion = AccumulatedNormalizedConnectivityCriterion(graph)
    for problemType in list(ProblemType):
        for algorithm in algorithms:
            order = algorithm.get_order_by_problemType(graph, problemType)
            final_value, record = criterion(order, problemType)
            resultDict[f"{algorithm.name}-{problemType.name}"] = final_value
            recordDict[f"{algorithm.name}-{problemType.name}"] = record
    return resultDict, recordDict


def compare_algorithms_on_data(algorithms: Tuple[_Algorithm], data: Data):
    graph = to_networkx(data, to_undirected=True)
    return compare_algorithms_on_graph(algorithms, graph)


def compare_algorithms_on_dataset(algorithms: Tuple[_Algorithm], dataset: Dataset):
    if len(dataset) == 1:
        data = dataset[0]
        return compare_algorithms_on_data(algorithms, data)
    else:
        resultDict, recordDict = {}, {}
        for problemType in list(ProblemType):
            for algorithm in algorithms:
                resultDict[f"{algorithm.name}-{problemType.name}"] = []
                recordDict[f"{algorithm.name}-{problemType.name}"] = []
        for ind in range(len(dataset)):
            data = dataset[ind]
            a, b = compare_algorithms_on_data(algorithms, data)
            for key in a:
                resultDict[key].append(a[key])
                recordDict[key].append(b[key])
        return resultDict, recordDict


def plot_ANC_curve(
    algorithms: Tuple[_Algorithm],
    recordDict: dict,
    resultDict: dict,
    figSize: Tuple[int, int] = (10, 8),
    useRatio: bool = True,
    datasetName: str = None,
):
    with plt.style.context(["science", "no-latex"]):
        for problemType in list(ProblemType):
            plt.figure(figsize=figSize)
            markerGen = markers_generator()
            for algorithm in algorithms:
                records = [1.0] + [
                    round(value, 3)
                    for value in recordDict[f"{algorithm.name}-{problemType.name}"]
                ]
                if useRatio:
                    M = 11
                    plt.plot(
                        np.linspace(0, 1, M),
                        interp1d(
                            np.linspace(0, 1, len(records)), records, kind="cubic"
                        )(np.linspace(0, 1, M)),
                        label=f'{algorithm.name}({round(resultDict[f"{algorithm.name}-{problemType.name}"], 3)})',
                        marker=next(markerGen),
                    )
                else:
                    plt.plot(
                        np.linspace(0, 1, len(records)),
                        records,
                        label=f'{algorithm.name}({round(resultDict[f"{algorithm.name}-{problemType.name}"], 3)})',
                        marker=next(markerGen),
                    )
            if datasetName is None:
                title = f"{problemType.name} - Graph({len(records)-1})"
            else:
                title = f"{problemType.name} - {datasetName}({len(records)-1})"
            plt.title(title)
            plt.ylabel(f"Residual Connectivity")
            plt.xlabel("Fraction of disconnected nodes")
            plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
            plt.tight_layout()


def plot_ANC_bar(
    algorithms: Tuple[_Algorithm],
    instanceNum: int = 10,
    figSize: Tuple[int, int] = (15, 10),
):
    # 图类型、问题类型、节点数、算法类型
    graphModelClassList = [GraphER, GraphSW, GraphBA]
    pbar = tqdm(
        total=len(graphModelClassList)
        * len(
            list(ProblemType)
            * len(DEFAULT_NODE_NUM_RANGES)
            * instanceNum
            * len(algorithms)
        ),
        leave=False,
    )
    result_graphModel = {}
    for graphModelClass in graphModelClassList:
        result_problemType = {}
        for problemType in list(ProblemType):
            result_nodeNumRange = {}
            for nodeNumRange in DEFAULT_NODE_NUM_RANGES:
                # 初始化nodeNumRange级别的result
                result_nodeNumRange[nodeNumRange] = {}
                for algorithm in algorithms:
                    result_nodeNumRange[nodeNumRange][algorithm.name] = []
                # 累计求和
                for _ in range(instanceNum):
                    nodeNum = np.random.randint(nodeNumRange[0], nodeNumRange[1])
                    graph = graphModelClass(
                        nodeNum=nodeNum,
                        averageDegree=GraphTool.get_random_averageDegree(nodeNum),
                    ).build()
                    # 计算每张图在特定problemType上的ANC
                    result_graph = {}
                    criterion = AccumulatedNormalizedConnectivityCriterion(graph)
                    for algorithm in algorithms:
                        order = algorithm.get_order_by_problemType(graph, problemType)
                        pbar.update(1)
                        result_graph[algorithm.name] = criterion(order, problemType)[0]
                    for key in result_graph:
                        result_nodeNumRange[nodeNumRange][key].append(result_graph[key])
                for key in result_nodeNumRange[nodeNumRange]:
                    result_nodeNumRange[nodeNumRange][key] = round(
                        np.mean(result_nodeNumRange[nodeNumRange][key]), 3
                    )
            result_problemType[problemType.name] = result_nodeNumRange
        result_graphModel[graphModelClass.__name__] = result_problemType

    for graphModelName in result_graphModel:
        result_problemType = result_graphModel[graphModelName]
        for problemTypeName in result_problemType:
            figureData = result_problemType[problemTypeName]

            nodeNumRanges = list(figureData.keys())
            algorithmNames = list(figureData[nodeNumRanges[0]].keys())
            # 按照算法类型整理ANC
            algorithmRecord = {}
            for algorithmName in algorithmNames:
                algorithmRecord[algorithmName] = []
            for nodeNumRange in nodeNumRanges:
                for algorithmName in algorithmNames:
                    algorithmRecord[algorithmName].append(
                        figureData[nodeNumRange][algorithmName]
                    )

            plt.figure(figsize=figSize)

            x = np.arange(len(nodeNumRanges))  # the label locations
            width = 0.7 / len(algorithmNames)  # the width of the bars
            gap = 0.1 / len(algorithmNames)  # the gap between bars
            multiplier = 0

            for algorithmName, algorithmValues in algorithmRecord.items():
                offset = (width + gap) * multiplier
                rects = plt.bar(x + offset, algorithmValues, width, label=algorithmName)
                # plt.bar_label(rects, padding=3) # 显示数值
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            plt.title(f"{problemTypeName} - {graphModelName[5:]}")
            plt.ylabel(f"Accumulated Normalized Connectivity")
            plt.xticks(x + (width + gap) * (len(algorithmNames) - 1) / 2, nodeNumRanges)
            plt.ylim(0, max([max(values) for values in algorithmRecord.values()]) * 1.2)
            plt.legend(loc="upper left")
