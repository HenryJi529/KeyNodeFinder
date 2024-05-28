from matplotlib import pyplot as plt
from networkx import Graph, barabasi_albert_graph
from utils.toolbox import GraphTool
from algorithm_collection import *
from algorithm_comparation import *
from env import *
from data_processing import DatasetLoader
from algorithm_collection import _Algorithm


def get_algorithms(flag: True) -> list[_Algorithm]:
    algorithms = [
        get_proposed_algorithm(flag),
        HDAAlgorithm(),
        CIAlgorithm(2),
        KCoreAlgorithm(),
        PageRankAlgorithm(),
    ]
    return algorithms


def plot_on_synthetic_datasets(instanceNum: int = 3):
    algorithms = get_algorithms()
    plot_ANC_bar(algorithms, instanceNum=instanceNum)
    plt.show()


def plot_on_real_dataset(datasetName: str):
    algorithms = get_algorithms()
    datasets = DatasetLoader.load_real_dataset()
    dataset: Dataset = datasets[datasetName]
    resultDict, recordDict = AlgorithmComparator(algorithms).compare_on_dataset_by_all(
        dataset
    )
    plot_ANC_curve(
        algorithms, recordDict, resultDict, useRatio=True, datasetName=datasetName
    )
    plt.show()


def calc_anc_on_dataset(datasetName: str):
    algorithms = get_algorithms()
    datasets = DatasetLoader.load_real_dataset()
    dataset = datasets[datasetName]
    resultDict, _ = AlgorithmComparator(algorithms).compare_on_dataset_by_all(dataset)
    print(f"{datasetName}: {resultDict}")


def get_firstGraphInfo_from_dataset(datasetName: str):
    datasets = DatasetLoader.load_real_dataset()
    dataset = datasets[datasetName]
    graph = to_networkx(dataset[0], to_undirected=True)
    print(f"{datasetName}: {GraphTool.summarize_graph(graph)}")


def compare_time_by_graphNodeNums(
    networkDensity: float = 0.3,
    graphNum: int = 1,
    graphNodeNums: list[int] = [200, 400, 800, 1600, 3200],
):
    algorithms = get_algorithms()
    print(f"网络密度: {networkDensity}")
    for graphNodeNum in graphNodeNums:
        print(f"graphNodeNum: {graphNodeNum}")
        averageDegree = networkDensity * graphNodeNum
        graphs: list[Graph] = [
            barabasi_albert_graph(graphNodeNum, int(averageDegree / 2))
            for _ in range(graphNum)
        ]
        result = AlgorithmComparator(algorithms).compare_on_graphs_by_meanTime(graphs)
        print(result)


def compare_anc_by_averageDegrees(
    graphNum: int = 1,
    graphNodeNum: int = 500,
    averageDegreeList: list[int] = [2, 5, 10, 20, 50, 100],
):
    algorithms = get_algorithms()
    print(f"节点数: {graphNodeNum}")
    for averageDegree in averageDegreeList:
        print(f"averageDegree: {averageDegree}")
        graphs: list[Graph] = [
            barabasi_albert_graph(graphNodeNum, int(averageDegree / 2))
            for _ in range(graphNum)
        ]
        result = AlgorithmComparator(algorithms).compare_on_graphs_by_meanANC(graphs)
        print(result)


if __name__ == "__main__":
    # plot_on_synthetic_datasets(1)
    # datasetName = "karateclub"
    # plot_on_real_dataset(datasetName)
    # calc_anc_on_dataset(datasetName)
    # get_firstGraphInfo_from_dataset(datasetName)
    # compare_anc_by_averageDegrees(graphNodeNum=50, averageDegreeList=[2, 10])
    compare_time_by_graphNodeNums()
