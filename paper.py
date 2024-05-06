from matplotlib import pyplot as plt
from networkx import Graph, barabasi_albert_graph
from utils.toolbox import GraphTool
from algorithm_collection import *
from algorithm_comparation import *
from env import *
from data_processing import DatasetLoader
from algorithm_collection import _Algorithm


def get_algorithms() -> list[_Algorithm]:
    algorithms = [
        get_proposed_algorithm(),
        HDAAlgorithm(),
        CIAlgorithm(2),
        KCoreAlgorithm(),
        PageRankAlgorithm(),
    ]
    return algorithms


def plot_on_synthetic_datasets():
    algorithms = get_algorithms()
    plot_ANC_bar(algorithms, instanceNum=3)
    plt.show()


def plot_on_real_dataset(datasetName: str):
    algorithms = get_algorithms()
    datasets = DatasetLoader.load_real_dataset()
    dataset: Dataset = datasets[datasetName]
    resultDict, recordDict = compare_algorithms_on_dataset(algorithms, dataset)
    plot_ANC_curve(
        algorithms, recordDict, resultDict, useRatio=True, datasetName=datasetName
    )
    plt.show()


def calc_anc_on_dataset(datasetName: str):
    algorithms = get_algorithms()
    datasets = DatasetLoader.load_real_dataset()
    dataset = datasets[datasetName]
    resultDict, _ = compare_algorithms_on_dataset(algorithms, dataset)
    print(f"{datasetName}: {resultDict}")


def get_firstGraphInfo_from_dataset(datasetName: str):
    datasets = DatasetLoader.load_real_dataset()
    dataset = datasets[datasetName]
    graph = to_networkx(dataset[0], to_undirected=True)
    print(f"{datasetName}: {GraphTool.summarize_graph(graph)}")


def test_on_averageDegrees(
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
        result = compare_algorithms_on_graphs_by_anc(algorithms, graphs)
        print(result)


if __name__ == "__main__":
    # datasetName = "karateclub"
    # plot_on_synthetic_datasets()
    # plot_on_real_dataset(datasetName)
    # calc_anc_on_dataset(datasetName)
    # get_firstGraphInfo_from_dataset(datasetName)
    test_on_averageDegrees(graphNodeNum=500)
