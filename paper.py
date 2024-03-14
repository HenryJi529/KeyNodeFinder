from matplotlib import pyplot as plt

from utils.toolbox import GraphTool
from algorithm_collection import *
from algorithm_comparation import *
from env import *
from data_processing import DatasetLoader


def get_algorithms():
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


if __name__ == "__main__":
    datasetName = "karateclub"
    # plot_on_synthetic_datasets()
    plot_on_real_dataset(datasetName)
    # calc_anc_on_dataset(datasetName)
    # get_firstGraphInfo_from_dataset(datasetName)
