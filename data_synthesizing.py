"""
合成数据集构建
"""

from pathlib import Path
import argparse
from typing import List, Union, Tuple
from abc import ABC, abstractmethod
import shutil
import os

import networkx as nx
import numpy as np
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm
import torch

from utils.common import (
    colored_print,
    download_file_from_ftp,
    upload_file_to_ftp,
    zip_directory,
    unzip_file,
)
from utils.toolbox import (
    ProblemType,
    FeatureBuilder,
    GraphTool,
    RankingProcessor,
    DATA_PATH,
    SYNTHETIC_DATA_PATH,
    DEFAULT_NODE_NUM_RANGES,
)
from algorithm_collection import PageRankAlgorithm


class GraphModel(ABC):
    TYPE = None

    @abstractmethod
    def __init__(self, nodeNum: int, averageDegree: int):
        pass

    @abstractmethod
    def build(self, *args) -> nx.Graph:
        pass


class GraphER(GraphModel):
    """Erdős-Rényi(ER)模型"""

    TYPE = "erdos_renyi"

    def __init__(self, nodeNum: int, averageDegree: int):
        """<k> = p * (N - 1)"""
        self.n = nodeNum
        self.p = averageDegree / (nodeNum - 1)

    def build(self):
        return nx.erdos_renyi_graph(n=self.n, p=self.p)


class GraphSW(GraphModel):
    """Small-world(SW)模型"""

    TYPE = "small_world"

    def __init__(self, nodeNum: int, averageDegree: int):
        """<k> = k"""
        if averageDegree % 2 == 1:
            raise ValueError("K值必须为偶数")
        self.n = nodeNum
        self.k = averageDegree
        self.p = 0.1  # NOTE: 参考Finder的参数设计

    def build(self):
        return nx.connected_watts_strogatz_graph(n=self.n, k=self.k, p=self.p)


class GraphBA(GraphModel):
    """Barabási-Albert(BA)模型"""

    TYPE = "barabasi_albert"

    def __init__(self, nodeNum: int, averageDegree: int):
        """<k> = 2 * m"""
        self.n = nodeNum
        self.m = int(averageDegree / 2)

    def build(self):
        return nx.barabasi_albert_graph(n=self.n, m=self.m)


class Factory:
    def __init__(
        self,
        path: Path,
        graphModelClasses: List[GraphModel],
        nodeNumRanges: List[Tuple[int, int]],
        instanceNum: int,
    ):
        """根据参数分段生成足够的无节点属性无权无向图"""
        self.path = path
        self.path.mkdir(exist_ok=True, parents=True)

        self.graphModelClasses = graphModelClasses
        self.nodeNumRanges = nodeNumRanges
        self.instanceNum = instanceNum

    def produce(self):
        for graphModelClass in self.graphModelClasses:
            graph_model_path: Path = self.path / graphModelClass.TYPE
            if not graph_model_path.exists():
                graph_model_path.mkdir()
            for nodeNumRange in self.nodeNumRanges:
                instances_path = (
                    graph_model_path / f"n{nodeNumRange[0]}-{nodeNumRange[1]}"
                )
                instances_path.mkdir(exist_ok=True)
                for ind in range(self.instanceNum):
                    nodeNum = np.random.randint(nodeNumRange[0], nodeNumRange[1])
                    instance_path = instances_path / f"g_{ind}.csv"
                    if instance_path.exists():
                        continue
                    graph = graphModelClass(
                        nodeNum=nodeNum,
                        averageDegree=GraphTool.get_random_averageDegree(nodeNum),
                    ).build()
                    nx.write_edgelist(
                        graph, instance_path, delimiter=",", data=["weight"]
                    )
                    # nx.write_weighted_edgelist(graph, instance_path, delimiter=",")


class SyntheticDataset(Dataset):
    def __init__(
        self,
        root: Path = SYNTHETIC_DATA_PATH,
        datasetName: str = None,
        problemType: ProblemType = ProblemType.CN,
        graphModelClasses: List[GraphModel] = [
            GraphER,
            GraphSW,
            GraphBA,
        ],
        nodeNumRanges: List[Tuple[int, int]] = DEFAULT_NODE_NUM_RANGES,
        instanceNum: int = 10,  # NOTE: 优先级低于datasetName
        transform: BaseTransform = None,
    ):
        self.root = root
        self.problemType = problemType
        self.graphModelClasses = graphModelClasses
        self.nodeNumRanges = nodeNumRanges
        self.transform = transform

        self.rebuild = True if datasetName is None else False
        self.instanceNum = (
            instanceNum
            if self.rebuild
            else int(
                datasetName.split("-N")[1]
            )  # NOTE: 这里最好加上一道datasetName检查
        )

        self.graph_num = (
            len(self.graphModelClasses) * len(self.nodeNumRanges) * self.instanceNum
        )

        if self.rebuild:
            self.upload = (
                False
                if len(list(self.raw_root.glob("**/*.csv"))) > self.graph_num
                else True
            )
            if not self.isRawReady:
                colored_print("生成原始图数据...")
                Factory(
                    path=self.raw_root,
                    graphModelClasses=self.graphModelClasses,
                    instanceNum=self.instanceNum,
                    nodeNumRanges=self.nodeNumRanges,
                ).produce()
                colored_print("原始图数据生成完毕")
            if not self.isProcessedReady:
                colored_print("数据集生成中...")
                self.process()
                colored_print("数据集生成完毕")
            if self.upload:
                colored_print("数据集上传中...")
                zip_filename = f"SyntheticDataset-N{self.instanceNum}.zip"
                zip_directory(str(SYNTHETIC_DATA_PATH), str(DATA_PATH / zip_filename))
                upload_file_to_ftp(str(DATA_PATH), zip_filename)
                os.remove(DATA_PATH / zip_filename)
                colored_print("数据集上传完毕")
            else:
                print("由于存在不属于该数据集的图数据，因而自动制止了上传行为...")
        else:
            if self.root.exists():
                colored_print("清理过时数据集...")
                shutil.rmtree(self.root)
                colored_print("清理完成")
            colored_print("数据集下载中...")
            download_file_from_ftp(
                local_directory=str(self.root.parent),
                filename=datasetName + ".zip",
            )
            colored_print("数据集下载完成")
            colored_print("数据集解压中...")
            unzip_file(self.root, str(self.root.parent) + "/" + datasetName + ".zip")
            colored_print("数据集解压完成")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__len__()})"

    @property
    def raw_root(self):
        return self.root / "raw"

    @property
    def processed_root(self):
        return self.root / "processed"

    @property
    def isRawReady(self):
        if not self.raw_root.exists():
            print("raw_root路径不存在")
            return False
        if not self.raw_root.is_dir():
            print("raw_root路径为文件")
            raise ValueError("raw_root路径为文件")
        if len(self.raw_paths) != self.graph_num:
            print("raw_root路径下图数据量不满足参数要求")
            return False
        return True

    @property
    def isProcessedReady(self):
        if not self.processed_root.exists():
            print("processed_root路径不存在")
            return False
        if not self.processed_root.is_dir():
            print("processed_root路径为文件")
            raise ValueError("processed_root路径为文件")
        if len(self.processed_paths) != self.graph_num:
            print("processed_root路径下图数据量不满足参数要求")
            return False
        return True

    @property
    def raw_paths(self):
        raw_paths = sorted(list(self.raw_root.glob("**/*.csv")))
        if len(raw_paths) > self.graph_num:
            raw_paths = raw_paths[: self.graph_num]
        return raw_paths

    @property
    def processed_paths(self):
        processed_paths = sorted(list(self.processed_root.glob("**/*.pt")))
        if len(processed_paths) > self.graph_num:
            processed_paths = processed_paths[: self.graph_num]
        return processed_paths

    def process(self):
        for raw_path in tqdm(self.raw_paths, desc="生成"):
            # 获取对应的processed_path
            graph_filename = raw_path.stem + ".pt"
            graph_type = raw_path.parent.parent.name
            graph_param = raw_path.parent.name
            processed_path = (
                self.processed_root / graph_type / graph_param / graph_filename
            )
            # 忽略processed_path已存在的情况
            if processed_path.exists():
                continue
            # 计算data
            data = self.process_graph(raw_path)
            # 保存data
            if not processed_path.parent.exists():
                processed_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(data, processed_path)

    def process_graph(self, raw_path: Path):
        graph = self.getG(raw_path)
        X = self.getX(graph)
        Y = self.getY(graph)
        E = self.getE(graph)
        return Data(x=X, edge_index=E, y=Y)

    def __len__(self):
        assert len(self.processed_paths) == len(self.raw_paths), "原始数据处理不完全"
        return len(self.raw_paths)

    def __getitem__(self, slice_obj: Union[slice, int]):
        def get_data(path: Path):
            data = torch.load(path)
            if self.transform:
                data = self.transform(data)
            data.y = data.y[:, list(ProblemType).index(self.problemType)]
            return data

        if isinstance(slice_obj, slice):
            start = slice_obj.start if slice_obj.start else 0
            stop = slice_obj.stop if slice_obj.stop else self.__len__()
            step = slice_obj.step if slice_obj.step else 1 if start <= stop else -1
            return [
                get_data(self.processed_paths[ind]) for ind in range(start, stop, step)
            ]
        elif isinstance(slice_obj, int):
            self.processed_path = self.processed_paths[slice_obj]
            data = get_data(self.processed_path)
            return data
        else:
            raise TypeError("Invalid argument type")

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self.__len__()

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        return self.__getitem__(idx)

    @property
    def num_node_features(self):
        return self.__getitem__(0).x.shape[1]

    @property
    def num_edge_features(self):
        return 0

    def getG(self, path: Path) -> nx.Graph:
        return nx.read_edgelist(path, delimiter=",", nodetype=int)

    def getX(self, graph: nx.Graph) -> torch.Tensor:
        return FeatureBuilder.getFeatureArray(graph)

    def getY(self, graph) -> torch.Tensor:
        algorithm = PageRankAlgorithm()
        ranks = [
            RankingProcessor.get_rank_from_order(
                torch.tensor(algorithm.get_order_by_problemType(graph, problemType))
            )
            for problemType in list(ProblemType)
        ]
        return torch.vstack(ranks).t().contiguous()

    def getE(self, graph) -> torch.Tensor:
        return GraphTool.get_edgeIndex_from_graph(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument(
        "--instance_num", required=False, type=int, default=1, help="设置instanceNum"
    )
    args = parser.parse_args()
    dataset = SyntheticDataset(instanceNum=args.instance_num)
    print(dataset)

    colored_print("验证数据集合理性....")
    for ind in tqdm(range(len(dataset)), desc="验证"):
        assert not torch.isnan(dataset[ind].x).any(), "x不应该出现nan"
        assert not torch.isnan(dataset[ind].y).any(), "y不应该出现nan"
        assert dataset[ind].x.shape[0] == len(
            torch.unique(dataset[ind].edge_index)
        ), "节点数应该跟特征数相等"
        assert (
            dataset[ind].x.shape[0]
            == dataset[ind].y.shape[0]
            == len(dataset.getG(dataset.raw_paths[ind]).nodes)
        ), "Data的x,y的m应当相等且等于节点数"

    assert (
        len(dataset[:2]) == 2 and len(dataset[0:1]) == 1 and len(dataset[2:1]) == 1
    ), "dataset分片存在问题"
    colored_print("验证流程结束")
