"""
数据加载: 数据集加载到项目
"""

import os
import requests
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import (
    BaseTransform,
    NormalizeFeatures as NormalizeNodeTransform,
    Compose,
    ToUndirected,
)
from torch_geometric.datasets import (
    Airports,
    Amazon,
    CitationFull,
    EmailEUCore,
    KarateClub,
    BitcoinOTC,
)

from data_synthesizing import SyntheticDataset
from data_preprocessing import DoubanPreprocessor
from utils.toolbox import (
    ProblemType,
    FeatureBuilder,
    GraphTool,
    SYNTHETIC_DATA_PATH,
    REAL_DATA_PATH,
)


class BuildFeaturesTransform(BaseTransform):
    def __call__(self, data: Data):
        """(删除自环后)重构节点特征"""
        graph = nx.Graph()
        graph.add_edges_from(data.edge_index.t().tolist())

        # 删除自环
        selfloop_edges = list(nx.selfloop_edges(graph))
        if selfloop_edges:
            graph.remove_edges_from(selfloop_edges)
            # NOTE: 理论上要确保edge_index连续，但考虑到删除边不会破坏原本的连续性，因此这里可以简单的写
            data.edge_index = torch.tensor(list(graph.edges)).t().contiguous()

        return Data(x=FeatureBuilder.getFeatureArray(graph), edge_index=data.edge_index)


class StandardizeFeatureTransform(BaseTransform):
    """Feature-wise Z-score归一化"""

    def __call__(self, data: Data):
        means = torch.mean(data.x, dim=0)  # 计算每一列的均值
        stds = torch.std(data.x, dim=0)  # 计算每一列的标准差

        # 处理零方差列
        zero_stds_mask = stds == 0
        stds[zero_stds_mask] = 1.0  # 将零方差列的标准差设置为1，避免分母为零

        # 归一化
        data.x = (data.x - means) / stds

        # 将零方差列的数据设置为0
        data.x[:, zero_stds_mask] = 0.0

        return data


class ToFloat32Transform(BaseTransform):
    """转化数据为float32"""

    def __call__(self, data: Data):
        data.x = data.x.to(torch.float32)
        return data


class DatasetLoader:
    transform = Compose(
        [
            ToFloat32Transform(),
            StandardizeFeatureTransform(),
            NormalizeNodeTransform(),
        ]
    )
    pre_transform = Compose(
        [
            ToUndirected(),
            BuildFeaturesTransform(),
        ]
    )
    combine_transform = Compose([pre_transform, transform])

    @classmethod
    def load_real_dataset(cls):
        class PracticalDataset(Dataset):
            def len(self):
                return self.__len__()

            def get(self, index: int):
                return self.__getitem__(index)

        class SingleGraphDatasetFromInternet(PracticalDataset):
            URL = None

            def __init__(
                self,
                root: Path,
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                self.root_path = root
                self.graph = self._get_raw_graph()
                self.pre_transform = pre_transform
                self.transform = transform
                self._process()

            def __len__(self):
                return 1

            @property
            def filename(self) -> str:
                return os.path.basename(self.URL)

            def _get_raw_graph(self) -> nx.Graph:
                self.raw_path = self.root_path / "raw"
                self.raw_file: Path = self.raw_path / self.filename

                self.raw_path.mkdir(exist_ok=True, parents=True)
                if self.raw_file.exists():
                    pass
                else:
                    response = requests.get(self.URL)
                    with open(self.raw_file, "wb") as f:
                        f.write(response.content)

                return GraphTool.read_graph(self.raw_file)

            def _process(self):
                # 原始Data构建
                edge_index = GraphTool.get_edgeIndex_from_graph(self.graph)
                data = Data(edge_index=edge_index)
                # 执行preTransform并保存
                self.processed_path = self.root_path / "processed"
                self.processed_path.mkdir(exist_ok=True)
                if self.pre_transform:
                    data = self.pre_transform(data)
                torch.save(data, self.processed_path / "data.pt")

            def __getitem__(self, index: int):
                if index != 0:
                    return ValueError("该数据集只有一张图...")

                data = torch.load(self.processed_path / "data.pt")
                if self.transform:
                    data = self.transform(data)
                return data

        class Protein(SingleGraphDatasetFromInternet):
            """http://www.interactome-atlas.org/download"""

            URL = "http://www.interactome-atlas.org/data/HI-II-14.tsv"

            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "protein",
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                super().__init__(root, pre_transform, transform)

        class EuroRoad(SingleGraphDatasetFromInternet):
            """https://networkrepository.com/inf-euroroad.php"""

            URL = "https://morningstar369.com/media/others/datasets/euroroad.edges"

            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "euroroad",
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                super().__init__(root, pre_transform, transform)

        class Beacxc(SingleGraphDatasetFromInternet):
            """https://networkrepository.com/econ-beacxc.php"""

            URL = "https://morningstar369.com/media/others/datasets/beacxc.txt"

            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "beacxc",
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                super().__init__(root, pre_transform, transform)

        class Diseasome(SingleGraphDatasetFromInternet):
            """https://networkrepository.com/bio-diseasome.php"""

            URL = "https://morningstar369.com/media/others/datasets/diseasome.txt"

            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "diseasome",
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                super().__init__(root, pre_transform, transform)

        class Bible(SingleGraphDatasetFromInternet):
            """http://konect.cc/networks/moreno_names/"""

            URL = "https://morningstar369.com/media/others/datasets/bible.txt"

            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "bible",
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                super().__init__(root, pre_transform, transform)

        class Douban(PracticalDataset):
            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "douban",
                transform: BaseTransform = None,
            ):
                self.originDataDict = DoubanPreprocessor(
                    root_path=root
                ).read_processed_data()
                self.transform = transform

            def __len__(self):
                return len(self.originDataDict["items"])

            def get_year(self, index: int):
                return self.originDataset["items"][index]["endYear"]

            def __getitem__(self, index: int):
                """返回graph数据和对应的年份"""
                graph = self.originDataDict["items"][index]["graph"]
                edge_index = GraphTool.get_edgeIndex_from_graph(graph)
                data = Data(edge_index=edge_index)
                if self.transform:
                    data = self.transform(data)
                return data

        dataset_airport = Airports(
            root=REAL_DATA_PATH / "airport",
            name="USA",
            pre_transform=cls.pre_transform,
            transform=cls.transform,
        )
        dataset_amazon = Amazon(
            root=REAL_DATA_PATH / "amazon",
            name="Photo",
            pre_transform=cls.pre_transform,
            transform=cls.transform,
        )
        dataset_karateclub = KarateClub(transform=cls.combine_transform)
        dataset_coraml = CitationFull(
            root=REAL_DATA_PATH / "citation",
            name="Cora_ML",
            pre_transform=cls.pre_transform,
            transform=cls.transform,
        )
        dataset_email = EmailEUCore(
            root=REAL_DATA_PATH / "email",
            pre_transform=cls.pre_transform,
            transform=cls.transform,
        )
        dataset_bitcoin = BitcoinOTC(
            root=REAL_DATA_PATH / "bitcoin",
            edge_window_size=10,
            transform=cls.transform,
            pre_transform=cls.pre_transform,
        )

        dataset_protein = Protein(
            root=REAL_DATA_PATH / "protein",
            transform=cls.transform,
            pre_transform=cls.pre_transform,
        )
        dataset_euroroad = EuroRoad(
            root=REAL_DATA_PATH / "euroroad",
            transform=cls.transform,
            pre_transform=cls.pre_transform,
        )
        dataset_beacxc = Beacxc(
            root=REAL_DATA_PATH / "beacxc",
            transform=cls.transform,
            pre_transform=cls.pre_transform,
        )
        dataset_diseasome = Diseasome(
            root=REAL_DATA_PATH / "diseasome",
            transform=cls.transform,
            pre_transform=cls.pre_transform,
        )
        dataset_bible = Bible(
            root=REAL_DATA_PATH / "bible",
            transform=cls.transform,
            pre_transform=cls.pre_transform,
        )
        dataset_douban = Douban(
            root=REAL_DATA_PATH / "douban", transform=cls.combine_transform
        )

        return {
            "airport": dataset_airport,
            "amazon": dataset_amazon,
            "beacxc": dataset_beacxc,
            "bible": dataset_bible,
            "bitcoin": dataset_bitcoin,
            "coraml": dataset_coraml,
            "diseasome": dataset_diseasome,
            "douban": dataset_douban,
            "email": dataset_email,
            "euroroad": dataset_euroroad,
            "karateclub": dataset_karateclub,
            "protein": dataset_protein,
        }

    @classmethod
    def load_synthetic_dataset(
        cls,
        syntheticDatasetName: str = "SyntheticDataset-N1",
        problemType: ProblemType = ProblemType.CN,
    ):
        dataset_synthetic = SyntheticDataset(
            root=SYNTHETIC_DATA_PATH,
            datasetName=syntheticDatasetName,
            problemType=problemType,
            transform=cls.transform,
        )
        return dataset_synthetic

    @classmethod
    def load_single_graph(cls, graph: nx.Graph):
        """加载一个单图
        @return data, nodeMap: 返回处理后的data与原图跟data.edge_index中节点的对应关系
        """
        edge_index, node_mapping = GraphTool.get_edgeIndex_from_graph(
            graph, return_node_mapping=True
        )
        data = cls.combine_transform(Data(edge_index=edge_index))

        return data, node_mapping


if __name__ == "__main__":
    real_dataset = DatasetLoader().load_real_dataset()
    print(len(real_dataset["douban"]), real_dataset["douban"][10])
    print(len(real_dataset["protein"]), real_dataset["protein"][0])
    print(len(real_dataset["amazon"]), real_dataset["amazon"][0])
    print(len(real_dataset["diseasome"]), real_dataset["diseasome"][0])
    print(len(real_dataset["euroroad"]), real_dataset["euroroad"][0])
    print(len(real_dataset["beacxc"]), real_dataset["beacxc"][0])
    print(len(real_dataset["bible"]), real_dataset["bible"][0])
