"""
数据加载: 数据集加载到项目
"""

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
    GitHub,
    KarateClub,
    WikiCS,
    BitcoinOTC,
)

from data_synthesizing import SyntheticDataset
from data_preprocessing import DoubanPreprocessor, ProteinPreprocessor
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
        class CustomDataset(Dataset):
            def len(self):
                return self.__len__()

            def get(self, index: int):
                return self.__getitem__(index)

        class Protein(CustomDataset):
            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "protein",
                pre_transform: BaseTransform = None,
                transform: BaseTransform = None,
            ):
                self.root_path = root
                self.graph = ProteinPreprocessor(
                    root_path=self.root_path
                ).get_raw_graph()
                self.pre_transform = pre_transform
                self.transform = transform

                self.process()

            def __len__(self):
                return 1

            def process(self):
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

                data = torch.load(self.processed_path + "data.pt")
                if self.transform:
                    data = self.transform(data)
                return data

        class Douban(CustomDataset):
            def __init__(
                self,
                root: Path = REAL_DATA_PATH / "douban",
                transform: BaseTransform = None,
            ):
                self.originDataset = DoubanPreprocessor(
                    root_path=root
                ).read_whole_processed_data()
                self.transform = transform

            def __len__(self):
                return len(self.originDataset["items"])

            def get_year(self, index: int):
                return self.originDataset["items"][index]["endYear"]

            def __getitem__(self, index: int):
                """返回graph数据和对应的年份"""
                graph = self.originDataset["items"][index]["graph"]
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
            name="Computers",
            pre_transform=cls.pre_transform,
            transform=cls.transform,
        )
        dataset_karateclub = KarateClub(transform=cls.combine_transform)
        dataset_cora = CitationFull(
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
        dataset_github = GitHub(
            root=REAL_DATA_PATH / "github",
            pre_transform=cls.pre_transform,
            transform=cls.transform,
        )
        dataset_wiki = WikiCS(
            root=REAL_DATA_PATH / "wiki",
            is_undirected=False,
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
        dataset_douban = Douban(
            root=REAL_DATA_PATH / "douban", transform=cls.combine_transform
        )

        return {
            "airport": dataset_airport,  # 基础设施
            "amazon": dataset_amazon,  # 商品关联
            "bitcoin": dataset_bitcoin,  # 金融交易
            "cora": dataset_cora,  # 科学合作
            "douban": dataset_douban,  # 演员网络
            "email": dataset_email,  # 社交通信
            "github": dataset_github,  # 科学合作
            "karateclub": dataset_karateclub,  # 经典网络
            "protein": dataset_protein,  # 生物网络
            "wiki": dataset_wiki,  # 知识图谱
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
    print(len(real_dataset["douban"]))
    print(real_dataset["douban"][10])
    print(len(real_dataset["protein"]))
    print(real_dataset["douban"][0])
