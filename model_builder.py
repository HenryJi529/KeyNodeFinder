from typing import Union, List, Tuple
from enum import Enum

from torch import nn
from torch import Tensor
import torch
from torch.nn import functional as F
from torch_geometric import nn as pyg_nn
from torch_geometric.nn import Linear, TopKPooling, GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.profile import get_model_size

from utils.toolbox import GraphTool


def generate_concatenation(featureSlice: Tensor):
    length = featureSlice.shape[0]
    featureNum = featureSlice.shape[1]
    concatenationNum = int(length * (length - 1) / 2)
    concatenation = torch.zeros((concatenationNum, featureNum * 2)).to(
        featureSlice.device
    )
    index = 0
    for i in range(length - 1):
        for j in range(i + 1, length):
            concatenation[index] = torch.concat(
                [featureSlice[i, :], featureSlice[j, :]]
            )
            index += 1
    return concatenation


class Task(Enum):
    RANK = 1
    VALUE = 2


class NiceModel(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        embedding_units: Tuple[int, int],
        ranking_units: int,
        valuing_units: List[int],
    ):
        super().__init__()

        if embedding_units[0] < 2:
            return ValueError("嵌入层层数至少为2")

        # 节点嵌入层
        self.embedding_layer = nn.ModuleList()
        self.embedding_layer.append(GCNConv(input_features, embedding_units[1]))
        self.embedding_layer.append(pyg_nn.norm.BatchNorm(embedding_units[1]))
        for i in range(embedding_units[0]):
            self.embedding_layer.append(GATConv(embedding_units[1], embedding_units[1]))
            self.embedding_layer.append(pyg_nn.norm.BatchNorm(embedding_units[1]))
        self.embedding_layer.append(GCNConv(embedding_units[1], output_features))
        # 节点排序层
        self.ranking_layer = nn.Sequential(
            Linear(output_features * 2, ranking_units),
            pyg_nn.norm.LayerNorm(ranking_units),
            nn.GELU(),
            Linear(ranking_units, 1),
        )
        # 状态评估层
        valuing_unit_num = [output_features] + valuing_units + [1]
        valuing_layer_list = []
        for i in range(len(valuing_unit_num) - 1):
            valuing_layer_list.append(
                Linear(valuing_unit_num[i], valuing_unit_num[i + 1])
            )
            if i == len(valuing_layer_list) - 2:
                pass
            else:
                valuing_layer_list.append(
                    pyg_nn.norm.BatchNorm(valuing_unit_num[i + 1])
                )
                valuing_layer_list.append(nn.ReLU())

        self.valuing_layer = nn.Sequential(*valuing_layer_list)

    def embed(self, x: Tensor, edge_index: Tensor):
        normalizedAdjacencyMatrix = (
            GraphTool.get_normalizedAdjacencyMatrix_from_edgeIndex(
                edge_index, x.shape[0]
            )
        )
        for ind, layer in enumerate(self.embedding_layer):
            if ind % 2 == 0:
                if ind in [0, 2, ind == len(self.embedding_layer) - 2]:
                    x = layer(x, edge_index)
                else:
                    x = layer(x, edge_index) + normalizedAdjacencyMatrix @ x
            else:
                x = F.relu(layer(x))
        embedded_vectors = x
        return embedded_vectors

    def rank(self, batch: Tensor, embedded_vectors: Tensor):
        dataNum = (torch.max(batch) + 1).item()
        concatenationList = []
        for ind in range(dataNum):
            data_indexes = torch.where(batch == ind)[0]
            concatenationList.append(
                generate_concatenation(embedded_vectors[data_indexes, :])
            )
        ranking_logits = self.ranking_layer(torch.cat(concatenationList, dim=0))
        return ranking_logits

    def value(self, embedded_vectors: Tensor):
        return self.valuing_layer(embedded_vectors)

    def forward(self, databatch: Batch, task: Task = Task.RANK):
        x, edge_index, batch = databatch.x, databatch.edge_index, databatch.batch

        if torch.numel(edge_index) == 0:
            if task == Task.VALUE:
                # NOTE: 如果传入的是一个散点图(没有边)，则返回一个全0的values
                return torch.zeros([x.shape[0], 1], requires_grad=True).to(self.device)
            else:
                # NOTE: 理论上不存在这种可能
                pass

        embedded_vectors = self.embed(x, edge_index)
        if task == Task.RANK:
            return self.rank(batch, embedded_vectors)
        else:
            return self.value(embedded_vectors)

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def size(self) -> float:
        """
        @return: (float) size of the model in kb units
        """
        return round(get_model_size(self) / 1024, 1)


if __name__ == "__main__":
    from data_loading import create_dataloaders
    from data_processing import DatasetLoader
    from model_handler import get_modelParamDict_example
    from utils.toolbox import ProblemType

    dataset = DatasetLoader.load_synthetic_dataset(
        f"SyntheticDataset-N{5}", problemType=ProblemType.CN
    )

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset,
        (0.6, 0.3, 0.1),
        max_length=100,
        shuffles=[False, False, False],
        seed=42,
    )
    databatch: Batch = next(iter(train_dataloader))

    nodeNumList = [
        len(torch.where(databatch.batch == ind)[0])
        for ind in range((torch.max(databatch.batch) + 1).item())
    ]

    modelParamDict = get_modelParamDict_example()
    model = NiceModel(**modelParamDict)

    embedded_vectors: Tensor = model.embed(databatch.x, databatch.edge_index)
    ranking_logits: Tensor = model(databatch)

    assert embedded_vectors.shape == (
        databatch.x.shape[0],
        modelParamDict["output_features"],
    ), "embedding shape error"
    assert ranking_logits.shape == (
        int(sum([nodeNum * (nodeNum - 1) / 2 for nodeNum in nodeNumList])),
        1,
    ), "ranking shape error"
    assert type(model.size) == float, "Size of model must be a float"
