"""
数据加载: 数据集加载为数据批
- DynamicDataLoader: 支持固定分批和动态分批
"""

from typing import Tuple
from math import isclose
import random

from pympler import asizeof
from torch import manual_seed
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Batch
from torch.utils.data import random_split


def create_dataloaders(
    dataset: Dataset,
    ratios: Tuple[float, float, float],
    max_length: int = None,
    batch_size: int = 1,
    shuffles: Tuple[bool, bool, bool] = (True, True, False),
    seed: int = None,
):
    """NOTE: 这个dataloader包含了split_dataset跟load_dataset两个功能"""
    dataloaders = []
    for ind, dataset in enumerate(split_dataset(dataset, ratios, seed)):
        dataloaders.append(
            DynamicDataLoader(
                dataset,
                max_length=max_length,
                batch_size=batch_size,
                shuffle=shuffles[ind],
            ),
        )
    return dataloaders


def split_dataset(
    dataset: Dataset, ratios: Tuple[float, float, float], seed: int = None
):
    train_ratio, val_ratio, test_ratio = ratios
    assert isclose(train_ratio + val_ratio + test_ratio, 1, rel_tol=1e-3), "ratios之和应为1"

    # 计算划分数量
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if seed:
        # 划分数据集
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=manual_seed(seed)
        )
    else:
        # 划分数据集
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    return train_dataset, val_dataset, test_dataset


class DynamicDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        max_length: int = None,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        """
        如果设置了max_length，那就使用动态分批，否则使用固定分批
        NOTE: 动态模式下， 并不能保证每个DataBatch的长度都小于max_length, 因为总会有大的图一张就超过max_length"""
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.isDynamic = False if self.max_length is None else True
        self.databatches = self._build_batches()

    @property
    def size(self):
        return asizeof.asizeof(self.databatches)

    def _build_batches(self):
        databatches = []
        datasetLength = len(self.dataset)
        if self.shuffle:
            indexList = random.sample(range(datasetLength), datasetLength)
        else:
            indexList = list(range(datasetLength))

        if self.isDynamic:
            dataList = []
            current_length = 0
            for ind in indexList:
                data = self.dataset[ind]
                current_length += data.x.shape[0]
                dataList.append(data)
                if current_length >= self.max_length:
                    if len(dataList) > 1:
                        databatch = Batch.from_data_list(dataList[:-1])
                        databatches.append(databatch)
                        dataList = [dataList[-1]]
                        current_length = dataList[-1].x.shape[0]
                    else:
                        databatch = Batch.from_data_list([dataList[0]])
                        databatches.append(databatch)
                        dataList = []
                        current_length = 0
            # 处理剩余的data
            if dataList:
                databatches.append(Batch.from_data_list(dataList))
        else:
            for i in range(0, len(indexList), self.batch_size):
                slice_end = min(
                    i + self.batch_size, len(indexList)
                )  # 计算data_index的结束位置，不超过列表长度
                data_index = indexList[i:slice_end]  # 生成slice
                dataList = [self.dataset[index] for index in data_index]
                databatch = Batch.from_data_list(dataList)
                databatches.append(databatch)
            # 处理剩余的data
            if len(indexList) % self.batch_size != 0:
                leftIndexList = indexList[-(len(indexList) % self.batch_size) :]
                dataList = [self.dataset[index] for index in leftIndexList]
                databatch = Batch.from_data_list(dataList)
                databatches.append(databatch)
        return databatches

    def __iter__(self):
        def generator():
            for databatch in self.databatches:
                yield databatch

        return generator()

    def __len__(self):
        return len(self.databatches)

    def __str__(self):
        return str(
            [
                f"DataBatch(nodeNum={databatch.x.shape[0]}, dataNum={len(databatch)})"
                for databatch in self.databatches
            ]
        )


if __name__ == "__main__":
    import torch
    from data_processing import DatasetLoader
    from utils.toolbox import ProblemType

    max_length = 100
    dataset = DatasetLoader.load_synthetic_dataset(
        "SyntheticDataset-N1", problemType=ProblemType.CN
    )

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, (0.6, 0.3, 0.1))
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(
        dataset
    ), "dataset划分不影响data数"

    dataloader1 = DynamicDataLoader(dataset, max_length, shuffle=True)
    dataloader2 = DynamicDataLoader(dataset, max_length, shuffle=True)
    dataloader3 = DynamicDataLoader(dataset, max_length, shuffle=False)
    dataloader4 = DynamicDataLoader(dataset, max_length, shuffle=False)
    assert (
        str(dataloader1) != str(dataloader2)
        and str(dataloader2) != str(dataloader3)
        and str(dataloader3) == str(dataloader4)
    ), "验证shuffle实现预期效果"

    batch_size = 1
    dataloader5 = DynamicDataLoader(dataset, batch_size=batch_size, shuffle=False)
    for databatch in dataloader5:
        assert len(databatch) == batch_size, "使用非动态模式，每个databatch中的data数应该等于batch_size"

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset, (0.6, 0.3, 0.1), max_length=max_length
    )
    for dataloader in [train_dataloader, val_dataloader, test_dataloader]:
        for databatch in dataloader:
            assert not torch.isnan(databatch.x).any(), "databatch.x不应该出现nan"
            assert not torch.isnan(databatch.y).any(), "databatch.y不应该出现nan"
            assert databatch.x.shape[0] == len(
                torch.unique(databatch.edge_index)
            ), "databatch节点数应该跟特征数相等"
            assert (
                databatch.x.shape[0] == torch.unique(databatch.edge_index).max() + 1
            ), "databatch节点数最大值的限制"
            assert 0 == torch.unique(databatch.edge_index).min(), "databatch节点数最小值的限制"
            if databatch.x.shape[0] > max_length:
                assert (
                    torch.unique(databatch.batch).item() == 0
                ), "databatch长度超过max_length只能出现在单大图的情况下"
