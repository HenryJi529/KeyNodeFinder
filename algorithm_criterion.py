"""
评价指标:
- 归一化加权逆序数
- 相关性
- 累积归一化连接性
"""

from typing import List, Tuple, Dict, Callable, Union
import random

import networkx as nx
from scipy.stats import spearmanr, kendalltau

from utils.toolbox import (
    ProblemType,
    ProblemMetric,
    GraphTool,
)


class NormalizedWeightedInverseOrderCriterion:
    @staticmethod
    def is_order_valid(order):
        # NOTE: order必须要是一个元素属于0~N-1集合的列表
        if set(order) == set(range(len(order))):
            return True
        else:
            return False

    def __init__(self, correct_order):
        if not self.is_order_valid(correct_order):
            raise ValueError(f"{correct_order} 不是节点排序")
        self.correct_order = correct_order

        self.node_mapping = {
            node: index for index, node in enumerate(self.correct_order)
        }

        self.weightedInverseOrderSum = sum(
            [
                np.log(
                    np.prod(
                        [
                            np.e + distance
                            for distance in range(1, len(self.correct_order) - location)
                        ]
                    )
                )
                / (location + 1)
                for location in range(len(self.correct_order))
            ]
        )

    def __call__(self, input_order):
        if not self.is_order_valid(input_order):
            raise ValueError(f"{input_order} 不是节点排序")
        # 判断input_order中的元素集合是否与correct_order中一致
        if not len(input_order) == len(self.correct_order):
            raise ValueError(f"{input_order}的节点无法对应correct_order")
        # 获取input_index
        input_index = [self.node_mapping[node] for node in input_order]

        weightedInverseOrder = 0
        for i in range(1, len(input_index)):
            for j in range(i):
                index_i = input_index[i]
                index_j = input_index[j]
                if index_i < index_j:
                    location = index_i
                    distance = index_j - index_i
                    weightedInverseOrder += np.log(np.e + distance) / (location + 1)

        normalizedWeightedInverseOrder = (
            weightedInverseOrder / self.weightedInverseOrderSum
        )

        return normalizedWeightedInverseOrder


class CorrelationCriterion:
    def __init__(self, correct_order: list[int]):
        self.correct_order = correct_order

    def get_value_by_method(self, method: Callable, input_order: list[int]):
        coefficient, pValue = method(self.correct_order, input_order)
        return {"coefficient": coefficient, "pValue": pValue}

    def __call__(self, input_order):
        result = {}
        result["spearman"] = self.get_value_by_method(spearmanr, input_order)
        result["kendall"] = self.get_value_by_method(kendalltau, input_order)
        return result


class AccumulatedNormalizedConnectivityCriterion:
    def __init__(
        self,
        graph: nx.Graph,
        disconnection_cost: Dict[int, float] = None,
    ):
        self.graph = graph
        if disconnection_cost:
            assert len(disconnection_cost) == len(graph.nodes), "断连成本字典长度必须与节点数一致"
            self.disconnection_cost = {
                node: disconnection_cost[node] / sum(disconnection_cost.values())
                for node in disconnection_cost
            }
            self.weighted = True
        else:
            self.weighted = False

    def get_result_by_metricFn(
        self,
        disconnection_order: List[int],
        metric_fn: Callable[[nx.Graph], Union[int, float]],
    ) -> Tuple[float, List[float]]:
        """计算ANC得分
        :param metric_fn: `function` 公式中的sigma
        :return final_value: `float` 最终得分
        :return record: `List[float]` 连通性列表
        """
        origin_value = metric_fn(self.graph)
        final_value = 0
        record = []

        graph = self.graph.copy()
        for node in disconnection_order:
            GraphTool.disconnect_node(graph, node)
            current_value = metric_fn(graph)
            ratio = current_value / origin_value
            if self.weighted:
                final_value += ratio * self.disconnection_cost[node]
            else:
                final_value += ratio / len(disconnection_order)
            record.append(ratio)

        return final_value, record

    def __call__(self, disconnection_order: List[int], problemType: ProblemType):
        metricFn = ProblemMetric.get(problemType)
        return self.get_result_by_metricFn(disconnection_order, metricFn)


if __name__ == "__main__":
    import networkx as nx
    import numpy as np

    def test_AccumulatedNormalizedConnectivityCriterion():
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (1, 2),
                (2, 3),
                (3, 4),
                (5, 6),
                (1, 4),
            ]
        )  # 生成的图结构为[1, 2, 3, 4], [5, 6]
        disconnection_order = [1, 2, 3, 4, 5, 6]
        disconnection_order1 = [1, 2, 6]
        disconnection_order2 = [6, 2, 1]
        disconnection_cost = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}

        criterion_cost = AccumulatedNormalizedConnectivityCriterion(
            graph, disconnection_cost
        )
        criterion = AccumulatedNormalizedConnectivityCriterion(graph)

        assert np.isclose(
            criterion_cost(disconnection_order, ProblemType.ND)[0],
            0.416667,
        ), "带权重ND计算错误"
        assert np.isclose(
            criterion(disconnection_order, ProblemType.ND)[0],
            0.458333,
        ), "无权重ND计算错误"
        assert np.isclose(
            criterion(disconnection_order, ProblemType.CN)[0],
            0.190476,
        ), "无权重CN计算错误"
        assert (
            criterion(disconnection_order1, ProblemType.CN)[0]
            < criterion(disconnection_order2, ProblemType.CN)[0]
        ), "无权重CN逻辑错误, 先断连重要的点获得的值应该更小"
        assert (
            criterion(disconnection_order1, ProblemType.ND)[0]
            < criterion(disconnection_order2, ProblemType.ND)[0]
        ), "无权重ND逻辑错误, 先断连重要的点获得的值应该更小"

    def test_CorrelationCriterion():
        correct_order = list(range(0, 10000))
        input_order1 = random.sample(correct_order, len(correct_order))

        criterion = CorrelationCriterion(correct_order)
        assert np.isclose(
            criterion(correct_order)["spearman"]["coefficient"], 1
        ), "一致的序列得分应该接近1"
        assert np.isclose(
            criterion(input_order1)["spearman"]["coefficient"],
            0,
            atol=5e-2,
        ), "随机的序列得分应该趋于0"

    def test_NormalizedWeightedInverseOrderCriterion():
        correct_order = [3, 4, 1, 0, 2]
        input_order1 = [0, 2, 3, 4, 1]

        criterion = NormalizedWeightedInverseOrderCriterion(correct_order)
        criterion_test = NormalizedWeightedInverseOrderCriterion([0, 1, 2])

        help_func = lambda position, distance: np.log(np.e + distance) / (position + 1)
        assert np.isclose(
            criterion_test.weightedInverseOrderSum,
            help_func(0, 1) + help_func(0, 2) + help_func(1, 2 - 1),
        ), "算法求权部分出错"

        correct_result = (
            help_func(0, 3 - 0)
            + help_func(0, 4 - 0)
            + help_func(1, 3 - 1)
            + help_func(1, 4 - 1)
            + help_func(2, 3 - 2)
            + help_func(2, 4 - 2)
        ) / criterion.weightedInverseOrderSum

        assert np.isclose(criterion(input_order1), correct_result), "算法求和部分错误"

    """验证评价准则"""

    # 验证AccumulatedNormalizedConnectivityCriterion
    test_AccumulatedNormalizedConnectivityCriterion()
    # 验证CorrelationCriterion
    test_CorrelationCriterion()
    # 验证NormalizedWeightedInverseOrderCriterion
    test_NormalizedWeightedInverseOrderCriterion()
