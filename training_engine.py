from typing import Dict, Tuple, List, Union
from pathlib import Path
from random import random, shuffle

from tqdm.auto import tqdm
import numpy as np
from torch import optim
from torch.nn import Module
from torch import Tensor
from torch.nn import MSELoss
from torch.nn.functional import sigmoid
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
from torch.optim import Optimizer
import torch
from torchmetrics import Metric
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)
from torch.utils.tensorboard import SummaryWriter

from utils.common import capitalize_string
from utils.toolbox import (
    DataBatchDecomposer,
    RankLossFunction,
    ProblemType,
)
from utils.experiment import DEVICE
from algorithm_modeling import Action, State, Game, get_stateData
from model_builder import NiceModel, Task


class PreTrainer:
    @staticmethod
    def calc_metrics(
        metrics: Dict[str, Metric], concatenated_logits: Tensor, databatch: Batch
    ):
        differencesList = DataBatchDecomposer.get_differencesList_from_databatch(
            databatch
        )
        logitsList = DataBatchDecomposer.get_logitsList_from_databatch(
            concatenated_logits, databatch
        )
        for pred_logits, true_differences in zip(logitsList, differencesList):
            for metricName in metrics:
                pred_probs = sigmoid(pred_logits.squeeze())
                pred_differences = torch.where(
                    pred_probs >= 0.5, torch.tensor(1), torch.tensor(0)
                )
                metrics[metricName](pred_differences, true_differences)

    @staticmethod
    def setup_metrics(
        device: torch.device = DEVICE,
    ) -> Dict[str, Metric]:
        with torch.device(device):
            # Setup Metrics
            metrics = {
                "accuracy": Accuracy(task="binary"),
                "recall": Recall(task="binary"),
                "precision": Precision(task="binary"),
                "f1score": F1Score(task="binary"),
                "confmat": ConfusionMatrix(task="binary"),
            }
        return metrics

    @staticmethod
    def train_step(
        model: Module,
        dataloader: DataLoader,
        lossFn: RankLossFunction,
        optimizer: Optimizer,
        metrics: Dict[str, Metric],
        epoch: int,
        device: torch.device = DEVICE,
    ) -> Tuple[float, Dict[str, Tensor]]:
        """Trains a PyTorch model for a single epoch.
        @param model: (Module) A PyTorch model to be trained.
        @param dataloader: (DataLoader[Graph]) A DataLoader instance for the model to be trained on.
        @param lossFn: (RankLossFunction) A Specific PyTorch loss function to minimize.
        @param optimizer: (Optimizer) A PyTorch optimizer to help minimize the loss function.
        @param metrics: (List[Metric]) A list of metrics.
        @param epoch: (int) A number indicating which epoch is being trained.
        @param device: (device) A target device to compute on (e.g. "cuda", "mps", "cpu").
        @return train_loss, train_metrics: (float, dict) A tuple of training loss and training metrics.
        """
        # 启用模型的train模式
        model.train()

        # 初始化train_loss与train_metrics
        train_loss = 0
        for metricName in metrics:
            metrics[metricName].reset()

        # Loop through data loader data batches
        for databatch in tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Training(Epoch{epoch})",
            leave=False,
        ):
            # Send data to target device
            databatch = databatch.to(device)

            # 1. Forward pass
            concatenated_logits = model(databatch)
            # 2. Calculate  and accumulate loss
            loss = lossFn(concatenated_logits, databatch)
            train_loss += loss.item()
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            # 4. Loss backward
            loss.backward()
            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate metric across all batches
            PreTrainer.calc_metrics(metrics, concatenated_logits, databatch)

        train_loss = train_loss / len(dataloader)
        train_metrics = {}
        for metricName in metrics:
            train_metrics[metricName] = metrics[metricName].compute()

        return train_loss, train_metrics

    @staticmethod
    def val_step(
        model: Module,
        dataloader: DataLoader,
        lossFn: RankLossFunction,
        metrics: Dict[str, Metric],
        epoch: int,
        device: torch.device = DEVICE,
    ) -> Tuple[float, Dict[str, Tensor]]:
        """Validates a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a validation dataset.

        Args:
            model: A PyTorch model to be validated.
            dataloader: A DataLoader instance for the model to be validated on.
            lossFn: A Specific PyTorch loss function to calculate loss on the validation data.
            metrics: A dictionary of metric names to their corresponding metric.
            epoch: A number indicating which epoch is being trained.
            device: A target device to compute on (e.g. "cuda", "mps", "cpu").

        Returns:
            A tuple of validation loss and validation metrics.
            In the form (val_loss, val_metrics).
        """
        # Put model in eval mode
        model.eval()

        # Setup val loss and val metrics values
        val_loss = 0
        for metricName in metrics:
            metrics[metricName].reset()

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for databatch in tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"Validating(Epoch{epoch})",
                leave=False,
            ):
                # Send data to target device
                databatch = databatch.to(device)

                # 1. Forward pass
                concatenated_logits = model(databatch)
                # 2. Calculate and accumulate loss
                loss = lossFn(concatenated_logits, databatch)
                val_loss += loss.item()

                # Calculate and accumulate metric across all batches
                PreTrainer.calc_metrics(metrics, concatenated_logits, databatch)

        val_loss = val_loss / len(dataloader)
        val_metrics = {}
        for metricName in metrics:
            val_metrics[metricName] = metrics[metricName].compute()

        return val_loss, val_metrics

    @staticmethod
    def train(
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        lossFn: RankLossFunction,
        epochNum: int,
        writer: SummaryWriter = None,
        device: torch.device = DEVICE,
        verbose: bool = False,
    ) -> Dict[str, List]:
        """Trains and Validates a PyTorch model.

        Passes a target PyTorch models through train_step() and val_step()
        functions for a number of epochs, training and validating the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Args:
            model: A PyTorch model to be trained and validated.
            train_dataloader: A DataLoader instance for the model to be trained on.
            val_dataloader: A DataLoader instance for the model to be validated on.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            lossFn: A PyTorch loss function to calculate loss on both datasets.
            epochNum: An integer indicating how many epochs to train for.
            writer: A SummaryWriter instance to write training and validation metrics to.
            device: A target device to compute on (e.g. "cuda", "mps", "cpu").
            verbose: A boolean indicating whether to print training and validation metrics.

        Returns:
            A dictionary of training and validating loss as well as training and
            validating metrics. Each metric has a value in a list for
            each epoch.
            In the form:
                        {
                            train_loss: [...],
                            train_metrics: [...],
                            val_loss: [...],
                            val_metrics: [...]
                        }
        """

        metrics = PreTrainer.setup_metrics(device=device)

        # Create empty result dictionary
        result = {
            "train_loss": [],
            "train_metrics": [],
            "val_loss": [],
            "val_metrics": [],
        }

        # Make sure model on target device
        model.to(device)

        # Loop through training and validating steps for a number of epochs
        for epoch in tqdm(range(epochNum), desc=f"Total Epochs", leave=True):
            train_loss, train_metrics = PreTrainer.train_step(
                model=model,
                dataloader=train_dataloader,
                lossFn=lossFn,
                optimizer=optimizer,
                metrics=metrics,
                epoch=epoch,
                device=device,  # NOTE: 实际上这个device是用来迁移数据的
            )
            val_loss, val_metrics = PreTrainer.val_step(
                model=model,
                dataloader=val_dataloader,
                lossFn=lossFn,
                metrics=metrics,
                epoch=epoch,
                device=device,  # NOTE: 实际上这个device是用来迁移数据的
            )

            if verbose:
                print(f"\nEpoch {epoch+1}: ")
                print(
                    f"\tloss => train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
                )
                for metricName in train_metrics:
                    if metricName == "confmat":
                        print(
                            f"\t{metricName} => train_{metricName}: {train_metrics[metricName].view(-1)} | val_{metricName}: {val_metrics[metricName].view(-1)}"
                        )
                    else:
                        print(
                            f"\t{metricName} => train_{metricName}: {train_metrics[metricName]:.4f} | val_{metricName}: {val_metrics[metricName]:.4f}"
                        )

            # Update result dictionary
            result["train_loss"].append(train_loss)
            result["train_metrics"].append(train_metrics)
            result["val_loss"].append(val_loss)
            result["val_metrics"].append(val_metrics)

            ### Experiment tracking ###
            if writer:
                # See SummaryWriter documentation
                writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
                    global_step=epoch,
                )
                for metricName in metrics:
                    if metricName == "confmat":
                        continue
                    writer.add_scalars(
                        main_tag=capitalize_string(metricName),
                        tag_scalar_dict={
                            f"train_{metricName}": train_metrics[metricName],
                            f"val_{metricName}": val_metrics[metricName],
                        },
                        global_step=epoch,
                    )
                # Close the writer
                writer.close()

        # Return the filled result at the end of the epochs
        return result

    @staticmethod
    def evaluate(
        model: Module,
        test_dataloader: DataLoader,
        device: torch.device = DEVICE,
    ):
        """给出模型的最终评价(accuracy, recall, precision, f1score, confusion_matrix)"""

        # Setup Metrics
        metrics = PreTrainer.setup_metrics(device=device)

        # Make sure model on target device
        model.to(device)

        # Put model in eval mode
        model.eval()

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for databatch in test_dataloader:
                # Send data to target device
                databatch = databatch.to(device)

                # Forward pass
                concatenated_logits = model(databatch)

                # Calculate and accumulate metrics
                PreTrainer.calc_metrics(metrics, concatenated_logits, databatch)

        result = {}
        for metricName in metrics:
            result[metricName] = metrics[metricName].compute()

        return result


class Trainer:
    def __init__(
        self,
        model: NiceModel,
        dataloaders: Tuple[DataLoader],
        roundNum: int = 10,
        greedyRate: float = 0.05,
        discountRate: float = 0.99,
        learningRate: float = 1e-3,
        weightDecay: float = 1e-3,
        problemType: ProblemType = ProblemType.CN,
        verbose: bool = False,
    ):
        self.model = model
        if len(dataloaders) != 3:
            raise ValueError("datloaders参数应该由三个Dataloader组成")
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.roundNum = roundNum
        self.greedyRate = greedyRate
        self.discountRate = discountRate
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.problemType = problemType
        self.verbose = verbose

        # 固定embedding_layer
        for param in self.model.embedding_layer.parameters():
            param.requires_grad = False

        # 设置optimizer与lossFn
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay
        )
        self.lossFn = MSELoss()

    def init_game(self, databatch: Batch):
        if len(databatch) == 1:
            # 只包含一个数据
            single_data = databatch[0]
            single_graph = to_networkx(single_data, to_undirected=True)
            self.currentTrainOriginData = single_data
            return Game(
                single_graph,
                roundNum=self.roundNum,
                problemType=self.problemType,
                verbose=self.verbose,
            )
        else:
            # 包含多个数据
            raise ValueError("只支持含有一个data的databatch")

    def train(self):
        for ind, databatch in tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Training",
            leave=True,
        ):
            databatch = databatch.to(self.model.device)
            self.game = self.init_game(databatch)
            self.train_databatch(ind)

    def get_state(self):
        """返回游戏的当前状态"""
        return self.game.round.state

    def get_stateData(self, oldState: State, require_grad: bool = False):
        """获得状态的对应data[修改原始data的edge_index]"""
        stateData = get_stateData(oldState, self.currentTrainOriginData, require_grad)
        return stateData

    def get_action(self, state: State, databatch_index: int) -> Action:
        """根据神经网络，选择最佳的action【要排除已经去掉的action】
        NOTE: tradeoff exploration / exploitation
        NOTE: 当前只支持DISCONNECT动作类型
        """
        actionType = Action.ActionType.DISCONNECT
        if random() < self.greedyRate / np.log(databatch_index + np.e):
            # : 随机数小于greedyRate时, 使用随机生成的action【贪心程度应该从大到小】
            action = state.get_random_action(actionType)
        else:
            # : 随机数不小于greedyRate时, 使用模型选择action
            # 获得当前的currentData
            currentData = self.get_stateData(state)
            # 获得当前state下，每个action的value
            values: Tensor = self.model(
                Batch.from_data_list([currentData]), task=Task.VALUE
            )
            # 选择state下可选的value最高的action
            action = state.get_best_action(values.squeeze(), actionType)
        return action

    def train_step(
        self,
        oldState: State,
        reward: float,
        action: Action,
        newState: State,
    ):
        self.model.train()
        oldData = self.get_stateData(oldState, require_grad=True)
        newData = self.get_stateData(newState, require_grad=True)

        # 此时获得的values是oldState所有action的values
        values: Tensor = self.model(Batch.from_data_list([oldData]), task=Task.VALUE)
        # 找到当前action中targetNodeIndex对应的value，即Q(s,a)的预测值
        pred_Q = values[action.targetNodeIndex].squeeze()
        # Q(s, a)的真实值使用$$Q(s, a)=R(s)+\gamma \max _{a'} Q(s', a')$$
        real_Q = reward + self.discountRate * torch.max(
            self.model(Batch.from_data_list([newData]), task=Task.VALUE)
        )
        loss: Tensor = self.lossFn(pred_Q, real_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_databatch(self, databatch_index: int):
        while True:
            # 获得oldState
            oldState = self.get_state()

            # 根据oldState获得action
            action = self.get_action(oldState, databatch_index)

            # 执行action获得reward, newState等
            reward, roundDone, gameDone, _ = self.game.play_step(action)
            newState = self.get_state()

            # 存储步骤
            self.game.remember(oldState, reward, action, newState)

            if roundDone:
                # 整个memory的数据，进行一次完整训练(本质上还是train_short_menory)
                shuffle(self.game.memory)
                for step in self.game.memory:
                    # 单次训练
                    self.train_step(*step)

            if gameDone:
                # 结束当前的game
                break


if __name__ == "__main__":
    from torch import optim

    from utils.experiment import create_writer
    from model_builder import NiceModel
    from model_handler import get_modelParamDict_example
    from utils.common import colored_print
    from utils.experiment import set_seeds
    from data_processing import DatasetLoader
    from data_loading import create_dataloaders
    from utils.toolbox import RankLossFunction

    def test_pretrain(seed: int = 38, instanceNum: int = 1, maxLength: int = 100):
        set_seeds(seed)
        # 准备数据
        dataset = DatasetLoader.load_synthetic_dataset(
            f"SyntheticDataset-N{instanceNum}", problemType=ProblemType.CN
        )
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            dataset,
            (0.6, 0.3, 0.1),
            max_length=maxLength,
            shuffles=[False, False, False],
            seed=seed,
        )
        # 准备模型
        model = NiceModel(**get_modelParamDict_example())
        # 准备训练
        lossFn = RankLossFunction(verbose=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        writer = create_writer("test_experiment", targetDir=Path("logs"))
        # 训练与评估
        train_result = PreTrainer.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lossFn=lossFn,
            optimizer=optimizer,
            epochNum=5,
            writer=writer,
            verbose=True,
        )
        evaluate_result = PreTrainer.evaluate(model, test_dataloader)
        print(train_result)
        print(evaluate_result)

    def test_train(instanceNum=1, seed=40):
        set_seeds(seed)
        model = NiceModel(**get_modelParamDict_example())
        dataset = DatasetLoader.load_synthetic_dataset(
            f"SyntheticDataset-N{instanceNum}", problemType=ProblemType.CN
        )
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            dataset,
            (0.6, 0.3, 0.1),
            batch_size=1,
            shuffles=[False, False, False],
            seed=seed,
        )
        trainer = Trainer(
            model,
            (train_dataloader, val_dataloader, test_dataloader),
            roundNum=2,
            greedyRate=0.05,
            discountRate=0.99,
            learningRate=1e-3,
            problemType=ProblemType.CN,
            verbose=True,
        )
        trainer.train()

    colored_print("单元测试: pretrain...")
    test_pretrain()
    colored_print("=" * 80)
    colored_print("单元测试: train...")
    test_train()
    colored_print("=" * 80)
