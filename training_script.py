import argparse
from pathlib import Path
from typing import Dict

from torch import optim
import yaml

from data_processing import DatasetLoader
from data_loading import create_dataloaders
from model_handler import save_model_info
from model_builder import NiceModel
from training_engine import PreTrainer, Trainer
from utils.experiment import set_seeds, create_writer, generate_model_filename
from utils.common import (
    colored_print,
    timeit,
    set_proxy,
    get_stringified_params,
    get_required_params,
)
from utils.toolbox import RankLossFunction, ProblemType, FeatureBuilder


def show_args(args):
    for key, value in vars(args).items():
        colored_print(f"{key}={value}", end=" ")
    print()


def get_cleaned_params(args, exclude_params: list[str]) -> Dict:
    cleaned_params = {}
    for key, value in vars(args).items():
        if key in exclude_params or value is None:
            continue
        cleaned_params[key] = value
    return cleaned_params


def get_pretrain_environment_params(args) -> Dict:
    exclude_params = [
        "config",
        "rounds",
        "greedy_rate",
        "discount_rate",
        "learning_rate",
        "environment",
        "verbose",
    ]
    cleaned_params = get_cleaned_params(args, exclude_params=exclude_params)
    return cleaned_params


@timeit
def main(args):
    show_args(args)
    # 设置随机种子
    if args.seed:
        print(f"设置随机种子为: {args.seed}")
        set_seeds(args.seed)
        shuffles = [False, False, False]
    else:
        shuffles = [True, False, False]

    # 设置保存路径与代理
    if args.environment == "local":
        baseDir = Path(".")
        print("[INFO] 启用proxy...")
        set_proxy()
    elif args.environment == "colab":
        baseDir = Path("./drive/MyDrive/KeyNodeFinder")
    elif args.environment == "matpool":
        baseDir = Path(".")
    else:
        raise ValueError(f"environment不支持{args.environment}选项...")

    # 设置问题类型
    try:
        problems = [problemType.name for problemType in list(ProblemType)]
        problemType = list(ProblemType)[problems.index(args.problem)]
    except:
        raise ValueError("不存在的问题类型")

    # 获取合成数据集
    dataset = DatasetLoader.load_synthetic_dataset(
        f"SyntheticDataset-N{args.instances}", problemType=problemType
    )
    print(
        f"节点特征数: {dataset.num_node_features}, 边特征数: {dataset.num_edge_features}"
    )

    """预训练过程"""
    # 加载数据集
    (
        pretrain_train_dataloader,
        pretrain_val_dataloader,
        pretrain_test_dataloader,
    ) = create_dataloaders(
        dataset,
        (0.6, 0.3, 0.1),
        max_length=args.pre_max_length,
        shuffles=shuffles,
        seed=args.seed,
    )
    # 设置TB writer
    pretrainWriter = create_writer(
        experiment_name="pretrain-"
        + get_stringified_params(get_pretrain_environment_params(args)),
        targetDir=baseDir / "logs",
    )
    # 初始化模型
    modelParamNameList = get_required_params(NiceModel)
    modelParamDict = {"input_features": FeatureBuilder.FEATURE_NUM}
    for paramName in modelParamNameList:
        if not modelParamDict.get(paramName):
            modelParamDict[paramName] = getattr(args, paramName)
    model = NiceModel(**modelParamDict)
    # 设置代价函数与优化器
    pretrainlossFn = RankLossFunction(verbose=False)
    pretrainOptimizer = optim.Adam(
        model.parameters(),
        lr=args.pre_learning_rate,
        weight_decay=args.pre_weight_decay,
    )
    # 预训练模型
    _ = PreTrainer.train(
        model=model,
        train_dataloader=pretrain_train_dataloader,
        val_dataloader=pretrain_val_dataloader,
        lossFn=pretrainlossFn,
        optimizer=pretrainOptimizer,
        epochNum=args.pre_epochs,
        writer=pretrainWriter,
        verbose=args.verbose,
    )
    # 预训练评估
    pretrain_evaluate_result = PreTrainer.evaluate(model, pretrain_test_dataloader)
    print(f"预训练模型评价: {pretrain_evaluate_result}")

    """训练过程"""
    # 加载数据集
    (
        train_train_dataloader,
        train_val_dataloader,
        train_test_dataloader,
    ) = create_dataloaders(
        dataset,
        (0.6, 0.3, 0.1),
        batch_size=1,
        shuffles=shuffles,
        seed=args.seed,
    )
    # 训练模型
    trainer = Trainer(
        model=model,
        dataloaders=(
            train_train_dataloader,
            train_val_dataloader,
            train_test_dataloader,
        ),
        roundNum=args.rounds,
        greedyRate=args.greedy_rate,
        discountRate=args.discount_rate,
        learningRate=args.learning_rate,
        weightDecay=args.weight_decay,
        problemType=problemType,
        verbose=args.verbose,
    )
    trainer.train()

    """模型保存"""
    save_model_info(
        model=model,
        hyperparameters=vars(args),
        evaluateResult=pretrain_evaluate_result,
        targetDir=baseDir / "models",
        modelFilename=generate_model_filename(model, label=problemType.name, length=8),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    # 参数传递方式【是否通过config.yaml】
    parser.add_argument(
        "--config",
        action="store_true",
        required=False,
        default=False,
        help="是否通过config.yaml传递参数",
    )
    # 建模相关参数
    parser.add_argument(
        "--problem",
        required=False,
        default="CN",
        choices=["CN", "ND"],
        help="设置problemType",
    )
    # 模型相关参数
    parser.add_argument(
        "--output_features",
        required=False,
        type=int,
        default=10,
        help="设置output_features",
    )
    parser.add_argument(
        "--embedding_units",
        nargs="+",
        type=int,
        default=(2, 10),
        help="设置embedding_units【Tuple: 第一个元素是nums，第二个元素是out_channels】",
    )
    parser.add_argument(
        "--ranking_units",
        required=False,
        type=int,
        default=5,
        help="设置ranking_units",
    )
    parser.add_argument(
        "--valuing_units",
        nargs="+",
        type=int,
        default=[10, 5],
        help="设置valuing_units",
    )
    # 数据相关参数
    parser.add_argument(
        "--instances",
        required=False,
        type=int,
        default=1,
        help="设置instance_num",
    )
    # 训练相关参数
    parser.add_argument(
        "--pre_max_length",
        required=False,
        type=int,
        default=100,
        help="设置pretrain_max_length",
    )
    parser.add_argument(
        "--pre_epochs",
        required=False,
        type=int,
        default=1,
        help="设置pretrain_epoch_num",
    )
    parser.add_argument(
        "--pre_learning_rate",
        required=False,
        type=float,
        default=0.001,
        help="设置pretrain_learning_rate",
    )
    parser.add_argument(
        "--pre_weight_decay",
        required=False,
        type=float,
        default=0.001,
        help="设置pretrain_weight_decay",
    )
    parser.add_argument(
        "--rounds",
        required=False,
        type=int,
        default=1,
        help="设置train_round_num",
    )
    parser.add_argument(
        "--greedy_rate",
        required=False,
        type=float,
        default=0.05,
        help="设置train_greedy_rate",
    )
    parser.add_argument(
        "--discount_rate",
        required=False,
        type=float,
        default=0.99,
        help="设置train_discount_rate",
    )
    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        default=0.001,
        help="设置train_learning_rate",
    )
    parser.add_argument(
        "--weight_decay",
        required=False,
        type=float,
        default=0.001,
        help="设置train_weight_decay",
    )
    # 实验相关参数
    parser.add_argument(
        "--environment",
        choices=["local", "colab", "matpool"],
        required=False,
        default="local",
        help="选择环境(支持local和colab)",
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        help="设置随机种子",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="是否显示全部信息",
    )

    args = parser.parse_args()
    # 加载config.yaml中的参数
    if args.config:
        with open(Path(__file__).parent / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        # 更新命令行参数
        for key, value in config.items():
            # 若value为default则不复盖默认值
            if value == "DEFAULT":
                continue
            setattr(args, key, value)

    main(args)
