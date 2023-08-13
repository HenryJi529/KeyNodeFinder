from typing import Dict
from pathlib import Path

from torchinfo import summary
from torch.nn import Module
from torch import save, load
import torch
from torch_geometric.data import Data, Batch

from utils.common import upload_file_to_ftp, download_file_from_ftp, get_required_params
from utils.toolbox import FeatureBuilder
from model_builder import NiceModel
from utils.experiment import DEVICE
from utils.visualization import view_featureVectors


def get_modelParamDict_example():
    return {
        "input_features": FeatureBuilder.FEATURE_NUM,
        "output_features": 10,
        "embedding_units": (2, 10),
        "ranking_units": 5,
        "valuing_units": [10, 5],
    }


def get_model_summary(model: Module, depth: int = 2):
    return summary(
        model,
        col_names=[
            "num_params",
            "params_percent",
            "trainable",
        ],
        depth=depth,
        col_width=18,
        row_settings=["var_names"],
        verbose=0,
    )


def get_embedding_from_data(data: Data, model: NiceModel):
    single_data_batch: Batch = Batch.from_data_list([data])
    embeddedVectors = model.embed(single_data_batch.x, single_data_batch.edge_index)
    return embeddedVectors


def view_embedding(data: Data, model: NiceModel):
    embedded_vectors = get_embedding_from_data(data, model)
    view_featureVectors(embedded_vectors, data.y)


def save_model_info(
    model: Module,
    hyperparameters: Dict,
    evaluateResult: Dict,
    targetDir: Path,
    modelFilename: str,
):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        hyperparameters: A dictionary of hyperparameters used to train the model.
        evaluateResult: A dictionary of evaluation result from the model.
        targetDir: A directory for saving the model to.
        modelFilename: A filename for the saved model.
            Should include either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    targetDir.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert modelFilename.endswith(".pth") or modelFilename.endswith(
        ".pt"
    ), "modelFilename should end with '.pt' or '.pth'"
    model_save_path = targetDir / modelFilename

    # Save model
    print(f"[INFO] Saving model to: {model_save_path}")
    obj = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": hyperparameters,
        "evaluateResult": evaluateResult,
    }
    save(obj=obj, f=model_save_path)

    # Upload model
    print(f"[INFO] Uploading model to FTP")
    upload_file_to_ftp(targetDir, modelFilename)


def load_model_info(targetDir: Path, modelFilename: str, device: torch.device = DEVICE):
    # Download model info
    print(f"[INFO] Downloading model to local")
    download_file_from_ftp(targetDir, modelFilename)

    # Load model info
    info = load(f=targetDir / modelFilename, map_location=device)
    model_state_dict = info["model_state_dict"]
    hyperparameters = info["hyperparameters"]
    evaluateResult = info["evaluateResult"]

    # Create model
    modelParamNameList = get_required_params(NiceModel)
    modelParamDict = {"input_features": FeatureBuilder.FEATURE_NUM}
    for paramName in modelParamNameList:
        if not modelParamDict.get(paramName):
            modelParamDict[paramName] = hyperparameters[paramName]
    model = NiceModel(**modelParamDict)

    # Load state dict
    model.load_state_dict(model_state_dict)

    return model, hyperparameters, evaluateResult


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from data_processing import DatasetLoader
    from utils.toolbox import ProblemType

    def test_get_model_summary():
        model = NiceModel(**get_modelParamDict_example())
        summary = get_model_summary(model)
        print(summary)

    def test_view_embedding(instanceNum: int = 5):
        dataset = DatasetLoader.load_synthetic_dataset(
            f"SyntheticDataset-N{instanceNum}", problemType=ProblemType.CN
        )
        data = dataset[0]
        model = NiceModel(**get_modelParamDict_example())
        view_embedding(data, model)

    test_get_model_summary()
    test_view_embedding()
    plt.show()
