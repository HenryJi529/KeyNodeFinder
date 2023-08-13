from pathlib import Path

from utils.common import get_filenames_from_ftp
from model_handler import load_model_info


class ModelAnalyser:
    def __init__(self, targetDir: Path = Path(__file__).parent / "models") -> None:
        model_filenames = [
            filename for filename in get_filenames_from_ftp() if "Model" in filename
        ]

        self.modelDict = {}
        self.hyperparametersDict = {}
        self.evaluateResultDict = {}
        for model_filename in model_filenames:
            model, hyperparameters, evaluateResult = load_model_info(
                targetDir=targetDir,
                modelFilename=model_filename,
            )

            self.modelDict[model_filename] = model
            self.hyperparametersDict[model_filename] = hyperparameters
            self.evaluateResultDict[model_filename] = evaluateResult

    def __iter__(self):
        def generator():
            for model_filename in self.modelDict.keys():
                yield {
                    "name": model_filename,
                    "hyperparameters": self.hyperparametersDict[model_filename],
                    "evaluateResult": self.evaluateResultDict[model_filename],
                }

        return generator()


if __name__ == "__main__":
    analyser = ModelAnalyser()
    for model_info in analyser:
        print(model_info)
        print("=" * 90)
