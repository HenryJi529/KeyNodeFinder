from sklearn.manifold import TSNE
import torch
from torch import Tensor
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
import scienceplots


def view_featureVectors(featureVectors: Tensor, ranks: Tensor):
    """featureVectors与ranks可视化"""
    featureVectors = featureVectors.detach().cpu().numpy()
    ranks = ranks.detach().cpu().numpy()

    if featureVectors.shape[1] > 2:
        perplexity = 5 if featureVectors.shape[0] < 30 else 30
        coordinateFeatureVectors = TSNE(
            n_components=2, perplexity=perplexity
        ).fit_transform(featureVectors)
    else:
        coordinateFeatureVectors = featureVectors
    with plt.style.context(["science", "no-latex"]):
        fig = plt.figure()
        scatter = plt.scatter(
            coordinateFeatureVectors[:, 0],
            coordinateFeatureVectors[:, 1],
            c=ranks,
            cmap=plt.cm.cool,
        )
        fig.colorbar(scatter)
        plt.xticks([])  # 移除x轴刻度
        plt.yticks([])  # 移除y轴刻度

        fig, axes = plt.subplots(featureVectors.shape[0], 1)
        for ind, ax in enumerate(axes):
            ax.imshow([featureVectors[ind, :]], aspect="auto", cmap="pink")
            ax.axis("off")  # 移除坐标轴
            ax.set_xticks([])  # 移除x轴刻度
            ax.set_yticks([])  # 移除y轴刻度
            ax.text(
                -1,
                0,
                f"rank: {ranks[ind]}",
                fontsize=12,
                color="black",
                ha="left",
                va="center",
            )


def view_confusionMatrix(confusionMatrix: Tensor):
    plot_confusion_matrix(
        conf_mat=confusionMatrix.cpu().numpy(),
        class_names=["ge", "lt"],
        figsize=(10, 10),
    )
    plt.title("Confusion Matrix")


if __name__ == "__main__":
    from torchmetrics import ConfusionMatrix

    def test_view_confusionMatrix():
        target = torch.tensor([0, 1, 0, 1, 0, 1])
        preds = torch.tensor([1, 1, 0, 1, 1, 1])
        confusionMatrix = ConfusionMatrix(task="binary")
        view_confusionMatrix(confusionMatrix(preds, target))

    def test_view_featureVectors():
        featureVectors = torch.rand(10, 10)
        ranks = torch.range(0, 9).to(dtype=int)[torch.randperm(10)]
        view_featureVectors(featureVectors, ranks)

    test_view_confusionMatrix()
    test_view_featureVectors()
    plt.show()
