from pathlib import Path
import random
import string
import datetime

import pytz
import numpy as np
import torch
from torch.nn import Module
from torch.cuda import is_available as is_cuda_available
from torch.backends.mps import is_built as is_mps_built
from torch.utils.tensorboard import SummaryWriter


# NOTE: pyg中很多功能mps未支持，不推荐使用
# DEVICE = torch.device("cuda" if is_cuda_available() else "mps" if is_mps_built() else "cpu")
DEVICE = torch.device("cuda" if is_cuda_available() else "cpu")


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def create_writer(experiment_name: str, targetDir: Path = Path("runs")):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory."""

    # Get timestamp of current date in reverse order
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

    # Create log directory path
    logDir = targetDir / timestamp / experiment_name
    print(f"[INFO] Created SummaryWriter saving to {logDir}")
    return SummaryWriter(log_dir=logDir)


def generate_model_filename(model: Module, label: str, length: int = 8):
    # 生成随机英文字母序列
    def random_string(length):
        letters = string.ascii_letters
        return "".join(random.choice(letters) for _ in range(length))

    # 当前时刻的字符串格式(北京时间)
    beijing_tz = pytz.timezone("Asia/Shanghai")
    time_str = datetime.datetime.now(beijing_tz).strftime("%Y-%m-%d-%H-%M")

    # 生成随机字符串
    random_letters = random_string(length)

    # 创建文件名
    filename = f"{model.__class__.__name__}_{label}_{time_str}_{random_letters}.pth"

    return filename


if __name__ == "__main__":
    import re
    from torch.nn import Linear

    assert (
        re.match(
            r"^Linear_Pretrained_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_[a-zA-Z]{5}.pth$",
            generate_model_filename(Linear(10, 1), "Pretrained", 5),
        )
        is not None
    ), "功能实现错误"
