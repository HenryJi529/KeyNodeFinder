# Matpool Setup

1. 依赖安装
    ```bash
    pip uninstall -y torchvision torchaudio
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
    pip install torchmetrics
    pip install torchinfo
    pip install pympler
    pip install -U mlxtend
    pip install torch-tb-profiler
    pip install pyyaml
    pip install colorama
    pip install SciencePlots
    pip install powerlaw
    ```

2. 配置VSCode【需要先连接VSCode】
    ```bash
    code --install-extension ms-python.python
    code --install-extension ms-python.vscode-pylance
    code --install-extension ms-python.black-formatter
    ```

3. 源码同步【本地执行】
    ```bash
    # 打包
    git archive --format=zip --output=release/archive.zip main
    # 传输
    rsync ./release/archive.zip matpool:/mnt/archive.zip
    # 解压
    ssh matpool "unzip -o /mnt/archive.zip -d /root/KeyNodeFinder"
    # AllInOne
    git archive --format=zip --output=release/archive.zip main; rsync ./release/archive.zip matpool:/mnt/archive.zip; ssh matpool "unzip -o /mnt/archive.zip -d /root/KeyNodeFinder"
    ```


