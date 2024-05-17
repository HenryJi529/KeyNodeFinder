import time, datetime
import os
import zipfile
from ftplib import FTP
import platform
from pathlib import Path
from typing import Dict
import inspect

import matplotlib as mpl
import colorama

FTP_SERVER = "ftp.morningstar369.com"
FTP_USERNAME = "ftp"
FTP_PASSWORD = "1234asdw"
FTP_DIR = "KeyNodeFinder"


def set_matplotlib_engine():
    if platform.system() == "Darwin":
        mpl.use("GTK4Agg")


def markers_generator():
    markers = [
        "D",  # Diamond
        "p",  # Pentagon
        "^",  # Triangle Up
        "o",  # Circle
        "P",  # plus(filled)
        "*",  # Star
        "s",  # Square
        "h",  # Hexagon1
        "X",  # x(filled)
        "8",  # Octagon
    ]
    for marker in markers:
        yield marker


def set_proxy():
    # 代理服务器的地址和端口号
    proxy_address = "http://127.0.0.1:7890"

    # 设置http和https的代理
    os.environ["http_proxy"] = proxy_address
    os.environ["https_proxy"] = proxy_address


def colored_print(var, end: str = "\n"):
    formatted_output = (
        colorama.Fore.YELLOW
        + colorama.Style.BRIGHT
        + str(var)
        + colorama.Style.RESET_ALL
    )
    print(formatted_output, end=end)


def timeit(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        duringTime = int(endTime - startTime)
        print(
            colorama.Fore.BLUE
            + colorama.Style.BRIGHT
            + f"[INFO] Time cost: {str(datetime.timedelta(seconds=duringTime))}"
            + colorama.Style.RESET_ALL
        )
        return result

    return wrapper


def get_filenames_from_ftp(
    server: str = FTP_SERVER,
    username: str = FTP_USERNAME,
    password: str = FTP_PASSWORD,
    remote_directory: str = FTP_DIR,
):
    ftp = FTP(server)
    ftp.login(user=username, passwd=password)
    ftp.cwd(remote_directory)

    return list(ftp.nlst())


def download_file_from_ftp(
    local_directory: str,
    filename: str,
    server: str = FTP_SERVER,
    username: str = FTP_USERNAME,
    password: str = FTP_PASSWORD,
    remote_directory: str = FTP_DIR,
):
    ftp = FTP(server)
    ftp.login(user=username, passwd=password)
    ftp.cwd(remote_directory)

    with open(Path(local_directory) / filename, "wb") as file:
        ftp.retrbinary("RETR " + filename, file.write)

    ftp.quit()


def upload_file_to_ftp(
    local_directory: str,
    filename: str,
    server: str = FTP_SERVER,
    username: str = FTP_USERNAME,
    password: str = FTP_PASSWORD,
    remote_directory: str = FTP_DIR,
):
    ftp = FTP(server)
    ftp.login(username, password)
    ftp.cwd(remote_directory)

    with open(Path(local_directory) / filename, "rb") as file:
        ftp.storbinary("STOR " + filename, file)

    ftp.quit()


def zip_directory(directory_path: str, zip_file_path: str):
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))


def unzip_file(directory_path: str, zip_file_path: str, clean: bool = True):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(directory_path)
    if clean:
        os.remove(zip_file_path)


def capitalize_string(string: str):
    # 将字符串转换为首字母大写
    string = string.capitalize()
    strlist = list(string)

    # 使用正则表达式查找数字后的第一个字母，并将其替换为大写
    for i in range(0, len(strlist) - 1):
        if strlist[i].isdigit():
            if strlist[i + 1].isalpha():
                strlist[i + 1] = strlist[i + 1].upper()
    return "".join(strlist)


def snake_to_lower_camel(snake_case: str):
    words = snake_case.split("_")
    camel_case = words[0] + "".join(word.capitalize() for word in words[1:])
    return camel_case


def get_stringified_params(params: Dict):
    stringified_params = ""
    for key, value in params.items():
        if type(value) == list and type(value[0]) == int:
            value_str = "|".join([str(item) for item in value])
        else:
            value_str = str(value)
        stringified_params += f"{snake_to_lower_camel(key)}{value_str}_"
    return stringified_params[:-1]


def get_required_params(class_obj):
    # 使用inspect模块来获取构造函数的信息
    constructor = inspect.signature(class_obj.__init__)

    # 获取构造函数参数名
    params = constructor.parameters

    # 过滤掉第一个"self"参数（通常是实例本身）
    required_params = [param for param in params.values() if param.name != "self"]

    return [param.name for param in required_params]


if __name__ == "__main__":
    upload_file_to_ftp(Path(__file__).parent, Path(__file__).name)
    assert capitalize_string("f1score") == "F1Score", "功能实现错误"
    assert get_stringified_params({"cool_nums": 6}) == "coolNums6", "功能实现错误"

    class TestClass:
        def __init__(self, name, id):
            self.name = name

    assert get_required_params(TestClass) == ["name", "id"], "功能实现错误"
