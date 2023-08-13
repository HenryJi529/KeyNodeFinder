"""
真实数据集预处理
"""

import pickle
import datetime
from typing import Dict
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import networkx as nx

from utils.common import timeit, colored_print, download_file_from_ftp, unzip_file
from utils.toolbox import REAL_DATA_PATH, GraphTool


class DoubanPreprocessor:
    RAW_FILENAME = "DoubanDataset-raw.zip"

    def __init__(
        self,
        year_range: tuple[int, int] = (2000, 2020),
        root_path: Path = REAL_DATA_PATH / "douban",
    ):
        self.year_range = year_range
        self.root_path = root_path
        self.raw_path = self.root_path / "raw"
        self.processed_path = self.root_path / "processed"
        self.movie_data_path = self.raw_path / "movie.csv"
        self.person_data_path = self.raw_path / "person.csv"

        self.root_path.mkdir(exist_ok=True)
        self.raw_path.mkdir(exist_ok=True)
        self.processed_path.mkdir(exist_ok=True)

        """原始数据下载"""
        if self.movie_data_path.exists() and self.person_data_path.exists():
            pass
        else:
            colored_print("原始数据下载中...")
            download_file_from_ftp(self.raw_path, self.RAW_FILENAME)
            unzip_file(str(self.raw_path), str(self.raw_path / self.RAW_FILENAME))
        colored_print("原始数据已就绪")

        """数据预处理"""
        # NOTE: 通过data.pkl是否存在判断是否要重新预处理
        if (self.processed_path / Path("data.pkl")).exists():
            pass
        else:
            colored_print("启动数据预处理...")
            self.__initialize()
        colored_print("数据预处理完成")

    def read_processed_data(self):
        path = self.processed_path / Path("data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def __handle_genre(self, genreAlias):
        genreMap = {
            "家庭": ["家庭"],
            "爱情": ["爱情", "愛情 Romance"],
            "歌舞": ["歌舞"],
            "古装": ["古装"],
            "音乐": ["音乐"],
            "同性": ["同性"],
            "传记": ["传记", "傳記 Biography"],
            "情色": ["情色", "Adult"],
            "恐怖": ["惊栗", "惊悚", "驚悚 Thriller", "恐怖", "鬼怪"],
            "武侠": ["武侠"],
            "冒险": [
                "冒险",
            ],
            "奇幻": ["奇幻"],
            "悬疑": ["悬念", "悬疑"],
            "战争": ["战争"],
            "灾难": ["灾难"],
            "运动": ["运动"],
            "动作": ["动作"],
            "科幻": ["科幻"],
            "黑色电影": ["黑色电影"],
            "喜剧": [
                "喜剧",
                "Comedy",
                "喜劇 Comedy",
            ],
            "脱口秀": ["脱口秀"],
            "儿童": [
                "儿童",
                "兒童 Kids",
            ],
            "动画": ["动画"],
            "剧情": [
                "剧情",
                "劇情 Drama",
            ],
            "西部": ["西部"],
            "纪录片": ["纪录片", "紀錄片 Documentary"],
            "舞台艺术": ["舞台艺术", "戏曲"],
            "真人秀": ["真人秀", "Reality-TV"],
            "历史": ["历史"],
            "犯罪": ["犯罪"],
            "荒诞": ["荒诞"],
            "短片": ["短片"],
            "其他": [],
        }
        for realName, aliasList in genreMap.items():
            if genreAlias in aliasList:
                return realName
        return "其他"

    def __handle_language(self, languageAlias):
        """
        语言可以按照语种进行分类，常见的语种分类包括：
        1. 汉语系语言：包括汉语、粤语、闽南语、客家语等。
        2. 英语系语言：包括英语、爱尔兰语、苏格兰语、威尔士语等。
        3. 法语系语言：包括法语、西班牙语、葡萄牙语、意大利语等。
        4. 德语系语言：包括德语、荷兰语、丹麦语、挪威语等。
        5. 斯拉夫语系语言：包括俄语、乌克兰语、波兰语、捷克语等。
        6. 日语系语言：包括日语、朝鲜语等。
        7. 阿拉伯语系语言：包括阿拉伯语、波斯语等。
        8. 印欧语系语言：包括印度语、梵语、希腊语、拉丁语等。
        9. 非洲语系语言：包括斯瓦希里语、豪萨语、祖鲁语等。
        10. 美洲原住民语系语言：包括纳瓦特尔语、克里语等。
        """
        languageMap = {
            "其他": [
                "拉脱维亚语",
                "斯华西里语",
                "塔吉克语",
                "Afrikaans",
                "捷克",
                "桑海语",
                "Aramaic",
                "无声",
                "苏语",
                "米斯特克语",
                "silent",
                "马拉雅拉姆语 Malayalam",
                "阿拉伯语",
                "波兰语",
                "泰卢固语  Telugu",
                "荷兰语",
                "德清方言",
                "天马电影制片厂",
                "Pashtu",
                "拉达克语",
                "British Sign Language",
                "Tonga (Tonga Islands)",
                "Mongolian",
                "瑞典语",
                "瑞典语  Swedish",
                "西班牙语",
                "科萨语",
                "南非荷兰语",
                "意第绪语",
                "孟加拉语",
                "Persian",
                "河北井陉话",
                "泰固卢语",
                "Zulu",
                "奥地利",
                "吉尔吉斯语",
                "克罗地亚语",
                "北印度语 HIndi",
                "库尔德语 Kurdish",
                "馬拉雅拉姆語 Malayalam",
                "English",
                "宗卡语",
                "Silent",
                "波兰语 Polish",
                "Bulgarian",
                "土尔其语",
                "英文",
                "Swahili",
                "佛兰德斯语",
                "班巴拉语",
                "卢森堡语",
                "默片",
                "湘西方言",
                "依地语",
                "泰语",
                "German",
                "乌克兰语",
                "阿布哈兹语",
                "坎纳达语",
                "罗马尼亚语 Romanian",
                "马来语 Malayalam",
                "法語French",
                "中文",
                "日本語",
                "Assamese",
                "冰岛语 Icelandic",
                "维吾尔语",
                "Serbian",
                "国语",
                "Telegu",
                "老挝语",
                "克里奥尔语",
                "塞尔维亚-克罗地亚语",
                "Wolof",
                "少数民族语",
                "世界语",
                "印度语",
                "孟加拉语 Bengali",
                "Sicilian",
                "武汉话",
                "加泰罗尼亚语",
                "东北方言",
                "泰尔米语",
                "塞尔维亚语",
                "俄罗斯语 Russian",
                "武汉方言",
                "Bhojpuri",
                "西班牙",
                "菲律宾语",
                "Marathi",
                "泰卢固语 Telugu",
                "阿尔巴尼亚语",
                "西班牙語 Spanish",
                "贵州独山话",
                "客家话",
                "土著语",
                "沪语",
                "瓜拉尼语",
                "韃靼方言",
                "门德语",
                "Punjabi",
                "晋语",
                "泰卢固语",
                "苗语",
                "乌兹别克语",
                "芬兰语",
                "Punjab语",
                "纳瓦霍语",
                "印尼语",
                "泰盧固語 Telugu",
                "Min Nan",
                "山西话",
                "越剧",
                "Swedish",
                "Creole",
                "印地语",
                "赛德克语",
                "台湾原住民语言",
                "韩语",
                "美国手语",
                "客家语",
                "罗马尼语",
                "阿塞拜疆语",
                "玛雅语",
                "祖鲁语",
                "瑞典语 Swedish",
                "印地语  Hindi",
                "陕西话",
                "潮州语",
                "Hindi",
                "卡纳达语",
                "盖尔语",
                "Nahuatl",
                "斯洛伐克语",
                "克林贡语",
                "爱沙尼亚语",
                "斯洛伐克语 Slovak",
                "大同方言",
                "印度尼西亚语",
                "塞索托语",
                "Navajo",
                "Ukrainian",
                "Kuna",
                "persian",
                "乌尔都语 Ordu",
                "四川话",
                "Shanghainese",
                "马其顿语",
                "Telugu",
                "南非荷兰语afrikaans",
                "丹麦语",
                "澳洲土著语",
                "印地语 Hindi",
                "维语",
                "波尼族语",
                "斯洛文尼亚语",
                "Panjabi",
                "哈萨克语",
                "越南语",
                "捷克语",
                "高棉语",
                "科西嘉语",
                "Sindarin",
                "Chaozhou",
                "菏泽话",
                "Malayalam",
                "立陶宛语",
                "贵州话",
                "马林凯语",
                "塞尔维亚－克罗地亚语",
                "加利西亚语",
                "Phono-Kinema",
                "北印度语 Hindi",
                "林加拉语",
                "沃洛夫语",
                "Saami",
                "克伦语",
                "俄语 Russia",
                "印地语，乌尔杜语",
                "闽南话",
                "Gujarati",
                "印地語 Hindi",
                "塞尔维亚-克罗埃西亚语",
                "Tamil",
                "缅甸语",
                "冰岛语",
                "印地语 hindi",
                "羌语",
                "拉脱维亚",
                "波斯语",
                "阿萨姆语",
                "塞尔维亚克罗地亚语",
                "古吉拉特语",
                "Spanish",
                "Dari",
                "东北话",
                "俄语Russian",
                "Icelandic",
                "印第安苏族语",
                "刚卡尼语",
                "陕西方言",
                "埃塞俄比亚语",
                "图阿雷格语",
                "北印度语",
                "柏柏尔语",
                "塔加拉族语",
                "廈語",
                "挪威",
                "希腊语",
                "撒丁语",
                "尼泊尔语",
                "粵語",
                "Chinese",
                "俄语",
                "吉普赛语",
                "法语手语",
                "坦米尔语 Tamil",
                "丹麦",
                "Syriac",
                "俄语 Russian",
                "Bengali",
                "闽南语",
                "中国方言",
                "岗德语",
                "因纽特语",
                "印地语 HIndi",
                "俄罗斯 Russian",
                "藏语",
                "锡伯语",
                "波斯语 Persian",
                "泰米尔语 tamil",
                "中文广东话",
                "塔加路族语",
                "希伯来语",
                "古加拉特语",
                "达里语",
                "潮汕方言",
                "无对白",
                "弗拉芒语",
                "塔塔尔语",
                "迪尤拉语 Dyula",
                "他加禄语",
                "台语 Taiwanese",
                "南非语",
                "阿拉伯 Arabic",
                "普通话",
                "Washoe",
                "Albanian",
                "Sioux",
                "印地",
                "Mono",
                "卡拜尔语",
                "湖北方言",
                "朝鲜语",
                "河南话",
                "挪威语",
                "泰米尔 Tamil",
                "阿姆哈拉语",
                "亚美尼亚语",
                "波斯尼亚语",
                "Tamil |Hindi |Telugu",
                "波斯尼亚语 Bosnian",
                "西北方言",
                "胶辽官话",
                "粤语",
                "世界语 Esperanto",
                "Aidoukrou",
                "Apache languages",
                "蒙古语",
                "Belarusian",
                "厦语",
                "韩国语",
                "车臣语",
                "匈牙利语",
                "english",
                "Tupi",
                "上海话",
                "Catalan",
                "斯瓦希里语",
                "大连话",
                "海达语",
                "马来语",
                "罗马尼亚语",
                "拉贾斯坦语",
                "朝鲜手语",
                "保加利亚语",
                "俄罗斯语",
                "台语",
                "Tzotzil",
                "Creek",
                "中英字幕",
                "中文 Chinese",
                "乌尔都语 Urdu",
                "旁遮普语",
                "西西里语",
                "安徽方言",
                "瓦尤语",
                "泰庐固语 Telugu",
                "他加禄语 Tagalog",
                "天津话",
                "乌尔都语",
                "丹麦语 Danish",
                "Armenian",
                "印地语   Hindi",
                "菏兰语",
                "客家話",
                "山西方言",
                "台湾手语",
                "捷克语 Czech",
                "巴西手语",
                "夏安语",
                "Tamil |Telugu |Hindi",
                "蚌埠话",
                "马拉提语",
                "Malay",
                "荷兰语 Dutch",
                "温州方言",
                "瑞士德语",
                "僧伽罗语",
                "格鲁吉亚语",
                "北印度语 HINDI",
                "萨米语",
                "Japanese Sign Language",
                "克什米尔语",
                "那不勒斯语",
                "傈僳语",
                "hindi",
                "阿拉伯语 Arabic",
                "埃纳德语",
                "巴斯克语",
                "泰米尔语 Tamil",
                "Russian",
                "约鲁巴语",
                "閩南語",
                "英国",
                "河南方言",
                "Kabuverdianu",
                "韩国手语",
                "拉丁语",
                "亚美尼加语",
                "越南语 Vietnamese",
                "皮钦语",
                "泰卢固语Telugu",
                "福建话",
                "马拉雅拉姆语",
                "北美印第安语",
                "库尔德语",
                "Hawaiian",
                "Croatian",
                "Esperanto",
                "哈尼语",
                "土耳其语",
                "比印度语 Hindi",
                "德语",
                "意大利",
                "阿尔巴尼亚语 Albanian",
                "阿伊努语",
                "意大利语",
                "匈奴语",
                "印度手语",
                "马尔他语",
                "泰米尔语",
                "重庆方言",
                "Parsee",
                "Klingon",
                "闽南",
                "Tajik",
                "四川方言",
                "阿拉姆语",
                "Inuktitut",
                "康纳达语",
                "奇楚亚语",
                "台語",
                "Gaelic",
                "格陵兰语",
                "普什图语",
                "古英语",
                "Athapascan languages",
                "Galician",
                "克丘亚语",
                "毛利语",
                "芬兰语 Finnish",
                "泰文",
                "印度语 Hindi",
                "Ungwatsi",
                "南京方言",
                "拉丁语 Latin",
                "Alsatian",
                "满族语",
                "满语",
                "塔加路语",
                "西安话",
                "英語 English",
                "Maya",
                "坎纳达语 Kannada",
                "坦纳岛西南部方言",
                "天津方言",
                "加里西亚语",
                "索马里语",
                "波利尼西亚语",
                "Welsh",
            ],
            "汉语": [
                "汉语普通话",
                "东北普通话",
                "汉语方言",
                "汉语东北话",
                "云南方言",
                "北京话",
                "内蒙古方言",
                "湖南方言",
                "青岛话",
                "台湾腔",
                "闽南语 Min Nan",
                "唐山话",
                "潮州话",
                "重庆话",
                "山东话",
                "贵州方言",
                "甘肃方言",
            ],
            "英语": [
                "英語",
                "英语",
                "爱尔兰语",
                "爱尔兰",
                "爱尔兰盖尔语",
                "苏格兰盖立语",
                "苏格兰盖尔语",
                "威尔士语",
            ],
            "法语": [
                "法语",
                "葡萄牙语",
                "French",
                "葡萄牙语 Portuguese",
                "葡萄牙语 Português",
                "葡萄牙语  Portuguese",
                "葡萄牙语 Portugese",
                "葡萄牙",
            ],
            "日语": [
                "日语",
                "日語",
            ],
            "手语": [
                "英语手语",
                "法国手语",
                "日本手语",
            ],
        }
        for realName, aliasList in languageMap.items():
            if languageAlias in aliasList:
                return realName
        return "其他"

    def __read_raw_data(self, ratio: float = None):
        """数据读取"""
        if ratio:
            movie_df = pd.read_csv(
                self.movie_data_path,
                index_col="MOVIE_ID",
                nrows=int(sum(1 for l in open(self.movie_data_path)) * 0.1),
            )
        else:
            movie_df = pd.read_csv(self.movie_data_path, index_col="MOVIE_ID")
        person_df = pd.read_csv(self.person_data_path, index_col="PERSON_ID")

        return movie_df, person_df

    @timeit
    def __preprocess_person(self, person_df):
        """person数据处理"""
        # 保留特定列
        person_df = person_df[["NAME", "SEX", "BIRTH"]]

        # 根据PERSON_ID排序
        person_df = person_df.sort_index()

        # 获取演员的出身年份
        person_df["BIRTH"] = person_df["BIRTH"].fillna("")
        person_df["BIRTH_YEAR"] = person_df["BIRTH"].apply(
            lambda x: x.split("-")[0] if x else None
        )
        person_df = person_df.drop(columns=["BIRTH"])

        # 处理SEX信息
        person_df["SEX"] = person_df["SEX"].apply(lambda x: 1 if x == "男" else 0)
        # 填补SEX中的None
        maleProbability = (
            person_df["SEX"].value_counts()[1] / person_df["SEX"].value_counts().sum()
        )
        person_df["SEX"] = person_df["SEX"].apply(
            lambda x: (
                int(np.random.binomial(n=1, p=maleProbability, size=1))
                if x != 0 and x != 1
                else x
            )
        )

        # 分离NAME信息
        NAME = person_df["NAME"].copy()
        person_df = person_df.drop(columns=["NAME"])

        # 保存其他信息
        personOtherInfo = {"NAME": NAME}

        return person_df, personOtherInfo

    @timeit
    def __preprocess_movie(self, movie_df):
        """movie数据处理"""

        # 保留特定列
        movie_df = movie_df[
            [
                "NAME",
                "GENRES",
                "LANGUAGES",
                "MINS",
                "YEAR",
                "DOUBAN_SCORE",
                "DOUBAN_VOTES",
                "ACTOR_IDS",
                "DIRECTOR_IDS",
            ]
        ]

        # 根据MOVIE_ID排序
        movie_df = movie_df.sort_index()

        # 清理无person的条目
        movie_df = movie_df.dropna(subset=["ACTOR_IDS", "DIRECTOR_IDS"], how="all")
        movie_df = movie_df.fillna("")

        # 获取每部电影相关的person
        movie_df["PERSON_IDS"] = movie_df["ACTOR_IDS"] + "|" + movie_df["DIRECTOR_IDS"]

        # 只保留person的id信息
        def processColumn(columnName):
            # Alternative: movie_df[columnName].map(lambda x: list(filter(None,list(set(x.split("|"))))))
            movie_df[columnName] = movie_df[columnName].str.split("|", expand=False)

            def get_person_id(persons):
                ids = []
                for person in persons:
                    # 确保person的格式为"NAME:ID"(person数据要存在于person_df中)
                    if len(person.split(":")) == 2 and person.split(":")[1] != "":
                        ids.append(int(person.split(":")[1]))
                ids = list(set(ids))
                return ids

            movie_df[columnName] = movie_df[columnName].map(get_person_id)

        for columnName in ["ACTOR_IDS", "DIRECTOR_IDS", "PERSON_IDS"]:
            processColumn(columnName)

        # 去除无合作的movie NOTE: 网络至此全连通
        movie_df = movie_df[movie_df["PERSON_IDS"].map(len) > 1]

        # 处理DOUBAN_VOTES
        movie_df["DOUBAN_VOTES"].apply(lambda x: int(x))

        def remove_element(object_, val):
            try:
                object_.remove(val)
            except ValueError:
                pass
            return object_

        def encode(input_list, class_list):
            # 统计类别数目
            num_classes = len(class_list)
            # 初始化编码结果
            encoded = [0] * num_classes
            # 遍历输入列表，将包含的类别编码为1
            for x in input_list:
                if x in class_list:
                    encoded[class_list.index(x)] = 1
            return encoded

        # 处理GENRES
        movie_df["GENRES"] = movie_df["GENRES"].str.split("/", expand=False)
        movie_df["GENRES"] = movie_df["GENRES"].apply(
            lambda x: remove_element([genre.strip() for genre in x], "")
        )
        movie_df["GENRES"] = movie_df["GENRES"].apply(
            lambda x: [self.__handle_genre(genreAlias) for genreAlias in x]
        )
        GENRE_LIST = list(set(movie_df["GENRES"].sum()))
        movie_df["GENRES"] = movie_df["GENRES"].apply(lambda x: encode(x, GENRE_LIST))

        # 处理LANGUAGES
        movie_df["LANGUAGES"] = movie_df["LANGUAGES"].str.split("/", expand=False)
        movie_df["LANGUAGES"] = movie_df["LANGUAGES"].apply(
            lambda x: remove_element([language.strip() for language in x], "")
        )
        movie_df["LANGUAGES"] = movie_df["LANGUAGES"].apply(
            lambda x: [self.__handle_language(languageAlias) for languageAlias in x]
        )
        LANGUAGE_LIST = list(set(movie_df["LANGUAGES"].sum()))
        movie_df["LANGUAGES"] = movie_df["LANGUAGES"].apply(
            lambda x: encode(x, LANGUAGE_LIST)
        )

        # 分离NAME信息
        NAME = movie_df["NAME"].copy()
        movie_df = movie_df.drop(columns=["NAME"])

        # 保存其他信息
        movieOtherInfo = {
            "NAME": NAME,
            "GENRE_LIST": GENRE_LIST,
            "LANGUAGE_LIST": LANGUAGE_LIST,
        }

        return movie_df, movieOtherInfo

    @timeit
    def __build_graph(self, movie_df):
        graph = nx.Graph()

        print("根据每个电影中的相关人员，逐步添加边")
        for i, (index, row) in tqdm(
            enumerate(movie_df.iterrows()),
            total=len(movie_df),
            desc="\t 进度",
            unit="%",
            ncols=80,
        ):
            person_ids = row["PERSON_IDS"]
            # 遍历当前电影的每个人员ID
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    # 添加这两个人员之间的边
                    if graph.has_edge(person_ids[i], person_ids[j]):
                        # 如果边已经存在，则将其权重加1
                        graph[person_ids[i]][person_ids[j]]["weight"] += 1
                    else:
                        # 如果边不存在，则添加一条权重为1的边
                        graph.add_edge(person_ids[i], person_ids[j], weight=1)

        return graph

    def __save_processed_data(self, data: Dict):
        path = self.processed_path / Path("data.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @timeit
    def __build_featureDataFrame(
        self,
        movie_df: pd.DataFrame,
        person_df: pd.DataFrame,
        movieOtherInfo: Dict,
        personOtherInfo: Dict,
    ):
        def countProfession(row):
            actorIds = row["ACTOR_IDS"]
            directorIds = row["DIRECTOR_IDS"]
            for directorId in directorIds:
                personProfession[directorId][1] += 1
            for actorId in actorIds:
                personProfession[actorId][0] += 1

        # 获取真实的任务ID列表
        print("\t获取真实的任务ID列表")
        personList = []
        movie_df["PERSON_IDS"].apply(lambda ids: personList.extend(ids))
        personList = list(set(personList))

        # 初始化featureDataFrame
        print("\t初始化featureDataFrame")
        featureDataFrame = DataFrame(
            None,
            index=personList,
            columns=[
                "SEX",
                "BIRTH_YEAR",
                "ACTOR_COUNT",
                "DIRECTOR_COUNT",
                "MOVIE_COUNT",
                "MEDIAN_DOUBAN_VOTES",
                "MEDIAN_DOUBAN_SCORE",
                "MEDIAN_MINS",
                "GENRE_COUNT",
                "LANGUAGE_COUNT",
                "MEDIAN_YEAR",
            ],
        )
        # 初始化为列表的特征值
        featureDataFrame["GENRE_COUNT"] = Series(
            data=[
                len(movieOtherInfo["GENRE_LIST"]) * [0]
                for i in range(featureDataFrame.shape[0])
            ],
            index=personList,
        )
        featureDataFrame["LANGUAGE_COUNT"] = Series(
            data=[
                len(movieOtherInfo["LANGUAGE_LIST"]) * [0]
                for i in range(featureDataFrame.shape[0])
            ],
            index=personList,
        )

        # 统计person的职业(NOTE: 相当于处理边的类型)
        print("\t统计person的职业")
        personProfession = {
            personId: [0, 0] for personId in personList
        }  # NOTE: 索引0是演员次数，索引1是导演次数
        movie_df.apply(countProfession, axis=1)

        # 提取person_df的数据
        print("\t提取person_df的数据")
        for i, (index, row) in tqdm(
            enumerate(person_df.iterrows()),
            desc="\t 进度",
            total=len(person_df),
            unit="%",
            ncols=80,
        ):
            if index not in personList:
                pass
            else:

                def get_year(string):
                    if not string:
                        return None
                    try:
                        date = datetime.datetime.strptime(string, "%Y-%m-%d")
                        year = date.year
                    except ValueError:
                        try:
                            date = datetime.datetime.strptime(string, "%Y-%m")
                            year = date.year
                        except ValueError:
                            try:
                                year = int(string)
                            except ValueError:
                                year = None
                    return year

                sex = row["SEX"]
                birthYear = get_year(row["BIRTH_YEAR"])
                featureDataFrame.loc[index, "SEX"] = sex
                featureDataFrame.loc[index, "BIRTH_YEAR"] = birthYear

        # 获取person与movie的反向关系
        print("\t获取person与movie的反向关系")
        personInMovies = {personId: [] for personId in personList}
        for index, row in movie_df.iterrows():
            for personId in row["PERSON_IDS"]:
                personInMovies[personId].append(index)

        # 提取movie_df的数据
        print("\t提取movie_df的数据")
        featureDataFrame["MOVIE_COUNT"] = [
            len(x) for x in personInMovies.values()
        ]  # 获取参与数量
        for i, (index, row) in tqdm(
            enumerate(featureDataFrame.iterrows()),
            total=len(featureDataFrame),
            desc="\t 进度",
            unit="%",
            ncols=80,
        ):
            movieIndexList = personInMovies[index]
            # NOTE: 使用中位数是为了更方便的处理缺失值
            featureDataFrame.loc[index, "MEDIAN_DOUBAN_VOTES"] = movie_df.loc[
                movieIndexList, "DOUBAN_VOTES"
            ].median()
            featureDataFrame.loc[index, "MEDIAN_DOUBAN_SCORE"] = movie_df.loc[
                movieIndexList, "DOUBAN_SCORE"
            ].median()
            featureDataFrame.loc[index, "MEDIAN_MINS"] = movie_df.loc[
                movieIndexList, "MINS"
            ].median()

            genreCount = [
                int(x)
                for x in np.array(
                    list(movie_df.loc[movieIndexList, "GENRES"].values)
                ).sum(axis=0)
            ]
            for ind in range(len(movieOtherInfo["GENRE_LIST"])):
                featureDataFrame.loc[index, "GENRE_COUNT"][ind] = genreCount[ind]

            languageCount = [
                int(x)
                for x in np.array(
                    [
                        np.array(item)
                        for item in movie_df.loc[movieIndexList, "LANGUAGES"].values
                    ]
                ).sum(axis=0)
            ]
            for ind in range(len(movieOtherInfo["LANGUAGE_LIST"])):
                featureDataFrame.loc[index, "LANGUAGE_COUNT"][ind] = languageCount[ind]

            featureDataFrame.loc[index, "MEDIAN_YEAR"] = movie_df.loc[
                movieIndexList, "YEAR"
            ].median()

        # 边特征嵌入
        print("\t边特征嵌入")
        featureDataFrame["ACTOR_COUNT"] = [x[0] for x in personProfession.values()]
        featureDataFrame["DIRECTOR_COUNT"] = [x[1] for x in personProfession.values()]

        # 填补MEDIAN_YEAR
        print("\t填补MEDIAN_YEAR")
        featureDataFrame["MEDIAN_YEAR"] = featureDataFrame["MEDIAN_YEAR"].apply(
            lambda year: (
                np.random.uniform(low=1950, high=2015, size=1)[0]
                if year < 1900.0
                else year
            )
        )

        # 填补SEX中的None NOTE: 这个填补不是因为person_df中的缺失，而是因为person_df中压根没有这个id
        print("\t填补SEX")
        try:
            maleProbability = (
                featureDataFrame["SEX"].value_counts()[1]
                / featureDataFrame["SEX"].value_counts().sum()
            )
        except IndexError:
            maleProbability = 0.5
        featureDataFrame["SEX"] = featureDataFrame["SEX"].apply(
            lambda x: (
                int(np.random.binomial(n=1, p=maleProbability, size=1)[0])
                if x != 0 and x != 1
                else x
            )
        )

        # 填补BIRTH_YEAR中的None
        # NOTE: 根据一些研究和数据，大部分演员的重要作品出现的年纪大约在 30 到 50 岁之间，这个区间可以作为一个大致的参考。
        # 而重要作品的发布年份一般出现在作品年份的中位数附近
        print("\t填补BIRTH_YEAR")
        for index, row in featureDataFrame.loc[
            featureDataFrame["BIRTH_YEAR"].isna()
        ].iterrows():
            if row["MEDIAN_YEAR"] > 0:
                featureDataFrame.loc[index, "BIRTH_YEAR"] = int(
                    row["MEDIAN_YEAR"] - np.random.uniform(low=30, high=50, size=1)[0]
                )

        # 整理dtype
        print("\t整理dtype")
        featureDataFrame["BIRTH_YEAR"] = featureDataFrame["BIRTH_YEAR"].astype("int64")
        featureDataFrame["MEDIAN_DOUBAN_VOTES"] = featureDataFrame[
            "MEDIAN_DOUBAN_VOTES"
        ].astype("float64")
        featureDataFrame["MEDIAN_DOUBAN_SCORE"] = featureDataFrame[
            "MEDIAN_DOUBAN_SCORE"
        ].astype("float64")
        featureDataFrame["MEDIAN_MINS"] = featureDataFrame["MEDIAN_MINS"].astype(
            "float64"
        )

        return featureDataFrame

    @timeit
    def __build_featureArray(self, featureDataFrame, movieOtherInfo):
        featureArray = np.zeros(
            (
                featureDataFrame.shape[0],
                9
                + len(movieOtherInfo["LANGUAGE_LIST"])
                + len(movieOtherInfo["GENRE_LIST"]),
            )
        )

        print("\t按照新索引来生成array")
        for i, (index, value) in tqdm(
            enumerate(featureDataFrame.reset_index().iterrows()),
            total=len(featureDataFrame),
            desc="\t 进度",
            unit="%",
            ncols=80,
        ):
            featureArray[index][0] = value["SEX"]
            featureArray[index][1] = value["BIRTH_YEAR"]
            featureArray[index][2] = value["ACTOR_COUNT"]
            featureArray[index][3] = value["DIRECTOR_COUNT"]
            featureArray[index][4] = value["MOVIE_COUNT"]
            featureArray[index][5] = value["MEDIAN_DOUBAN_VOTES"]
            featureArray[index][6] = value["MEDIAN_DOUBAN_SCORE"]
            featureArray[index][7] = value["MEDIAN_MINS"]
            featureArray[index][8] = value["MEDIAN_YEAR"]
            featureArray[index][9 : 9 + len(movieOtherInfo["GENRE_LIST"])] = value[
                "GENRE_COUNT"
            ]
            featureArray[index][9 + len(movieOtherInfo["GENRE_LIST"]) :] = value[
                "LANGUAGE_COUNT"
            ]

        return featureArray

    @timeit
    def __initialize(self):
        colored_print("读取原始数据...")
        movie_df, person_df = self.__read_raw_data()
        colored_print("处理movie.csv...")
        movie_df, movieOtherInfo = self.__preprocess_movie(movie_df)
        colored_print("处理person.csv...")
        person_df, personOtherInfo = self.__preprocess_person(person_df)

        data = {"items": []}
        data["movieOtherInfo"] = movieOtherInfo
        data["personOtherInfo"] = personOtherInfo
        data["movie_df"] = movie_df
        data["person_df"] = person_df

        for endYear in range(*self.year_range):
            colored_print(f"生成网络数据[endYear: {endYear}]...")
            print("构建featureDataFrame...")
            featureDataFrame = self.__build_featureDataFrame(
                movie_df=movie_df[movie_df["YEAR"] < endYear],
                person_df=person_df,
                movieOtherInfo=movieOtherInfo,
                personOtherInfo=personOtherInfo,
            )
            print("保存featureDataFrame...")
            featureDataFrame.to_csv(
                self.processed_path / f"year{endYear}_feature_dataframe.csv"
            )
            print("构建featureArray...")
            featureArray = self.__build_featureArray(featureDataFrame, movieOtherInfo)
            print("保存featureArray...")
            np.savetxt(
                self.processed_path / f"year{endYear}_feature_array.csv",
                featureArray,
                delimiter=",",
            )
            print("保存graph...")
            graph = self.__build_graph(movie_df=movie_df[movie_df["YEAR"] < endYear])
            print("保存graph...")
            nx.write_weighted_edgelist(
                graph,
                self.processed_path / f"year{endYear}_weighted_edgelist.csv",
                delimiter=",",
            )
            data["items"].append(
                {
                    "endYear": endYear,
                    "featureDataFrame": featureDataFrame,
                    "featureArray": featureArray,
                    "graph": graph,
                }
            )
            colored_print("=" * 90)

        colored_print("保存全部数据...")
        self.__save_processed_data(data)

        colored_print("Done!!!")

    @staticmethod
    def test():
        colored_print("测试豆瓣数据预处理器...")
        douban_preprocessor = DoubanPreprocessor((2000, 2002))
        data = douban_preprocessor.read_processed_data()

        items = data["items"]
        movieOtherInfo = data["movieOtherInfo"]
        personOtherInfo = data["personOtherInfo"]
        movie_df = data["movie_df"]
        endYear, featureDataFrame, featureArray, graph = (
            items[0]["endYear"],
            items[0]["featureDataFrame"],
            items[0]["featureArray"],
            items[0]["graph"],
        )
        print(f"endYear: {endYear}")
        print(f"graph: {graph}")
        print(f"featureDataFrame:\n {featureDataFrame.head()}")


if __name__ == "__main__":
    DoubanPreprocessor()
