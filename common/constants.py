import os
import torch
import codecs
import re
# import gensim


print("Start loading constants ...")

# data path
current_path = os.getcwd().split("/")

DATA_PATH = "/".join(current_path[:-4]) + "/Datasets/"
PROJECT_PATH = "/".join(current_path[:-4]) + "/GIANT/"  # remind haojie
CODE_PATH = PROJECT_PATH + "src/model/GIANT/"
OUTPUT_PATH = PROJECT_PATH + "output/"
CHECKPOINT_PATH = PROJECT_PATH + "output/checkpoint/"
FIGURE_PATH = PROJECT_PATH + "output/figure/"
LOG_PATH = PROJECT_PATH + "output/log/"
PKL_PATH = PROJECT_PATH + "output/pkl/"
RESULT_PATH = PROJECT_PATH + "output/result/"

STOPWORDS_FILE_PATH = DATA_PATH + "original/function-words/stopwords_zh.txt"
PUNCTUATIONS_FILE_PATH = DATA_PATH + "original/function-words/punctuation.txt"

AILAB_W2V_TXT_PATH = DATA_PATH + "original/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"


# stop words
f_stopwords = open(STOPWORDS_FILE_PATH, "r", encoding="utf-8")
stopwords = f_stopwords.readlines()
STOPWORDS = list(set([word.rstrip() for word in stopwords]))


# punctuations
f_punctuations = open(PUNCTUATIONS_FILE_PATH, "r", encoding="utf-8")
punctuations = f_punctuations.readlines()
PUNCTUATIONS = list(set([word.rstrip() for word in punctuations]))


# special tokens
SPECIAL_TOKENS = {"pad": "<pad>", "oov": "<oov>", "sos": "<sos>", "eos": "<eos>"}
SPECIAL_TOKEN2ID = {"<pad>": 0, "<oov>": 1, "<sos>": 2, "<eos>": 3}


# word vector
# AILAB_W2V = gensim.models.KeyedVectors.load_word2vec_format(AILAB_W2V_BIN_PATH, binary=True)

# synonyms dict
SYNONYM_FILE = DATA_PATH + "original/synoanto/hanyu.synonym"
SYNONYM_DICT = {}
with codecs.open(SYNONYM_FILE, "r", encoding="utf-8") as fs:
    for line in fs:
        if line.rstrip() != "":
            line_split = line.split("`")
            key = line_split[0].rstrip()
            vals = [val[1:].rstrip() for val in line_split[1:]]
            SYNONYM_DICT[key] = vals
fs.close()


# special words which indicates strong signals about concepts around it.
SPECIAL_WORDS = [
    "十大", "盘点", "哪些", "大全", "汇总", "大盘点", "有哪些", "什么", "有什么", "排名", "排行", "排行榜"]
PATTERN_WORDS = [
    "十大", "盘点", "哪些", "大全", "汇总", "大盘点", "有哪些", "什么", "有什么",
    "排名", "排行", "排行榜", "人气",
    "最终", "性价比", "竞争力", "操控测试", "测试", "合规",
    "票房", "票房总", "战斗力", "吸引力", "魅力", "综合能力", "能力", "热度", "", " "]

CONCEPT_PATTERN_DICT = {}
for line in open(PROJECT_PATH + "tool/concept_ptn.txt"):
    token = line.strip()
    ptn = re.compile(token)
    CONCEPT_PATTERN_DICT[ptn] = token

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Finished loading constants ...")
