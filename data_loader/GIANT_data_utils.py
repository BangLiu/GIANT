import numpy as np
from tqdm import tqdm
from util.dict_utils import counter2ordered_dict
import platform
from ctypes import *
import torch
import networkx as nx
import torch_geometric


# the max buff len
max_len = 8192 * 10


def lib_suffix():
    return '.dylib' if platform.system() == 'Darwin' else '.so'


class NERecognition:
    NE_CLOSE = 0x00000001  # 关闭所有词典
    NE_DICT = 0x00000002  # 开启词典匹配逻辑，辅助识别人名、地名、机构名
    NE_REGEX = 0x00000004  # 开启正则匹配识别
    NE_BASIC = 0x10000000  # 开启人名、地名、机构名识别
    NE_ARTICLE = 0x20000000  # 开启作品类实体识别
    NE_DATETIME = 0x01000000  # 开启日期识别
    NE_CURRENCY = 0x02000000  # 开启日期和货币识别

    def __init__(self, so_path, res_path):
        libtagger = CDLL(so_path)
        libtagger.ner_model.restype = c_void_p
        self.model_ = libtagger.ner_model(res_path)
        self.tagger_ = libtagger.ner_tagger
        self.destroy_ = libtagger.ner_destroy

    def recognition(self, sentence, mode=NE_DATETIME | NE_BASIC | NE_ARTICLE, keep_space=False):
        buff = b' ' * max_len
        buff_size = c_ulong(max_len)
        self.tagger_(c_void_p(self.model_), sentence.strip(), c_int32(mode), c_int(keep_space),
                     buff, byref(buff_size))
        buff = buff[:buff_size.value]
        return buff

    def destroy(self):
        self.destroy_(c_void_p(self.model_))


def word2wid(word, word2id_dict, OOV="<oov>"):
    """
    Transform single word to word index.
    :param word: a word
    :param word2id_dict: a dict map words to indexes
    :param OOV: a token that represents Out-of-Vocabulary words
    :return: int index of the word
    """
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2id_dict:
            return word2id_dict[each]
    return word2id_dict[OOV]


def char2cid(char, char2id_dict, OOV="<oov>"):
    """
    Transform single character to character index.
    :param char: a character
    :param char2id_dict: a dict map characters to indexes
    :param OOV: a token that represents Out-of-Vocabulary characters
    :return: int index of the character
    """
    if char in char2id_dict:
        return char2id_dict[char]
    return char2id_dict[OOV]


def get_embedding(counter, data_type,
                  emb_file=None, size=None, vec_size=None,
                  limit=-1, specials=["<pad>", "<oov>", "<sos>", "<eos>"]):
    """
    Get embedding matrix and dict that maps tokens to indexes.
    :param counter: a Counter object that counts different tokens
    :param data_type: a string name of data type
    :param emb_file: file of embeddings, such as Glove file
    :param size: number of different tokens
    :param vec_size: dimension of embedding vectors
    :param limit: filter low frequency tokens with freq < limit
    :param specials: list of special tokens
    :return: emb_mat
                 2D list of embedding matrix
             token2idx_dict
                 dict that maps tokens to indexes
    NOTICE: here we put <pad> at index 0 is the best. Otherwise, some other
    code may have problem.
    """
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}

    # get sorted word counter: ordered dict
    counter = counter2ordered_dict(counter)
    filtered_elements = [k for k, v in counter.items()
                         if (v > limit and k not in specials)]

    # get embedding_dict: (word, vec) pairs
    current_word_idx = 0
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                if current_word_idx == size - len(specials):
                    break
                array = line.split()
                if len(array) < vec_size:
                    continue
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in filtered_elements:
                    embedding_dict[word] = vector
                    current_word_idx += 1
    else:
        assert vec_size is not None
        for token in filtered_elements:
            if size is not None and current_word_idx == size - len(specials):
                break
            embedding_dict[token] = [
                np.random.normal(scale=0.1) for _ in range(vec_size)]
            current_word_idx += 1

    # get token2idx_dict: (word, index) pairs
    token2idx_dict = {}
    nid = 0
    for token in counter:
        if token in embedding_dict:
            token2idx_dict[token] = nid + len(specials)
            nid += 1
    for i in range(len(specials)):
        token2idx_dict[specials[i]] = i
        embedding_dict[specials[i]] = [0. for _ in range(vec_size)]

    # get idx2emb_dict: (index, vec) pairs
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}

    # get emb_mat according to idx2emb_dict: num_words x emb_dim
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    print("{} / {} tokens have corresponding {} embedding vector".format(
        len(embedding_dict), len(filtered_elements) + len(specials),
        data_type))
    return emb_mat, token2idx_dict


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph or networkx.MultiDiGraph): A networkx graph.
    """

    G = G.to_directed() if not nx.is_directed(G) else G

    # get node to id map
    nodes = G.nodes
    node_index = {}
    i = 0
    for k in nodes:
        node_index[k] = i
        i += 1

    # get edge index
    edge_index = []
    # print(G.edges)
    for e in G.edges:
        edge_index.append([node_index[e[0]], node_index[e[1]]])
    edge_index = torch.tensor(edge_index).t().contiguous()

    # get node and edge features
    keys = []
    keys += list(list(G.nodes(data=True))[0][1].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}

    for _, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    # for key, item in data.items():
    #     data[key] = torch.tensor(item)

    data['edge_index'] = edge_index
    data['node_index'] = node_index
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
