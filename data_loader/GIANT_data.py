# -*- coding: utf-8 -*-
"""
Load different datasets for GIANT.

Sometimes, the output phrase contains words not in inputs.
The reasons are the following:
1. segmentation inconsistency
2. wrong characters in user input queries
3. wrong labeled phrase
"""
import json
import codecs
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from datetime import datetime
from collections import defaultdict
import re
import random
from .config import *
from util.file_utils import load, save
from util.dict_utils import counter2ordered_dict
from common.constants import STOPWORDS, PUNCTUATIONS, FIGURE_PATH, OUTPUT_PATH, SYNONYM_DICT, CONCEPT_PATTERN_DICT, SPECIAL_WORDS, PATTERN_WORDS  # SPECIAL_WORDS
import os
from nltk.parse.corenlp import CoreNLPDependencyParser
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from .GIANT_data_utils import char2cid, get_embedding, from_networkx
from torch_geometric.data import Data  # , DataLoader


DEP_PARSER = CoreNLPDependencyParser(url='http://localhost:9005')


def cover_count(title, entitydict):
    allvalue = 0
    for token, value in entitydict.items():
        if token in title:
            allvalue += value
    return allvalue


def select_sub_titles(title_candi, wordset):
    title_score = {}
    for title in title_candi:
        subline = re.split(r'[?!/,\(\)_:\-【】\[\]—！，\|。、？： 丨]+', title)
        goodtitle = ''
        for sub in subline:
            if len(sub) in range(6, 20):
                goodtitle = sub
                break
        if len(goodtitle) < 1:
            continue
        title_score[goodtitle] = cover_count(goodtitle, wordset)
    sort_titles = sorted(title_score.items(), key=lambda a: a[1] - len(a[0]), reverse=True)
    return sort_titles


def get_candidate_events(e):
    querys = e['queries']
    word_pv = defaultdict(int)
    for token in querys:
        for tk in token:
            word_pv[tk] += 1
    news_titles = []
    for i in range(min(len(e['titles']), len(e["is_title_matches_events"]))):
        if e["is_title_matches_events"][i] == "1":
            news_titles.append(e['titles'][i])
    subtitle = select_sub_titles(news_titles, word_pv)
    selected_subtitles = [a[0] for a in subtitle]
    e['candidate_phrases'] = [sub_t for sub_t in selected_subtitles if sub_t != ""]
    return e


def get_words_from_tag_result(tag_result):
    words_set = []
    word_tag_map = {}
    if tag_result is not None and tag_result != "":
        words_set = [normalize_text(word_tag.split("/")[0]) for word_tag in tag_result.split()]
        for word_tag in tag_result.split():
            split_wt = normalize_text(word_tag).rstrip().split("/")
            word = split_wt[0]
            tag = split_wt[1] if len(split_wt) == 2 else "non"
            word_tag_map[word] = tag
    return words_set, word_tag_map


def _unify_segment(tag_result, words_set, word_tag_map):
    if tag_result is None or tag_result.rstrip() == "":
        return tag_result

    finished = False
    tag_result_split = tag_result.split()
    while not finished:
        tag_result_split_new = []
        for word_tag in tag_result_split:
            word = normalize_text(word_tag.split("/")[0])
            tag = word_tag.split("/")[1] if len(word_tag.split("/")) == 2 else "non"
            splitted = False
            split_result = []
            for w in words_set:
                if len(w) >= 2 and w != word and w in word:
                    start = word.find(w)
                    end = start + len(w)
                    w_pre = word[0:start]
                    w_aft = word[end:]
                    split_result = []
                    if w_pre != "":
                        tag_pre = tag
                        if w_pre in word_tag_map:
                            tag_pre = word_tag_map[w_pre]
                        split_result.append(w_pre + "/" + tag_pre)
                    if w != "":
                        tag_mid = tag
                        if w in word_tag_map:
                            tag_mid = word_tag_map[w]
                        split_result.append(w + "/" + tag_mid)
                    if w_aft != "":
                        tag_aft = tag
                        if w_aft in word_tag_map:
                            tag_aft = word_tag_map[w_aft]
                        split_result.append(w_aft + "/" + tag_aft)
                    splitted = True
                    break
            if splitted:
                tag_result_split_new += split_result
            else:
                tag_result_split_new += [word_tag]

        tag_result_split = tag_result_split_new

        if len(tag_result_split_new) == len(tag_result_split):
            finished = True

    output = " ".join(tag_result_split)
    return output


def unify_word_segments(e):
    words_set, word_tag_map = get_words_from_tag_result(e["phrase_tagged"])
    for t in e["titles_tagged"]:
        new_words_set, new_word_tag_map = get_words_from_tag_result(t)
        words_set += new_words_set
        word_tag_map.update(new_word_tag_map)
    for q in e["queries_tagged"]:
        new_words_set, new_word_tag_map = get_words_from_tag_result(q)
        words_set += new_words_set
        word_tag_map.update(new_word_tag_map)
    words_set = list(set(words_set))

    e["phrase_tagged"] = _unify_segment(e["phrase_tagged"], words_set, word_tag_map)
    e["titles_tagged"] = [_unify_segment(t_tagged, words_set, word_tag_map) for t_tagged in e["titles_tagged"]]
    e["queries_tagged"] = [_unify_segment(q_tagged, words_set, word_tag_map) for q_tagged in e["queries_tagged"]]

    return e


def get_cnpts_via_ptn(ptn_dict, line):
    line = line.strip()
    cnpts = []
    for ptn in ptn_dict:
        m = ptn.search(line)
        if not m:
            continue
        if m.group(1).rstrip() is not "":
            cnpts.append(m.group(1))
    return cnpts


def get_cnpt_via_align(l1, l2):
    res = []
    start_token = l1[0]
    if start_token not in l2:
        return ""
    end_token = l1[-1]
    if end_token not in l2:
        return ""
    start_pos = l2.index(start_token)
    end_pos = l2.index(end_token)

    if end_pos > start_pos:
        res = l2[start_pos:end_pos + 1]
    cnpt = "".join(res)
    return cnpt


def del_slot(ptn_dict, token):
    for ptn in ptn_dict:
        m = ptn.search(token)
        if not m:
            continue
        return m.group(1)
    return token


def get_candidate_cnpts(e, ptn_dict):
    qtext = ""
    ttext = ""
    cnpts_by_align = []
    for q in e["queries_tagged"]:
        q = " ".join([x for x in q.split() if "/" in x])
        qsegs = [x for x, y in [x.split('/') for x in q.split()] if y not in ['w']]
        qtext += "".join(qsegs)
        for t in e["titles_tagged"]:
            t = " ".join([x for x in t.split() if "/" in x])
            tsegs = [x for x, y in [x.split('/') for x in t.split()] if y not in ['w']]
            ttext += "".join(qsegs)
            cnpt = get_cnpt_via_align(qsegs, tsegs)
            cnpt = del_slot(ptn_dict, cnpt)
            if cnpt.rstrip() is not "":
                cnpts_by_align.append(cnpt)
    cnpts_by_align = list(set(cnpts_by_align))

    # apply pattern matching to strings
    cnpts_by_ptn = []
    for text in [qtext, ttext]:
        cnpts = get_cnpts_via_ptn(ptn_dict, text)
        cnpts_by_ptn.extend(cnpts)
    cnpts_by_ptn = list(set(cnpts_by_ptn))

    cnpts = list(set(cnpts_by_ptn).union(set(cnpts_by_align)))
    e["candidate_phrases"] = cnpts
    e["candidate_phrases_by_pattern"] = cnpts_by_ptn
    e["candidate_phrases_by_align"] = cnpts_by_align
    # print("DEBUG: candidate_phrases:  ", e['candidate_phrases'])
    return e


def get_candidate_phrases(e):
    if e["phrase_type"] == "event":
        e = get_candidate_events(e)
    elif e["phrase_type"] == "concept":
        e = get_candidate_cnpts(e, CONCEPT_PATTERN_DICT)
    else:
        print("ERROR: type must be event or concept!")
    return e


def get_cleaned_qts(config, e):
    #! NOTICE: we only do the following operations for phrase mining task
    # so when it is multi task, or event element extraction task, we do not clean qt.
    # Because it may remove important element words.
    if "event" not in config.tasks:
        # print("old titles tagged: ", e["titles_tagged"])
        # print(e["candidate_phrases_concat"])
        cleaned_titles_tagged = []
        for t in e["titles_tagged"]:  # remove noisy words in titles by candidate_phrases.
            # print("before clean: ", t)
            t_split = t.split()
            new_t_split = []
            for token in t_split:
                token_w = token.split("/")[0]
                if token_w in e["candidate_phrases_concat"] or len(set(token_w).intersection(set(e["candidate_phrases_concat"]))) > 2:
                    new_t_split.append(token)
            new_t = " ".join(new_t_split)
            # print("after clean: ", new_t)
            cleaned_titles_tagged.append(new_t)
        e["titles_tagged"] = cleaned_titles_tagged
        # print("new titles tagged: ", e["titles_tagged"])
    return e


def keep_word(word, e):
    if word in e["candidate_phrases_concat"] or len(set(word).intersection(set(e["candidate_phrases_concat"]))) > 2:
        return True
    return False


def plot_graph(G, output):
    dot_file = output + ".dot"
    fig_file = output + ".png"
    write_dot(G, dot_file)
    command = 'dot -Tpng ' + dot_file + ' -Grankdir=LR > ' + fig_file
    os.system(command)


def normalize_text(text):
    """
    Replace some special characters in text.
    """
    # NOTICE: don't change the text length.
    # Otherwise, the answer position is changed.
    text = text.replace(",", "，")  # NOTICE: this solves the problem in plot_graph. When "," contained in words, it cannot work.
    return text


def get_raw_examples(config, filename, debug=False, debug_length=20):
    """
    Get a list of raw examples given input event filename.
    """
    print("Start get raw examples ...")
    start = datetime.now()
    raw_examples = []
    with open(filename, 'r') as fp:
        events = json.load(fp)
        for i in range(len(events)):
            e = events[str(i)]
            e = unify_word_segments(e)  # Unify the segments helps
            e = get_candidate_phrases(e)  # For events, we select sub titles; for concepts, we perform pattern matching and QT-align.
            if len(e["candidate_phrases"]) > 0:
                e["candidate_phrases_concat"] = " ".join(e["candidate_phrases"])
            else:
                e["candidate_phrases_concat"] = " ".join(e["titles"]) + " " + " ".join(e["queries"])
            # if config.use_clean_qt:
            #     e = get_cleaned_qts(config, e)#!!! we can also filter word by some property here
            raw_examples.append(e)
            if debug and len(raw_examples) >= debug_length:
                break
    print(("Time of get raw examples: {}").format(datetime.now() - start))
    print("Number of raw examples: ", len(raw_examples))
    return raw_examples


def filter_example(config, example):
    """
    Whether filter a given example according to configure.
    :param config: config contains parameters for filtering example
    :param example: an example instance
    :param mode: "train" or "test", they differs in filter restrictions
    :return: boolean
    """
    # TODO: add some filter criteria here
    return False


def get_filtered_examples(config, examples):
    """
    Get a list of filtered examples according to configure.
    """
    print("Numer of unfiltered examples: ", len(examples))
    filtered_examples = []
    for e in examples:
        if not filter_example(config, e):
            filtered_examples.append(e)
    print("Numer of filtered examples: ", len(filtered_examples))
    return filtered_examples


def text2features(config, text, tagged_text=None, candidate_phrases_concat=None):
    """
    Given input text (with word segment), get dependency parsing result and word tagging result.
    NOTICE: this function can add more features.
    """
    if text is None or text == "":  # NOTICE: in events.json, a few phrases or titles are ""
        print("text is none or empty!")
        result = {"text": text, "word": [], "tag": [], "dep": [], "is_digit": [], "is_stop": [], "is_special": []}
        return result

    result = {"text": normalize_text(text).rstrip()}

    # word, tag
    if tagged_text is not None:  # like this format: "浓眉哥/10 直播/n 剃/v 眉毛/n"
        tag_result = tagged_text.rstrip()
    else:
        tag_result = None
    if tag_result is not None:
        result_word_list = [normalize_text(word_tag.split("/")[0]) for word_tag in tag_result.split()]
        result_tag_list = [
            word_tag.split("/")[1]
            if len(word_tag.split("/")) == 2 else "non"
            for word_tag in tag_result.split()]  # NOTICE: here sometimes we have bad case like "tf boys/10007"
        result["word"] = result_word_list
        result["tag"] = result_tag_list

    # dep
    parses = DEP_PARSER.parse(result_word_list)
    dep_result = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
    result["dep"] = dep_result

    # is digit
    result["is_digit"] = [float(w.isdigit()) for w in result["word"]]

    # is punct
    result["is_punct"] = [float(w in PUNCTUATIONS) for w in result["word"]]

    # is stop
    result["is_stop"] = [float(w in STOPWORDS) for w in result["word"]]

    # is special
    def check_is_special(w):
        if w.rstrip() in PATTERN_WORDS:
            return 0.0
        if w.rstrip() in candidate_phrases_concat:
            return 1.0
        return 0.0

    result["is_special"] = [float(w in SPECIAL_WORDS) for w in result["word"]]
    if config.indicate_candidate:
        result["is_special"] = [float(check_is_special(w)) for w in result["word"]]  # notice: we revised is_special as whether it is candidate words

    # word length
    result["word_len"] = [len(w) for w in result["word"]]

    return result


def src_contains_tgt(src, tgt):
    if src != tgt and tgt in src:
        return True
    return False


def src_tgt_is_synonym(src, tgt, synonyms_dict):
    if src in synonyms_dict and tgt in synonyms_dict[src]:
        return True
    if tgt in synonyms_dict and src in synonyms_dict[tgt]:
        return True
    return False


def build_linguistic_features(config, e):
    """
    Given an example, we get its features / tags, and ids.
    NOTICE: here we can add key element extraction labels for event extraction
    """
    # feature settings
    # if "query" in e:  # NOTICE: the event dataset doesn't have "query" column
    e["queries_features"] = []
    for i in range(len(e["queries"])):
        e["queries_features"].append(text2features(config, e["queries"][i], e["queries_tagged"][i], e["candidate_phrases_concat"]))

    e["titles_features"] = []
    for i in range(len(e["titles"])):
        e["titles_features"].append(text2features(config, e["titles"][i], e["titles_tagged"][i], e["candidate_phrases_concat"]))

    e["phrase_features"] = text2features(config, e["phrase"], e["phrase_tagged"], e["candidate_phrases_concat"])
    return e


def get_featured_examples(config, examples):
    """
    Given spaCy processed examples, we further get featured examples
    using different functions to get different features.
    """
    print("Get featured examples...")
    total = 0
    total_ = 0
    examples_with_features = []
    for example in tqdm(examples):
        total_ += 1
        if filter_example(config, example):
            continue
        total += 1

        example = build_linguistic_features(config, example)
        examples_with_features.append(example)

    print("Built {} / {} instances of features in total".format(total, total_))
    return examples_with_features


def get_counters(examples, tags, not_count_tags):
    """
    Given a list of tags, such as ["word", "char", "ner", ...],
    return a dictionary of counters: tag -> Counter instance.
    """
    # init counters
    counters = {}
    for tag in tags:
        counters[tag] = Counter()
    for tag in not_count_tags:
        counters[tag] = Counter()
        for val in not_count_tags[tag]:
            counters[tag][val] += 1e30

    # update counters
    for e in examples:
        for tag in tags:
            for q_features in e["queries_features"]:
                if tag in q_features:
                    for val in q_features[tag]:
                        counters[tag][val] += 1
            for t_features in e["titles_features"]:
                if tag in t_features:
                    for val in t_features[tag]:
                        counters[tag][val] += 1
    return counters


def create_graph(tagged_query_titles_sample, edge_types_list=["seq", "dep", "contain", "synonym"], emb_dicts=None, config=None):
    # create graph
    G = nx.MultiDiGraph()

    # get data type: "event" or "concept"
    data_type = tagged_query_titles_sample["phrase_type"]

    # output words list
    output_sets = {
        "phrase": tagged_query_titles_sample["phrase_features"]["word"]
    }

    # create nodes: sos and eos
    sos_node = (
        "<sos>",  # node name
        {"id": 0,  # node features
         "tag": "sos",
         "fillcolor": "black",
         "fontcolor": "white",
         "style": "filled",
         "shape": "circle",
         "count_query": 0,
         "count_title": 0,
         "word": "<sos>",
         "is_digit": 0,
         "is_punct": 0,
         "is_stop": 0,
         "is_special": 0,
         "word_len": 0.1})
    eos_node = (
        "<eos>",
        {"id": -1,
         "tag": "eos",
         "fillcolor": "black",
         "fontcolor": "white",
         "style": "filled",
         "shape": "circle",
         "count_query": 0,
         "count_title": 0,
         "word": "<eos>",
         "is_digit": 0,
         "is_punct": 0,
         "is_stop": 0,
         "is_special": 0,
         "word_len": 0.1})

    # add embedding ids into node features dict
    # for example: get "word_id" by "word"
    if emb_dicts is not None:
        for tag in emb_dicts:
            if tag in sos_node[1]:
                new_tag = tag + "_id"  # word_id, ...., see emb_tags in config.
                sos_node[1][new_tag] = char2cid(sos_node[1][tag], emb_dicts[tag], OOV="<oov>")
                eos_node[1][new_tag] = char2cid(eos_node[1][tag], emb_dicts[tag], OOV="<oov>")

    # add y labels into node features dict. TODO: maybe add more y labels here
    sos_node[1]["y_phrase"] = 0  # whether node in output phrase. This is used for phrase generation.
    eos_node[1]["y_phrase"] = 0
    sos_node[1]["y_node_type"] = 0  # what kind of node it is. {0: sos and eos, normal node, 1: trigger, 2: entity, 3: location}
    eos_node[1]["y_node_type"] = 0  # this is used for event elements extraction

    # add sos and eos node to graph
    G.add_nodes_from([sos_node])
    G.add_nodes_from([eos_node])

    # node feature: position id. It indicates the sequence position.
    # this is also useful for merging the same words in titles and queries into one node.
    word_id_map = {"<sos>": 0, "<eos>": -1}

    # calculate node count features (number of show times in query and in titles)
    word_count_query = {"<sos>": 0, "<eos>": 0}  # node feature: how many times this word show in query
    word_count_title = {"<sos>": 0, "<eos>": 0}  # node feature: how many times this word show in titles

    query_word_list = []
    for q_features in tagged_query_titles_sample["queries_features"]:
        query_word_list.extend(q_features["word"])
    for word in query_word_list:
        if word in word_count_query:
            word_count_query[word] += 1
        else:
            word_count_query[word] = 1

    title_word_list = []
    for t_features in tagged_query_titles_sample["titles_features"]:
        title_word_list.extend(t_features["word"])
    for word in title_word_list:
        if word in word_count_title:
            word_count_title[word] += 1
        else:
            word_count_title[word] = 1

    # sub function to construct sub graph
    # text_features are query features or title features.
    # we can use this function to merge queries and titles into the final graph for each example.
    def _add_subgraph(G, text_features, seq_id, output_sets=None):
        cur_id = 0  # position id in sequence
        pre_node = "<sos>"

        for i in range(len(text_features["word"])):  # text_features["word"] is the list of segmented words in current seq (query or title)
            # get node features
            word = text_features["word"][i].rstrip()
            if word == "":
                continue

            if config.use_clean_qt and not keep_word(word, tagged_query_titles_sample):
                continue

            if word in word_id_map:  # a word already shows before
                cur_id += 1
                word_id = word_id_map[word]
            else:  # a new word
                cur_id += 1
                word_id = cur_id + 0
                word_id_map[word] = cur_id + 0

            if word in word_count_query and word in word_count_title:
                fillcolor = "purple"
                count_query = word_count_query[word]
                count_title = word_count_title[word]
            elif word in word_count_query:
                fillcolor = "red"
                count_query = word_count_query[word]
                count_title = 0
            else:
                fillcolor = "blue"
                count_query = 0
                count_title = word_count_title[word]

            # add position, tag, count_query, count_title, binary features.
            # node_label = word
            # if word == ",":
            #     node_label = "<,>"  # NOTICE: this is used to solve a bug in plot_graph function.
            node = (
                word,
                {"id": word_id,  # word position
                 "tag": text_features["tag"][i],  # word pos/ner tag by Tencent tool
                 "fillcolor": fillcolor,
                 "fontcolor": "white",
                 "style": "filled",
                 "shape": "circle",
                 "count_query": count_query,  # number of times shown in query
                 "count_title": count_title,  # number of times shown in title
                 "word": text_features["word"][i],  # word text
                 "is_digit": int(text_features["is_digit"][i]),  # whether it is a digit
                 "is_punct": int(text_features["is_punct"][i]),  # whether it is a punct
                 "is_stop": int(text_features["is_stop"][i]),  # whether it is a stop word
                 "is_special": int(text_features["is_special"][i]),  # whether it is a special word
                 "word_len": text_features["word_len"][i]
                 })

            # add embedding ids features
            if emb_dicts is not None:
                for tag in emb_dicts:
                    if tag in node[1]:
                        new_tag = tag + "_id"
                        node[1][new_tag] = char2cid(node[1][tag], emb_dicts[tag], OOV="<oov>")

            # add y labels
            # TODO: handle segmentation not consistant problem: such as 浓眉 哥 but answer is 浓眉哥， or inverse.
            node[1]["y_phrase"] = int(word in output_sets["phrase"])  # this y indicates whether a node shows in output phrase
            if data_type != "event":
                node[1]["y_node_type"] = 0
            else:  # NOTICE: in the following, we don't consider the importance level in our dataset. of course, we can consider it and change the condition.
                if word in tagged_query_titles_sample["triggers"]:
                    node[1]["y_node_type"] = 1  # type 1 is trigger word nodes
                elif word in tagged_query_titles_sample["entities"] or \
                        word in tagged_query_titles_sample["important_non_entities"]:
                    node[1]["y_node_type"] = 2  # type 2 is entity word nodes or important noun nodes
                elif word in tagged_query_titles_sample["locations"]:
                    node[1]["y_node_type"] = 3  # type 3 is location word nodes
                else:
                    node[1]["y_node_type"] = 0  # type 0 is all other normal nodes

            G.add_nodes_from([node])

            # add sequence edges
            if "seq" in edge_types_list and not G.has_edge(pre_node, word):
                G.add_edges_from([(pre_node, word, {"label": "s", "edge_type": "s", "color": "blue", "seq_id": seq_id})])  # NOTICE: use "edge_type" as key helps graphviz use dot plot edge label.
            if "req" in edge_types_list and not G.has_edge(word, pre_node):
                G.add_edges_from([(word, pre_node, {"label": "r", "edge_type": "r", "color": "red", "seq_id": seq_id})])  # NOTICE: use "edge_type" as key helps graphviz use dot plot edge label.
            pre_node = word

        if "seq" in edge_types_list and not G.has_edge(pre_node, "<eos>"):
            G.add_edges_from([(pre_node, "<eos>", {"label": "s", "edge_type": "s", "color": "blue", "seq_id": seq_id})])
        if "req" in edge_types_list and not G.has_edge("<eos>", pre_node):
            G.add_edges_from([("<eos>", pre_node, {"label": "r", "edge_type": "r", "color": "red", "seq_id": seq_id})])

        # add dependency edges
        if "dep" in edge_types_list:
            if len(text_features["dep"]) > 0:  # for example, event data does't have query. So the length is 0.
                input_dep = text_features["dep"][0]
                for dep in input_dep:
                    source = dep[0][0]
                    target = dep[2][0]
                    if (source not in text_features["word"]) or \
                            (target not in text_features["word"]) or \
                            source.rstrip() == "" or target.rstrip() == "":
                        continue  # NOTICE: in this case, the dep parser split words into more fine-grained words.
                    dep_type = "_".join(dep[1].split(":"))  # this replace : with _ for dot file.
                    if not G.has_edge(source, target):
                        if (config.use_clean_qt and keep_word(source, tagged_query_titles_sample) and keep_word(target, tagged_query_titles_sample)) or not config.use_clean_qt:
                            G.add_edges_from([(source, target, {"label": dep_type, "edge_type": dep_type, "color": "black", "seq_id": seq_id})])
                            G.add_edges_from([(target, source, {"label": "r_" + dep_type, "edge_type": "r_" + dep_type, "color": "black", "seq_id": seq_id})])
        return G

    # handle queries
    seq_id = 0
    for q in tagged_query_titles_sample["queries_features"]:
        G = _add_subgraph(G, q, seq_id, output_sets)
    seq_id += 1

    # handle titles
    t_idx = 0
    for t in tagged_query_titles_sample["titles_features"]:
        if data_type == "event":
            if len(tagged_query_titles_sample["is_title_matches_events"]) > t_idx and tagged_query_titles_sample["is_title_matches_events"][t_idx] == "1":
                    G = _add_subgraph(G, t, seq_id, output_sets)
        if data_type != "event":
            G = _add_subgraph(G, t, seq_id, output_sets)
        t_idx += 1
        seq_id += 1

    # NOTICE: maybe add more types of edges: similar words, synonyms
    if "contain" in edge_types_list or "synonym" in edge_types_list:
        all_node_labels = list(G.nodes)
        num_nodes = len(all_node_labels)
        for node_i in range(num_nodes):
            src = all_node_labels[node_i]
            for node_j in range(node_i, num_nodes):
                tgt = all_node_labels[node_j]
                if "contain" in edge_types_list:
                    if src_contains_tgt(src, tgt):
                        if (config.use_clean_qt and keep_word(src, tagged_query_titles_sample) and keep_word(tgt, tagged_query_titles_sample)) or not config.use_clean_qt:
                            G.add_edges_from([(src, tgt, {"label": "contain", "edge_type": "contain", "color": "green", "seq_id": 0})])
                            G.add_edges_from([(tgt, src, {"label": "contained", "edge_type": "contained", "color": "green", "seq_id": 0})])
                    if src_contains_tgt(tgt, src):
                        if (config.use_clean_qt and keep_word(src, tagged_query_titles_sample) and keep_word(tgt, tagged_query_titles_sample)) or not config.use_clean_qt:
                            G.add_edges_from([(src, tgt, {"label": "contained", "edge_type": "contained", "color": "green", "seq_id": 0})])
                            G.add_edges_from([(tgt, src, {"label": "contain", "edge_type": "contain", "color": "green", "seq_id": 0})])
                if "synonym" in edge_types_list:
                    if src_tgt_is_synonym(src, tgt, synonyms_dict=SYNONYM_DICT):
                        if (config.use_clean_qt and keep_word(src, tagged_query_titles_sample) and keep_word(tgt, tagged_query_titles_sample)) or not config.use_clean_qt:
                            G.add_edges_from([(src, tgt, {"label": "synonym", "edge_type": "synonym", "color": "yellow", "seq_id": 0})])
                            G.add_edges_from([(tgt, src, {"label": "synonym", "edge_type": "synonym", "color": "yellow", "seq_id": 0})])
    # TODO: maybe add global feature: whether this graph is an event.  binary label for whole graph.
    return G


def get_graph_examples(config, examples, edge_types_list=["seq", "dep", "contain", "synonym"],
                       emb_dicts=None,
                       edge_types2ids={},
                       update_edge_types2ids=True):
    """
    Given spaCy processed examples, we further get featured examples
    using different functions to get different features.
    """
    print("Get graph examples...")
    examples_with_graphs = []
    i = 0
    edge_types = []
    for example in tqdm(examples):
        example["G"] = create_graph(example, edge_types_list, emb_dicts, config=config)
        if len(example["G"].nodes) <= 2 or len(example["G"].edges) == 0:  # only have <sos> and <eos> node
            print("Skip empty graph.")
            continue
        example["G_for_decode"] = create_graph(example, ["seq"], emb_dicts=None, config=config)  # for ATSP decoder.
        examples_with_graphs.append(example)
        if config.debug:
            plot_graph(example["G"], FIGURE_PATH + str(i))
            plot_graph(example["G_for_decode"], FIGURE_PATH + str(i) + "_for_decode")
        example["G_data"] = from_networkx(example["G"])
        edge_types.extend(example["G_data"].edge_type)
        i += 1
    print("Get {} examples with graph.".format(i))

    edge_types = list(set(edge_types))
    edge_types.sort()

    # update edge_types2ids
    if update_edge_types2ids:
        eid = len(edge_types2ids)
        for e_type in edge_types:
            if e_type not in edge_types2ids:  # in this way, we can update it every time we use function get_graph_examples
                edge_types2ids[e_type] = eid
                eid += 1

    # print("edge_types2ids is: ", edge_types2ids)
    for example in tqdm(examples_with_graphs):
        example["G_data"].edge_type_id = torch.tensor(
            [edge_types2ids[e_type] for e_type in example["G_data"].edge_type])  # NOTICE: here we haven't consider e_type not in edge_types2ids
        example["G_data"].num_relations = len(edge_types)
    # print("""example["G_data"].edge_type: """, example["G_data"].edge_type)
    # print("""example["G_data"].edge_type_id: """, example["G_data"].edge_type_id)

    print("Finished get graph examples.")
    num_relations = len(edge_types)
    return examples_with_graphs, num_relations, edge_types2ids


def prepro(config):
    emb_tags = config.emb_tags
    emb_config = config.emb_config
    emb_mats = {}
    emb_dicts = {}

    debug = config.debug
    debug_length = config.debug_batchnum * config.batch_size

    # get examples and counters
    if not config.processed_example_features:
        examples = get_raw_examples(config, config.train_file, debug, debug_length)
        examples = get_featured_examples(config, examples)
        counters = get_counters(examples, config.emb_tags, config.emb_not_count_tags)

        save(config.train_examples_file, (examples, 0), message="examples")
        save(config.counters_file, counters, message="counters")
    else:
        examples, num_relations = load(config.train_examples_file)
        counters = load(config.counters_file)

    # get emb_mats and emb_dicts
    if not config.processed_emb:
        for tag in emb_tags:
            emb_mats[tag], emb_dicts[tag] = get_embedding(
                counters[tag], tag,
                emb_file=emb_config[tag]["emb_file"],
                size=emb_config[tag]["emb_size"],
                vec_size=emb_config[tag]["emb_dim"])
        save(config.emb_mats_file, emb_mats, message="embedding mats")
        save(config.emb_dicts_file, emb_dicts, message="embedding dicts")
    else:
        emb_mats = load(config.emb_mats_file)
        emb_dicts = load(config.emb_dicts_file)
    for k in emb_dicts:
        print("Embedding dict length: " + k + " " + str(len(emb_dicts[k])))

    if not config.processed_example_graph_features:
        # NOTICE: we should set update_edge_types2ids = True only for train dataset
        #if config.processed_emb and "edge_types" in emb_dicts:
        #    edge_types2ids = emb_dicts["edge_types"]
        #else:
        edge_types2ids = {}
        examples, num_relations, edge_types2ids = get_graph_examples(
            config, examples, config.edge_types_list, emb_dicts, edge_types2ids, update_edge_types2ids=True)
        emb_dicts["edge_types"] = edge_types2ids
        save(config.train_examples_file, (examples, num_relations), message="examples")
        save(config.emb_dicts_file, emb_dicts, message="embedding dicts")

    # print to txt to debug
    for k in emb_dicts:
        write_dict(emb_dicts[k], OUTPUT_PATH + "debug/emb_dicts_" + str(k) + ".txt")
    for k in counters:
        write_counter(counters[k], OUTPUT_PATH + "debug/counters_" + str(k) + ".txt")
    write_example(examples[5], OUTPUT_PATH + "debug/example.txt")


def write_example(e, filename):
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        for k in e:
            if (isinstance(e[k], np.ndarray) or
                    isinstance(e[k], list) or
                    isinstance(e[k], int) or
                    isinstance(e[k], float) or
                    isinstance(e[k], str)):
                fh.write(str(k) + "\n")
                fh.write(str(e[k]) + "\n\n")


def write_dict(d, filename):
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        for k in d:
            fh.write(str(k) + " " + str(d[k]) + "\n")


def write_counter(c, filename):
    ordered_c = counter2ordered_dict(c)
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        for k in ordered_c:
            fh.write(str(k) + " " + str(ordered_c[k]) + "\n")


def write_2d_list(list_2d, filename):
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        fh.writelines('\t'.join(str(j) for j in i) + '\n' for i in list_2d)


def get_loader(examples_file, batch_size, shuffle=False, debug=False, debug_length=20):
    examples, num_relations = load(examples_file)
    # print("num_relations: ", num_relations)
    data_list = []
    feature_dim = None

    num_e = 0
    for e in examples:
        num_e += 1
        feature_list = [
            e["G_data"].count_query,
            e["G_data"].count_title,
            e["G_data"].is_digit,
            e["G_data"].is_punct,
            e["G_data"].is_stop,
            e["G_data"].is_special,
            [word_len / 10.0 for word_len in e["G_data"].word_len],  # normalize word length
            [id_val / 20.0 for id_val in e["G_data"].id]  # normalize word position
        ]
        if feature_dim is None:
            feature_dim = len(feature_list)
        emb_ids_dict = {
            "word_id": torch.LongTensor(e["G_data"].word_id).unsqueeze(0),
            "tag_id": torch.LongTensor(e["G_data"].tag_id).unsqueeze(0),
            "is_digit_id": torch.LongTensor(e["G_data"].is_digit_id).unsqueeze(0),
            "is_punct_id": torch.LongTensor(e["G_data"].is_punct_id).unsqueeze(0),
            "is_stop_id": torch.LongTensor(e["G_data"].is_stop_id).unsqueeze(0),
            "is_special_id": torch.LongTensor(e["G_data"].is_special_id).unsqueeze(0)
        }
        x = torch.FloatTensor(feature_list).t().unsqueeze(0).contiguous()  # 1 * num_nodes * num_features
        edge_index = e["G_data"].edge_index
        edge_type = e["G_data"].edge_type_id
        #print("DEBUG node_idx: ", e["G_data"].node_index)
        #print("DEBUG edge_idx:  ", edge_index)
        #print("DEBUG edge_type: ", edge_type)
        y = torch.LongTensor(e["G_data"].y_phrase)
        y_node_type = torch.LongTensor(e["G_data"].y_node_type)
        words = e["G_data"].word
        data_list.append(Data(
            x=x, edge_type=edge_type, edge_index=edge_index, y=y, words=words,
            y_node_type=y_node_type, emb_ids_dict=emb_ids_dict,
            G_for_decode=e["G_for_decode"],
            queries_features=e["queries_features"],
            titles_features=e["titles_features"],
            phrase_features=e["phrase_features"]))
        if shuffle:
            random.shuffle(data_list)
        if debug and num_e >= debug_length:
            break

    return data_list, num_relations, feature_dim
