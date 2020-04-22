import elkai
import networkx as nx
import numpy as np


def decode(output_words_list, G_for_decode, queries_features, titles_features):
    if "<sos>" in output_words_list:
        output_words_list.remove("<sos>")
    if "<eos>" in output_words_list:
        output_words_list.remove("<eos>")
    ordered_output_words, success = heuristic_decode(output_words_list, queries_features, titles_features)
    if success:
        return ordered_output_words

    ordered_output_words = ATSP_decode(output_words_list, G_for_decode, queries_features, titles_features)
    return ordered_output_words


def heuristic_decode(output_words_list, queries_features, titles_features):
    success = False
    ordered_output_words = []
    num_output_words = len(output_words_list)
    features = queries_features + titles_features

    for feat in features:
        num_words = len(feat["word"])
        if num_words < num_output_words:
            continue

        for i in range(num_words - num_output_words + 1):
            chunk = feat["word"][i:i + num_output_words]
            if set(chunk) == set(output_words_list):  # continuous full matching
                ordered_output_words = chunk
                success = True
                return ordered_output_words, success

    return ordered_output_words, success


def ATSP_decode(output_words_list, G_for_decode, queries_features, titles_features):
    # STEP1: refine graph for decoding
    features = queries_features + titles_features

    # Connect <sos> node with the first word that belongs to output_words_list in each query or title
    for feat in features:
        for w in feat["word"]:
            if w in output_words_list:
                if not G_for_decode.has_edge("<sos>", w):
                    G_for_decode.add_edges_from([("<sos>", w, {"label": "s", "edge_type": "s", "color": "red", "seq_id": 0})])
                break

    # Connect <eos> node with the last word that belongs to output_words_list in each query or title
    for feat in features:
        feat["word"].reverse()
        for w in feat["word"]:
            if w in output_words_list:
                if not G_for_decode.has_edge(w, "<eos>"):
                    G_for_decode.add_edges_from([(w, "<eos>", {"label": "s", "edge_type": "s", "color": "red", "seq_id": 0})])
                break
        feat["word"].reverse()

    # define distances
    output_words_list = ["<sos>"] + output_words_list + ["<eos>"]
    num_nodes = len(output_words_list)
    D = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            src = output_words_list[i]
            tgt = output_words_list[j]
            if src == tgt:
                D[i, j] = 0
                continue
            if src == "<eos>" and tgt == "<sos>":
                D[i, j] = 0
                continue
            try:
                D[i, j] = nx.shortest_path_length(G_for_decode, src, tgt)
            except:
                D[i, j] = 10e5  # no path between src and tgt, then infinite distance
                continue

    # if EGTSP
    # transform into file format: http://www.cs.rhul.ac.uk/home/zvero/GTSPLIB/
    # solve by program: http://akira.ruc.dk/~keld/research/GLKH/

    # if ATSP
    ATSP_path = elkai.solve_int_matrix(D)
    ordered_output_words = [output_words_list[idx] for idx in ATSP_path]
    ordered_output_words.remove("<sos>")
    ordered_output_words.remove("<eos>")
    return ordered_output_words
