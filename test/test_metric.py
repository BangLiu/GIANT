import collections


def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)


def compute_f1(a_gold, a_pred):
    gold_toks = list(a_gold)
    pred_toks = list(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    print("common: ", common)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# a_gold = "王宝强离婚"
# a_pred = "王宝强马蓉"

# print(compute_exact(a_gold, a_pred))
# print(compute_exact(a_gold, a_gold))
# print(compute_f1(a_gold, a_pred))


a_gold = ["王宝强", "离婚"]
a_pred = ["王宝强", "马蓉"]


print(compute_exact(a_gold, a_pred))
print(compute_exact(a_gold, a_gold))
print(compute_f1(a_gold, a_pred))
