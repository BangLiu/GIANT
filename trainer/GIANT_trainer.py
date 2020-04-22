# -*- coding: utf-8 -*-
import os
import shutil
import json
import time
import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from .config import *
from modules.ema import EMA
from model.EGTSP_decoder import decode
import collections


def compute_exact(a_gold, a_pred):
    """
    Inputs can be either both string or both list of tokens.
    """
    return float(a_gold == a_pred)


def compute_f1(a_gold, a_pred):
    """
    Inputs can be either both string or both list of tokens.
    """
    gold_toks = list(a_gold)
    pred_toks = list(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    #print("common: ", common)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class Trainer(object):

    def __init__(self, args, model, train_dataloader, dev_dataloader, test_dataloader,
                 loss, optimizer, scheduler, device, emb_dicts=None,
                 logger=None, partial_models=None, partial_resumes=None,
                 partial_trainables=None):
        self.args = args
        self.device = device
        self.logger = logger
        self.dicts = emb_dicts

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.val_num_batches = args.val_num_examples // args.batch_size

        self.model = model
        self.identifier = type(model).__name__ + '_'

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = None
        if args.use_ema:
            self.ema = EMA(args.ema_decay)
            self.ema.register(model)

        # VARIABLE
        # self.do_order_outputs = (self.args.mode != "train")  # NOTICE: as it is time consuming, we only do it when eval or test
        self.do_order_outputs = True
        self.totalBatchCount = 0
        self.best_result_key = "performance"
        self.result_keys = {
            "phrase": ["acc", "f1", "recall", "precision", "cov", "performance", "ordered_recall", "ordered_precision"],
            "event": ["acc", "f1_macro", "f1_micro", "f1_weighted", "recall", "precision", "cov", "performance"]
        }
        self.best_result = {}
        for task in self.args.tasks:
            self.best_result[task] = {}
            for key in self.result_keys[task]:
                self.best_result[task][key] = 0.0

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        if args.resume:
            self._resume_checkpoint(args.resume)
            self.model = self.model.to(self.device)
            for state in self.optimizer.state.values():  # !!!
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if args.resume_partial:
            num_partial_models = len(partial_models)
            for i in range(num_partial_models):
                self._resume_model(
                    partial_resumes[i],
                    partial_models[i], partial_trainables[i])

    def _update_best_result(self, new_result, best_result):
        is_best = False
        # NOTICE: when we have multi tasks, we use task 0 as the criteria to select best model...
        if (new_result[self.args.tasks[0]][self.best_result_key] > best_result[self.args.tasks[0]][self.best_result_key]):
            is_best = True
        for task in self.args.tasks:
            for key in self.result_keys[task]:
                best_result[task][key] = max(best_result[task][key], new_result[task][key])
        return best_result, is_best

    def _result2string(self, result, result_keys):
        string = ""
        for task in self.args.tasks:
            for key in result_keys:
                string += task + "_" + key + "_" + ("{:.5f}").format(result[task][key])
        return string

    def train(self):
        patience = 0
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            result = self._train_epoch(epoch)

            self.best_result, is_best = self._update_best_result(
                result, self.best_result)

            if self.args.use_early_stop:
                if (not is_best):
                    patience += 1
                    if patience > self.args.early_stop:
                        print("Perform early stop!")
                        break
                else:
                    patience = 0

            if epoch % self.args.save_freq == 0:
                self._save_checkpoint(
                    epoch, result, ["performance"], is_best)  # !!! as we have too much results, we only use this to save model
        return self.model

    def eval(self, dataloader, output_file):
        result, pred_words, gold_words = self._valid(dataloader)
        print("Eval: ", result)
        if output_file is not None:
            with open(output_file, 'w', encoding="utf8") as outfile:
                json.dump(result, outfile, ensure_ascii=False)
                outfile.write("\n")
                for i in range(len(pred_words)):  # TODO: handle more outputs
                    json.dump(pred_words[i], outfile, ensure_ascii=False)
                    json.dump(gold_words[i], outfile, ensure_ascii=False)
                    outfile.write("\n")
                    if self.args.debug and i >= self.args.debug_num:
                        break
            outfile.close()
        return result

    def test(self, dataloader, output_file):
        pred_words = self._test(dataloader)
        if output_file is not None:
            with open(output_file, 'w', encoding="utf8") as outfile:
                for i in range(len(pred_words)):  # TODO: handle more outputs
                    json.dump(pred_words[i], outfile, ensure_ascii=False)
                    outfile.write("\n")
                    if self.args.debug and i >= self.args.debug_num:
                        break
            outfile.close()
        return pred_words

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)

        # initialize
        global_loss = {"total": 0}
        last_step = self.step - 1
        last_time = time.time()

        # train over batches
        for batch_idx, batch in enumerate(self.train_dataloader):
            # get batch
            self.totalBatchCount += 1
            batch = batch.to(self.device)

            # calculate loss and back propagation
            # self.optimizer.zero_grad()  # !!! optimizer or model???
            self.model.zero_grad()
            pred_list = self.model(batch.x, batch.emb_ids_dict, batch.edge_index, batch.edge_type, edge_norm=None)

            labels = {}
            if "phrase" in self.args.tasks:
                labels["phrase"] = batch.y.to(self.device)  # whether this node is in output phrase
            if "event" in self.args.tasks:
                labels["event"] = batch.y_node_type.to(self.device)  # what type of node it is for event elements.

            loss = 0
            for task_idx in range(len(self.args.tasks)):
                predict = pred_list[task_idx]
                label = labels[self.args.tasks[task_idx]]
                loss += F.nll_loss(predict, label)

            loss.backward()
            global_loss["total"] += loss.item()

            # gradient clip
            if (not self.args.no_grad_clip):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)

            # update model
            self.optimizer.step()

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # exponential moving avarage
            if self.args.use_ema:
                self.ema(self.model, self.step)

            # print training info
            if self.step % self.args.print_freq == self.args.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.args.batch_size * \
                    step_num / used_time
                batch_loss = {k: v / step_num for k, v in global_loss.items()}
                print(("step: {}/{} \t "
                       "epoch: {} \t "
                       "lr: {} \t "
                       "loss: {} \t "
                       "speed: {} examples/sec").format(
                           batch_idx, len(self.train_dataloader),
                           epoch,
                           self.optimizer.param_groups[0]['lr'],  # !!!!!!!! because we used special optim
                           # self.optimizer.lr,
                           str(batch_loss),
                           speed))
                global_loss = {k: 0 for k in global_loss}
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.args.debug and batch_idx >= self.args.debug_batchnum:
                break

        # evaluate, log, and visualize for each epoch
        # train_result, _, _ = self._valid(self.train_dataloader)
        dev_result, _, _ = self._valid(self.dev_dataloader)
        test_result, _, _ = self._valid(self.test_dataloader)

        for task in self.args.tasks:
            # print("Task: {}\nTrain result: {}\nDev result: {}\n".format(
            #     task, train_result, dev_result))
            print("Task: {}\nDev result: {}\nTest result: {}\n".format(
                task, dev_result, test_result))

        return dev_result

    def _valid(self, dataloader):
        if self.args.use_ema:
            self.ema.assign(self.model)
        self.model.eval()

        accumu_result = {}
        for task in self.args.tasks:
            accumu_result[task] = {}
            for key in self.result_keys[task]:
                accumu_result[task][key] = 0.0

        all_pred = []
        all_gold = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                batch = batch.to(self.device)

                # get output list. NOTICE: here we assume args.task_output_dims and args.tasks are equal length and correspond.
                # outputs is also a list of the same length with args.tasks
                pred_list = self.model(batch.x, batch.emb_ids_dict, batch.edge_index, batch.edge_type, edge_norm=None)

                labels = {}
                if "phrase" in self.args.tasks:
                    labels["phrase"] = batch.y.to(self.device)  # whether this node is in output phrase
                if "event" in self.args.tasks:
                    labels["event"] = batch.y_node_type.to(self.device)  # what type of node it is for event elements.

                pred_words_dict = {}
                for task in self.args.tasks:
                    pred_words_dict[task] = []  # such as {"phrase": [], "event": []}
                gold_words_dict = {}
                for task in self.args.tasks:
                    gold_words_dict[task] = []  # such as {"phrase": [], "event": []}

                for task_idx in range(len(self.args.tasks)):
                    task = self.args.tasks[task_idx]
                    gold = labels[task].tolist()
                    pred = pred_list[task_idx].max(1)[1].tolist()
                    num_nodes = len(pred)
                    # print("DEBUG pred of task {} is {}".format(task, pred))
                    # print("DEBUG gold of task {} is {}".format(task, gold))

                    pred_words_dict[task] = [
                        batch.words[i] + "_" + str(pred[i])
                        for i in range(num_nodes)
                        if pred[i] != 0]
                    gold_words_dict[task] = [
                        batch.words[i] + "_" + str(gold[i])
                        for i in range(num_nodes)
                        if gold[i] != 0]

                    unordered_pred_words = [
                        batch.words[i] for i in range(num_nodes) if pred[i] != 0]
                    ordered_gold_words = batch.phrase_features["word"]
                    unordered_pred_processed = [pred_w for pred_w in unordered_pred_words if pred_w not in ["的", "<sos>", "<eos>"]]
                    ordered_gold_processed = [gold_w for gold_w in ordered_gold_words if gold_w not in ["的", "<sos>", "<eos>"]]
                    if self.do_order_outputs and task == "phrase":
                        ordered_pred_words = decode(
                            unordered_pred_words, batch.G_for_decode, batch.queries_features, batch.titles_features)
                        ordered_pred_processed = [pred_w for pred_w in ordered_pred_words if pred_w not in ["的", "<sos>", "<eos>"]]
                        pred_words_dict[task] = ordered_pred_words
                        gold_words_dict[task] = ordered_gold_words

                    if task == "phrase":
                        accumu_result[task]["acc"] += accuracy_score(gold, pred)
                        # accumu_result[task]["f1"] += f1_score(gold, pred, average="binary")
                        accumu_result[task]["f1"] += compute_f1("".join(ordered_gold_processed), "".join(unordered_pred_processed))
                        accumu_result[task]["recall"] += compute_exact(set("".join(ordered_gold_processed)), set("".join(unordered_pred_processed)))
                        accumu_result[task]["cov"] += float(sum(pred) > 0 or (gold == pred))
                        if self.do_order_outputs:
                            accumu_result[task]["ordered_recall"] += compute_exact("".join(ordered_gold_processed), "".join(ordered_pred_processed))
                    if task == "event":
                        accumu_result[task]["acc"] += accuracy_score(gold, pred)
                        accumu_result[task]["f1_macro"] += f1_score(gold, pred, average="macro")
                        accumu_result[task]["f1_micro"] += f1_score(gold, pred, average="micro")
                        accumu_result[task]["f1_weighted"] += f1_score(gold, pred, average="weighted")
                        accumu_result[task]["recall"] += compute_exact(set(ordered_gold_processed), set(unordered_pred_processed))
                        accumu_result[task]["cov"] += float(sum(pred) > 0 or (gold == pred))

                all_pred.append(pred_words_dict)
                all_gold.append(gold_words_dict)

                if((batch_idx + 1) == self.val_num_batches):
                    break

                if self.args.debug and batch_idx >= self.args.debug_batchnum:
                    break

        # get evaluation result by comparing truth and prediction
        result = {}
        for task in self.args.tasks:
            result[task] = {k: v / len(dataloader) for k, v in accumu_result[task].items()}  # NOTICE: not quite right.
            result[task]["precision"] = (result[task]["recall"] + 10e-30) / (result[task]["cov"] + 10e-20)
            if task == "phrase" and self.do_order_outputs:
                result[task]["ordered_precision"] = (result[task]["ordered_recall"] + 10e-30) / (result[task]["cov"] + 10e-20)
            result[task]["performance"] = result[task]["f1"] if task == "phrase" else result[task]["f1_weighted"]  # for save model
        print("Valid results: ", result)

        if self.args.use_ema:
            self.ema.resume(self.model)
        self.model.train()
        return result, all_pred, all_gold

    def _test(self, dataloader):
        if self.args.use_ema:
            self.ema.assign(self.model)
        self.model.eval()

        all_pred = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                batch = batch.to(self.device)
                # ground truth
                pred_list = self.model(batch.x, batch.emb_ids_dict, batch.edge_index, batch.edge_type, edge_norm=None)

                pred_words_dict = {}
                for task in self.args.tasks:
                    pred_words_dict[task] = []  # such as {"phrase": [], "event": []}

                for task_idx in range(len(self.args.tasks)):
                    task = self.args.tasks[task_idx]
                    pred = pred_list[task_idx].max(1)[1].tolist()
                    num_nodes = len(pred)
                    pred_words_dict[task] = [
                        batch.words[i] + "_" + str(pred[i])
                        for i in range(len(pred))
                        if pred[i] != 0]
                    if self.do_order_outputs and task == "phrase":
                        unordered_pred_words = [
                            batch.words[i] for i in range(num_nodes) if pred[i] != 0]
                        ordered_pred_words = decode(
                            unordered_pred_words, batch.G_for_decode, batch.queries_features, batch.titles_features)
                        pred_words_dict[task] = ordered_pred_words

                all_pred.append(pred_words_dict)

                if self.args.debug and batch_idx >= self.args.debug_batchnum:
                    break

        if self.args.use_ema:
            self.ema.resume(self.model)
        self.model.train()
        return all_pred

    def _save_checkpoint(self, epoch, result, result_keys, is_best):
        if self.args.use_ema:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_result': self.best_result,
            'step': self.step + 1,
            'start_time': self.start_time}
        filename = os.path.join(
            self.args.checkpoint_dir,
            self.identifier +
            'checkpoint_epoch{:02d}'.format(epoch) +
            self._result2string(result, result_keys) + '.pth.tar')
        print("Saving checkpoint: {} ...".format(filename))
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(
                    self.args.checkpoint_dir, 'model_best.pth.tar'))
        if self.args.use_ema:
            self.ema.resume(self.model)
        return filename

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(
            resume_path, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_result = checkpoint['best_result']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.scheduler is not None:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))

    def _resume_model(self, resume_path, model, trainable=True):
        checkpoint = torch.load(
            resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if not trainable:
            for p in model.parameters():
                p.requires_grad = False
        print("Model loaded")
