# -*- coding: utf-8 -*-
"""
NOTICE: first run the following commands
NOTICE: the timeout parameter in java command 1500000 means 1500 seconds the process will terminate. Set it properly. Otherwise, we may have timeout error.

Commands:
1. command line window A: start stanford nlp server
cd some_path/stanford-corenlp-full-2018-10-05/
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9005  -port 9005 -timeout 1500000

2. train model
For example:

python3 GIANT_main.py \
    --epochs 10 \                                                           # set 10 for training
    --data_type event \                                                     # event or concept
    --train_file "../../../../Datasets/original/event/events.json" \        # events or concepts
    --emb_tags "word" "tag" "is_digit" "is_stop" "is_punct" "is_special" \  # use all or use none
    --task_output_dims 2 4 \                                                # whether multi-task, correspond with tasks
    --tasks "phrase" "event" \                                              # tasks
    --edge_types_list "seq" "dep" "contain" "synonym" \                     # test importance
    --not_processed_data --processed_emb \                                  # after processed data, don't set them.
    --d_model 128 \                                                         # 32, 64, 128, 256, ...
    --layers 3 \                                                            # 2, 3, 4, 5, ...
    --num_bases 10 \                                                        # 10, 20, 30, ...
    --debug                                                                 # debug or not
"""
import math
import torch
import torch.nn as nn
from datetime import datetime
from data_loader.GIANT_data import prepro, get_loader
from trainer.GIANT_trainer import Trainer
from util.file_utils import load
from util.exp_utils import set_device, set_random_seed  # logger
from util.exp_utils import summarize_model, get_checkpoint_dir
from config import *
from model.GIANT_model import GIANTNet as Model
from common.constants import RESULT_PATH
from optim.radam import RAdam


def main(args):
    # get revised args
    # NOTICE: here is our default data organization structure. Change it if you are different.
    # original_data_folder = DATA_PATH + "original/" + args.data_type + "/"  # data_type is event or concept.
    processed_data_folder = DATA_PATH + "processed/" + args.data_type + "/"
    args.train_examples_file = processed_data_folder + 'train-examples.pkl'
    args.dev_examples_file = processed_data_folder + 'dev-examples.pkl'
    args.test_examples_file = processed_data_folder + 'test-examples.pkl'
    args.train_output_file = processed_data_folder + 'train_output.txt'
    args.eval_output_file = processed_data_folder + 'eval_output.txt'
    args.test_output_file = processed_data_folder + 'test_output.txt'
    args.emb_mats_file = processed_data_folder + 'emb_mats.pkl'
    args.emb_dicts_file = processed_data_folder + 'emb_dicts.pkl'
    args.counters_file = processed_data_folder + 'counters.pkl'

    # get checkpoint save path
    args_for_checkpoint_folder_name = [
        args.net,
        args.data_type,
        "_".join(args.tasks),
        "_".join(args.emb_tags),
        "_".join(args.edge_types_list),
        args.d_model,
        args.layers,
        args.num_bases,
        args.lr,
        args.debug]  # NOTICE: change here
    save_dir = args.checkpoint_dir
    args.output_file_prefix = "_".join([str(s) for s in args_for_checkpoint_folder_name])
    args.checkpoint_dir = get_checkpoint_dir(save_dir, args_for_checkpoint_folder_name)
    if args.mode != "train":
        args.resume = args.checkpoint_dir + "model_best.pth.tar"  # NOTICE: so set --resume won't change it.

    print(args)

    # set device, random seed, logger
    device, use_cuda, n_gpu = set_device(args.no_cuda)
    set_random_seed(args.seed)
    # logger = set_logger(args.log_file)

    # check whether need data preprocessing. If yes, preprocess data
    if args.not_processed_data:  # use --not_processed_data --spacy_not_processed_data for complete prepro
        prepro(args)

    # # data
    emb_mats = load(args.emb_mats_file)
    emb_dicts = load(args.emb_dicts_file)

    data_list, num_relations, feature_dim = get_loader(args.train_examples_file, 3, shuffle=True, debug=args.debug)
    print("num_relations: ", num_relations)
    # num_relations = len(emb_dicts["edge_types"]) #!!!
    train_dataloader = data_list[0:math.floor(0.8 * len(data_list))]
    dev_dataloader = data_list[math.floor(0.8 * len(data_list)):math.floor(0.9 * len(data_list))]
    test_dataloader = data_list[math.floor(0.9 * len(data_list)):]

    # model
    model = Model(config=args, in_channels=feature_dim, out_channels=args.d_model, num_relations=num_relations, num_bases=args.num_bases,
                  emb_mats=emb_mats, emb_dicts=emb_dicts, dropout=0.1)  # TODO: set them according to args
    summarize_model(model)
    if use_cuda and args.use_multi_gpu and n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print("successfully get model")

    # optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # for p in parameters:
    #     if p.dim() == 1:
    #         p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
    #     # elif list(p.shape) == [args.tgt_vocab_limit, 300]:
    #     #     print("omit embeddings.")
    #     else:
    #         nn.init.xavier_normal_(p, math.sqrt(3))
    optimizer = torch.optim.Adam(
        params=parameters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=3e-7)
    cr = 1.0 / math.log(args.lr_warm_up_num)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log(ee + 1)
        if ee < args.lr_warm_up_num else 1)

    loss = {}
    # loss["P"] = torch.nn.CrossEntropyLoss()
    # loss["D"] = torch.nn.BCEWithLogitsLoss(reduction="sum")

    # trainer
    trainer = Trainer(
        args,
        model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_dataloader=test_dataloader,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        emb_dicts=emb_dicts)

    # start train/eval/test model
    start = datetime.now()
    if args.mode.lower() == "train":
        trainer.train()
    elif args.mode.lower() == "eval_train":
        args.use_ema = False
        train_output_file = RESULT_PATH + "train_output." + args.output_file_prefix + ".txt"
        trainer.eval(train_dataloader, train_output_file)
    elif args.mode.lower() in ["eval", "evaluation", "valid", "validation", "eval_dev"]:
        args.use_ema = False
        eval_output_file = RESULT_PATH + "dev_output." + args.output_file_prefix + ".txt"
        trainer.eval(dev_dataloader, eval_output_file)
    elif args.mode.lower() in ["eval_test"]:
        args.use_ema = False
        test_output_file = RESULT_PATH + "test_output." + args.output_file_prefix + ".txt"
        trainer.eval(test_dataloader, test_output_file)
    elif args.mode.lower() == "test":
        args.use_ema = False
        test_output_file = RESULT_PATH + "test_output." + args.output_file_prefix + ".txt"
        trainer.test(test_dataloader, test_output_file)
    else:
        print("Error: set mode to be train or eval or test or eval_train.")
    print(("Time of {} model: {}").format(args.mode, datetime.now() - start))


if __name__ == '__main__':
    main(parser.parse_args())
