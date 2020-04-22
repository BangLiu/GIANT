# -*- coding: utf-8 -*-
"""
Configuration of our project.
"""
import argparse
from common.constants import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# data directory
# NOTICE: we assume a specific structure for project organization.
# Change the paths in this file and in common/constants.py if your
# project structure is different.
dataset_name = "GIANT"  # NOTICE: change it for different datasets
original_data_folder = DATA_PATH + "original/" + dataset_name + "/"
processed_data_folder = DATA_PATH + "processed/" + dataset_name + "/"

# configure of embeddings
emb_config = {
    "word": {
        "emb_file": AILAB_W2V_TXT_PATH,  # or None if we not use glove
        "emb_size": 8000000,  # full size is int(2.2e6)
        "emb_dim": 200,
        "trainable": False,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "char": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 64,
        "trainable": True,
        "need_conv": True,
        "need_emb": True,
        "is_feature": False},
    # "bpe": {
    #     "emb_file": BPE_EMB_PATH,
    #     "emb_size": 50509,
    #     "emb_dim": 100,
    #     "trainable": False,
    #     "need_conv": True,
    #     "need_emb": True,
    #     "is_feature": False},
    "tag": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "ner": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "iob": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 3,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "dep": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "is_lower": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_stop": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_punct": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_digit": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "like_num": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_bracket": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_overlap": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_special": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 8,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True}}

emb_not_count_tags = {
    "is_digit": [0.0, 1.0],
    "is_stop": [0.0, 1.0],
    "is_punct": [0.0, 1.0],
    "is_special": [0.0, 1.0]}

# parser used to read argument
parser = argparse.ArgumentParser(description='FactorizedQG')

# experiment
parser.add_argument(
    '--seed', type=int, default=123)
parser.add_argument(
    '--mode',
    default='train', type=str,
    help='train, eval or test model (default: train)')
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')
parser.add_argument(
    '--no_cuda',
    default=False, action='store_true',
    help='not use cuda')
parser.add_argument(
    '--use_multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=20, type=int,
    help='only train and test a few batches when debug (default: 5)')
parser.add_argument(
    '--debug_num',
    default=20, type=int,
    help='only train and test a few examples when debug (default: 5)')

# data
parser.add_argument(
    '--not_processed_data',
    default=False, action='store_true',
    help='whether the dataset already processed')
parser.add_argument(
    '--processed_by_spacy',
    default=False, action='store_true',
    help='whether the dataset already processed by spacy')
parser.add_argument(
    '--processed_example_features',
    default=False, action='store_true',
    help='whether the dataset examples are completely processed')
parser.add_argument(
    '--processed_example_graph_features',
    default=False, action='store_true',
    help='whether the dataset examples are completely processed')
parser.add_argument(
    '--processed_emb',
    default=False, action='store_true',
    help='whether the embedding files already processed')

parser.add_argument(
    '--data_file',
    default=original_data_folder + 'data.txt',
    type=str, help='path of full dataset')
parser.add_argument(
    '--train_file',
    default=original_data_folder + 'train.txt',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=original_data_folder + 'dev.txt',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--test_file',
    default=original_data_folder + 'test.txt',
    type=str, help='path of test dataset')

parser.add_argument(
    '--train_examples_file',
    default=processed_data_folder + 'train-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=processed_data_folder + 'dev-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--test_examples_file',
    default=processed_data_folder + 'test-examples.pkl',
    type=str, help='path of test dataset examples file')

parser.add_argument(
    '--train_output_file',
    default=processed_data_folder + 'train_output.txt',
    type=str, help='path of train result file')
parser.add_argument(
    '--eval_output_file',
    default=processed_data_folder + 'eval_output.txt',
    type=str, help='path of evaluation result file')
parser.add_argument(
    '--test_output_file',
    default=processed_data_folder + 'test_output.txt',
    type=str, help='path of test result file')

parser.add_argument(
    '--emb_mats_file',
    default=processed_data_folder + 'emb_mats.pkl',
    type=str, help='path of embedding matrices file')
parser.add_argument(
    '--emb_dicts_file',
    default=processed_data_folder + 'emb_dicts.pkl',
    type=str, help='path of embedding dicts file')
parser.add_argument(
    '--counters_file',
    default=processed_data_folder + 'counters.pkl',
    type=str, help='path of counters file')

parser.add_argument(
    '--lower',
    default=False, action='store_true',
    help='whether lowercase all texts in data')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=1, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=10, type=int,
    help='number of total epochs (default: 20)')
parser.add_argument(
    '--val_num_examples',
    default=10000, type=int,
    help='number of examples for evaluation (default: 10000)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--checkpoint_dir',
    default=OUTPUT_PATH + 'checkpoint/', type=str,
    help='directory of saved model (default: checkpoint/)')
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--resume_partial',
    default=False, action='store_true',
    help='whether resume partial pretrained model component(s)')
parser.add_argument(
    '--print_freq',
    default=1000, type=int,
    help='print training information frequency (default: 1000 steps)')
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--no_grad_clip',
    default=False, action='store_true',
    help='whether use gradient clip')
parser.add_argument(
    '--max_grad_norm',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')
parser.add_argument(
    '--ema_decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=20, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--emb_config',
    default=emb_config, type=dict,
    help='config of embeddings')
parser.add_argument(
    '--emb_not_count_tags',
    default=emb_not_count_tags, type=dict,
    help='tags of embeddings that we will not count by counter')
parser.add_argument(
    '--data_type',
    default='concept', type=str,
    help='which dataset to use')
parser.add_argument(
    '--net',
    default='GIANTNet', type=str,
    help='which neural network model to use')
parser.add_argument(
    '--output_file_prefix',
    default='', type=str,
    help='file name prefix when output file during training, eval or test')

parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--layers', type=int, default=3,
    help='Number of layers in the gcn encoder/decoder')
parser.add_argument(
    '--num_bases', type=int, default=10,
    help='Number of bases used for basis-decomposition in RGCN')

parser.add_argument(
    '--emb_tags', nargs='*', type=str,
    default=["word", "tag", "is_digit", "is_stop", "is_punct", "is_special"],
    help='tags of embeddings that we will use in model')
parser.add_argument(
    '--task_output_dims', nargs='*', type=int,
    default=[2, 4],
    help='node classification tasks')
parser.add_argument(
    '--tasks', nargs='*', type=str,
    default=["phrase", "event"],
    help='node classification tasks')
parser.add_argument(
    '--edge_types_list', nargs='*', type=str,
    default=["seq", "dep", "contain", "synonym"],
    help='types of edges in GIANT input graphs')
parser.add_argument(
    '--use_clean_qt',
    default=False, action='store_true', help='whether clean queries and titles by top maxNQ and maxNT candidate phrases')
parser.add_argument(
    '--indicate_candidate',
    default=False, action='store_true', help='whether clean queries and titles by top maxNQ and maxNT candidate phrases')
parser.add_argument(
    '--maxNQ', type=int, default=10,
    help='Maximum number of Queries to use')
parser.add_argument(
    '--maxNT', type=int, default=10,
    help='Maximum number of Titles to use')

# doing experiments for paper
parser.add_argument(
    '--experiment',
    default=False, action='store_true', help='do experiment for complete model')
parser.add_argument(
    '--ablation_no_emb',
    default=False, action='store_true', help='remove embedding information')
parser.add_argument(
    '--ablation_no_multitask',
    default=False, action='store_true', help='remove multitask setting')
parser.add_argument(
    '--ablation_no_dep',
    default=False, action='store_true', help='remove dependency edges information')
