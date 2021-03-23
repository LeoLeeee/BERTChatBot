# coding = utf-8

import torch
from torch import nn
import os
import data_load_for_Transformer as dlft
import BERT_pretraining_single
import BERTModel
import Utility

print("Initiating the hyperparameters...")
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 128, 16
lr, num_epochs, device = 0.001, 2000, Utility.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

