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

train_iter, vocab = dlft.load_data_xhj_for_Transformer(batch_size, num_steps)

print("Rebuilding the Model...")
bert = BERTModel.BERTModel(len(vocab), num_hiddens=num_hiddens, norm_shape=norm_shape, 
    ffn_num_input=ffn_num_input, ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads, 
    num_layers=num_layers, dropout=dropout, key_size=key_size, query_size=query_size, 
    value_size=value_size, hid_in_features=value_size, mlm_in_features=value_size, 
    nsp_in_features=value_size)
bert = BERT_pretraining_single.load_bert(bert, device)
for v in bert.parameters():
    v.requires_grad = False
encoder = BERTModel.BERTEncoder(bert, hid_in_features=value_size, num_outputs=value_size)
decoder = BERTModel.BERTDecoder(
    len(vocab), num_hiddens=num_hiddens, norm_shape=norm_shape, 
    ffn_num_input=ffn_num_input, ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads, 
    num_layers=num_layers, dropout=dropout, key_size=key_size, query_size=query_size, 
    value_size=value_size, hid_in_features=value_size, mlm_in_features=value_size, 
    nsp_in_features=value_size)
net = BERTModel.EncoderDecoder(encoder, decoder)

try:
    checkpoint_prefix = os.path.join("model_data/model_bert_ed.pt")
    checkpoint = torch.load(checkpoint_prefix)
    net.load_state_dict(checkpoint['model_state_dict'])
except Exception as e:
    print("Can not load the model with error:", e)

print("Ready to working...")
def predict(src_sentence):
    return BERTModel.predict_seq2seq(net, src_sentence, vocab, vocab, num_steps,
                    device, save_attention_weights=False)


