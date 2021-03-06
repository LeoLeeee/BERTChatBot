# coding = utf-8

import os
import random
import data_tokenize
import torch
import Utility

#@save
def _read_xhj(token='char'):
    convs = data_tokenize.get_convs()
    paragraphs = data_tokenize.tokenize(convs, token)
    random.shuffle(paragraphs)
    return paragraphs

#@save
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

#@save
def build_array_xhj_for_Transformer(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    ask_lines = [l[0] + [vocab['<eos>']] for l in lines]
    answer_lines = [l[1] + [vocab['<eos>']] for l in lines]
    ask_array = torch.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in ask_lines])
    ask_valid_len = (ask_array != vocab['<pad>']).type(torch.int32).sum(1)
    answer_array = torch.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in answer_lines])
    answer_valid_len = (answer_array != vocab['<pad>']).type(torch.int32).sum(1)
    return ask_array, ask_valid_len, answer_array, answer_valid_len

#@save
def load_data_xhj_for_Transformer(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    xhj_data = _read_xhj('word')
    vocab = data_tokenize.Vocab(xhj_data, min_freq=0,
                          reserved_tokens=['<pad>', 'mask2', '<bos>', '<eos>'])
    ask_array, ask_valid_len, answer_array, answer_valid_len = build_array_xhj_for_Transformer(xhj_data,vocab, num_steps)
    data_arrays = (ask_array, ask_valid_len, answer_array, answer_valid_len)
    data_iter = Utility.load_array(data_arrays, batch_size)
    print("total training size=", len(data_iter))
    return data_iter, vocab
