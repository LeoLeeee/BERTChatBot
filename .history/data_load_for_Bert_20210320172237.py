# coding=utf-8

import os
import random
import data_tokenize
import torch

def _read_xhj(token='word'):
    convs = data_tokenize.get_convs()
    paragraphs = data_tokenize.tokenize(convs, token)
    random.shuffle(paragraphs)
    return paragraphs

def get_tokens_and_segments(tokens_a, tokens_b=None):  #@save
    if tokens_b is not None:
        tokens = ['<bos>'] + tokens_a 
        segments = [0] * (len(tokens_a) + 1)
        tokens += ['<bos>'] + tokens_b + ['<eos>']
        segments += [1] * (len(tokens_b) + 2)
    else:
        tokens = tokens_a + ['<eos>']
        segments = [0] * (len(tokens_a) + 1)
    return tokens, segments

# def _get_next_sentence(sentence, next_sentence, paragraphs): #@save
#     if random.random() < 0.5:
#         is_next = True
#     else:
#         next_sentence = random.choice(random.choice(paragraphs))
#         is_next = False
#     return sentence, next_sentence, is_next

# def _get_train_data_from_paragraph(paragraph, paragraphs, vocab, max_len): #@save
#     train_data_from_paragraph = []
#     for i in range(len(paragraph) - 1):
#         tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i][0], paragraph[i][1], paragraphs)
#         if len(tokens_a) + len(tokens_b) + 3 > max_len:
#             continue
#         tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
#         train_data_from_paragraph.append((tokens, segments, is_next))
#     return train_data_from_paragraph


def _get_train_data_from_paragraph(paragraph, paragraphs, vocab, max_len): #@save
    train_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        for j in range(2):
            tokens_a = paragraph[i][j]
            if len(tokens_a)  + 2 > max_len:
                tokens_a = tokens_a[:max_len-2]
            tokens, segments = get_tokens_and_segments(tokens_a, None)
            train_data_from_paragraph.append((tokens, segments))
    return train_data_from_paragraph

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab): #@save
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab): #@save
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['<bos>', '<eos>']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions  = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels  = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

# def _pad_bert_inputs(examples, max_len, vocab):
#     max_num_mlm_preds = round(max_len * 0.15)
#     all_tokens_ids, all_segments, valid_lens = [], [], []
#     all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
#     xhj_labels = []
#     for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
#         all_tokens_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
#         all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
#         valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
#         all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
#         all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
#         all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
#         xhj_labels.append(torch.tensor(is_next, dtype=torch.long))
#     return (all_tokens_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, xhj_labels)

def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_tokens_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    xhj_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments) in examples:
        all_tokens_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
    return (all_tokens_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels)


#@save
# class _xhjDataset(torch.utils.data.Dataset):
#     def __init__(self, paragraphs, max_len):
#         self.vocab = data_tokenize.Vocab(paragraphs, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<bos>', '<eos>'])
#         examples = _get_train_data_from_paragraph(paragraphs, paragraphs, self.vocab, max_len)
#         # for paragraph in paragraphs:
#         #     examples.extend(
#         #         _get_train_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len)
#         #     )

#         examples = [
#             (_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples
#         ]
#         (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels, self.xhj_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

#     def __getitem__(self, idx):
#         return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx], self.xhj_labels[idx])
    
#     def __len__(self):
#         return len(self.all_token_ids)

class _xhjDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        self.vocab = data_tokenize.Vocab(paragraphs, min_freq=3, reserved_tokens=['<pad>', '<mask>', '<bos>', '<eos>'])
        examples = _get_train_data_from_paragraph(paragraphs, paragraphs, self.vocab, max_len)
        # for paragraph in paragraphs:
        #     examples.extend(
        #         _get_train_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len)
        #     )

        examples = [
            (_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, )) for tokens, segments in examples
        ]
        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx])
    
    def __len__(self):
        return len(self.all_token_ids)

def load_data_xhj(batch_size, max_len):
    paragraphs = _read_xhj("word")
    train_set = _xhjDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    return train_iter, train_set.vocab