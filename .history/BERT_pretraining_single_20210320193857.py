# coding = utf-8

import torch
from torch import nn
import data_load_for_Bert as dlfb
import BERTModel
import Utility
import AttentionModel as am
import os

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y):
    # Forward pass
    _, mlm_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    return mlm_l

def train_bert(train_iter, net, loss, vocab_size, device, num_steps, lr):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    try:
        checkpoint_prefix = os.path.join("model_data/model_BERT_pretraining_single.pt")
        checkpoint = torch.load(checkpoint_prefix)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    except Exception as e:
        print("Can not load the model with error:", e)

    checkpoint_prefix = os.path.join("model_data/model_BERT_pretraining_single.pt")


    step, timer = 0, Utility.Timer()
    animator = am.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = am.Accumulator(2)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y in train_iter:
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_x = valid_lens_x.to(device)
            pred_positions_X = pred_positions_X.to(device)
            mlm_weights_X = mlm_weights_X.to(device)
            mlm_Y= mlm_Y.to(device)
            optimizer.zero_grad()
            timer.start()
            l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y)
            l.backward()
            optimizer.step()
            timer.stop()
            with torch.no_grad():
                metric.add(l, tokens_X.shape[0])
                animator.add(step + 1,
                            (metric[0] / metric[1]))
            if (step + 1) % 50 == 0:
                torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)

            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[1]:.3f}')
    print(f'{metric[1] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(device)}')

def load_pretrained_model(net, device):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    try:
        checkpoint_prefix = os.path.join("model_data/model_BERT_pretraining.pt")
        checkpoint = torch.load(checkpoint_prefix)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    except Exception as e:
        print("Can not load the model with error:", e)
    return net

def load_bert(net, device):
    # net = net.to(device)
    try:
        checkpoint_prefix = os.path.join("model_data/model_BERT_pretraining_single.pt")
        checkpoint = torch.load(checkpoint_prefix)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
    except Exception as e:
        print("Can not load the model with error:", e)
    return net
