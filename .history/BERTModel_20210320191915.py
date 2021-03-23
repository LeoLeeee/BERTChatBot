# coding = utf-8
import torch
from torch import nn
import AttentionModel as am
import Utility
import math
import os
import data_load_for_Transformer as dlft
import random

# class BERTModel(nn.Module):
#     def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
#                  ffn_num_hiddens, num_heads, num_layers, dropout,
#                  max_len=1000, key_size=768, query_size=768, value_size=768,
#                  hid_in_features=768, mlm_in_features=768,
#                  nsp_in_features=768):
#         super(BERTModel, self).__init__()
#         self.encoder = am.BERTEncoder(vocab_size, num_hiddens, norm_shape,
#                     ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
#                     dropout, max_len=max_len, key_size=key_size,
#                     query_size=query_size, value_size=value_size)
#         self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
#                                     nn.Tanh())
#         self.mlm = am.MaskLM(vocab_size, num_hiddens, mlm_in_features)
#         self.nsp = am.NextSentencePred(nsp_in_features)

#     def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
#         encoded_X = self.encoder(tokens, segments, valid_lens)
#         if pred_positions is not None:
#             mlm_Y_hat = self.mlm(encoded_X, pred_positions)
#         else:
#             mlm_Y_hat = None
#         # The hidden layer of the MLP classifier for next sentence prediction.
#         # 0 is the index of the '<cls>' token
#         nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
#         return encoded_X, mlm_Y_hat, nsp_Y_hat

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = am.BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = am.MaskLM(vocab_size, num_hiddens, mlm_in_features)
        # self.nsp = am.NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        # nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat

class BERTEncoder(nn.Module):
    def __init__(self, bert, hid_in_features=768, num_outputs=768):
        super(BERTEncoder, self).__init__()
        self.encoder = bert.encoder
        # self.hidden = bert.hidden
        self.output = nn.Linear(hid_in_features, num_outputs)

    def forward(self, X, valid_lens=None):
        segments = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long).to(X.device)
        X = self.encoder(X, segments, valid_lens)
        # X = self.hidden(X)
        encoded_X = self.output(X)+X  # imitate the Resnate in case the output is layer is just to be an identity layer.
        return X

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = am.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                am.EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = am.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                am.DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

class BERTDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(BERTDecoder, self).__init__(**kwargs)
        self.decoder = am.BERTDecoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        decoded_X, state = self.decoder(X, state)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

def train_seq2seq(net, data_iter, lr, num_epochs, batch_size, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
    # net.apply(xavier_init_weights)
    try:
        checkpoint_prefix = os.path.join("model_data/model_transformer.pt")
        checkpoint = torch.load(checkpoint_prefix)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    except Exception as e:
        net.apply(xavier_init_weights)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        net.to(device)
        print("Can not load the model with error:", e)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    loss = am.MaskedSoftmaxCELoss()
    net.train()
    animator = am.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs*batch_size])

    
    checkpoint_prefix = os.path.join("model_data/model_transformer.pt")
    # ratio = 100 / len(data_iter)
    # print("ratio=", ratio)
    num_trained = 0
    for epoch in range(num_epochs):
        timer = Utility.Timer()
        metric = am.Accumulator(2)  # Sum of training loss, no. of tokens
        # print("epoch ...", epoch)
        for i, batch in enumerate(data_iter):
            # if random.random() < (1 - ratio * 1.5):
            #     continue
            num_trained += 1
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            Utility.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            # if (i + 1) % 100 == 0:
            # print("    batch>>>", i)
            if (num_trained + 1) % 100 == 0:
                animator.add(num_trained + 1, (metric[0] / metric[1],))
                # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
                torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
        # if (epoch + 1) % 10 == 0:
        # animator.add(epoch + 1, (metric[0] / metric[1],))
        # # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
        # torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
        # sys.stdout.flush()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net = net.to(device)
    net.eval()
    src_sentence = [word for word in src_sentence]
    print("src",src_sentence)
    src_tokens = src_vocab[src_sentence] + [
        src_vocab['<eos>']]
    print("src_tokens=", src_tokens)
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = dlft.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    print("tp_src_tokens", src_tokens)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    print("enc_outputs", enc_outputs)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    print("dec_state", dec_state)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        print("Y", Y[:,:,:10], Y.argmax(dim=2), Y.max(dim=2))
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            print("break", tgt_vocab.to_tokens(pred), pred)
            break
        output_seq.append(pred)
    print(output_seq)
    print(tgt_vocab.to_tokens(output_seq))
    # return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
    return ' '.join(tgt_vocab.to_tokens(output_seq))

def train_bert(net, data_iter, lr, num_epochs, batch_size, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
    # net.apply(xavier_init_weights)
    try:
        checkpoint_prefix = os.path.join("model_data/model_bert.pt")
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
        net.apply(xavier_init_weights)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print("Can not load the model with error:", e)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    loss = am.MaskedSoftmaxCELoss()
    net.train()
    animator = am.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs*batch_size])

    
    checkpoint_prefix = os.path.join("model_data/model_bert.pt")
    # ratio = 100 / len(data_iter)
    # print("ratio=", ratio)
    num_trained = 0
    for epoch in range(num_epochs):
        timer = Utility.Timer()
        metric = am.Accumulator(2)  # Sum of training loss, no. of tokens
        # print("epoch ...", epoch)
        for i, batch in enumerate(data_iter):
            # if random.random() < (1 - ratio * 1.5):
            #     continue
            num_trained += 1
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            # Utility.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            # if (i + 1) % 100 == 0:
            # print("    batch>>>", i)
            if (num_trained + 1) % 100 == 0:
                animator.add(num_trained + 1, (metric[0] / metric[1],))
                # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
                torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
        # if (epoch + 1) % 10 == 0:
        # animator.add(epoch + 1, (metric[0] / metric[1],))
        # # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
        # torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
        # sys.stdout.flush()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')