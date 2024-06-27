import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

from nltk import (sent_tokenize as splitter, wordpunct_tokenize as tokenizer)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Vocab:
    def __init__(self, counter, sos, eos, pad, unk, min_freq=None):
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk

        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        self._token2idx = {
            self.sos: self.sos_idx,
            self.eos: self.eos_idx,
            self.pad: self.pad_idx,
            self.unk: self.unk_idx,
        }
        self._idx2token = {idx:token for token, idx in self._token2idx.items()}


        idx = len(self._token2idx)
        min_freq = 0 if min_freq is None else min_freq

        for token, count in counter.items():
            if count > min_freq:
                self._token2idx[token] = idx
                self._idx2token[idx]   = token
                idx += 1

        self.vocab_size = len(self._token2idx)
        self.tokens     = list(self._token2idx.keys())

    def token2idx(self, token):
        return self._token2idx.get(token, self.pad_idx)

    def idx2token(self, idx):
        return self._idx2token.get(idx, self.pad)

    def sent2idx(self, sent):
        return [self.token2idx(i) for i in sent]

    def idx2sent(self, idx):
        return [self.idx2token(i) for i in idx]

    def __len__(self):
        return len(self._token2idx)


class TwitterDataset(Dataset):
    def __init__(self, path):
        data = pickle.load(open(path, 'rb'))
        data = pd.DataFrame.from_dict(data)

        texts  = data['text'].values
        labels = data['label'].values

        train_texts, val_texts, train_labels, val_labels = \
            train_test_split(texts, labels,test_size=0.33, random_state=42)

        words_list = []
        for s in train_texts:
            words_list += s
        words_counter = Counter(words_list)

        sos = "<sos>"
        eos = "<eos>"
        pad = "<pad>"
        unk = "<unk>"

        self.vocab = Vocab(words_counter,
                           sos, eos, pad, unk)

        self.train_texts  = [self.vocab.sent2idx(row) for row in train_texts]
        self.val_texts    = [self.vocab.sent2idx(row) for row in val_texts]
        self.train_labels = train_labels
        self.val_labels  = val_labels

    def __len__(self):
        return len(self.train_texts)

    def get_batch(self, batch_size, val=False):
        pad_token = 0
        if val:
            texts, labels = self.val_texts,   self.val_labels
        else:
            texts, labels = self.train_texts, self.train_labels

        random_idxs  = np.random.randint(0, len(texts), batch_size)
        batch_texts  = [texts[idx] for idx in random_idxs]
        batch_labels = [labels[idx] for idx in random_idxs]
        texts_lens   = list(map(len, batch_texts))

        sorted_texts_lens, sorted_texts, sorted_labels = list(zip(*sorted(zip(texts_lens, batch_texts, batch_labels), key=lambda x: x[0] ,reverse=True)))

        max_lens = sorted_texts_lens[0]

        sorted_padded_texts = [sorted_texts[i] + [pad_token] * (max_lens - sorted_texts_lens[i]) for i in range(batch_size)]
        texts      = torch.LongTensor(sorted_padded_texts)
        labels     = torch.FloatTensor(sorted_labels)
        texts_lens = torch.FloatTensor(sorted_texts_lens)
        return texts, labels, texts_lens
        
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pad_idx):
        super(RNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size,embed_size,padding_idx = pad_idx)
        self.rnn = nn.LSTM(embed_size,hidden_size,batch_first=True)
        self.linear  = nn.Linear(hidden_size, output_size)

    def forward(self, text, text_lengths):
        # text = [batch size, sent len, ]
        # embedded = [ batch size, sent len, emb dim]
        # hidden = [num layers * num directions, batch size, hid dim]

        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        return self.linear(hidden.squeeze(0))