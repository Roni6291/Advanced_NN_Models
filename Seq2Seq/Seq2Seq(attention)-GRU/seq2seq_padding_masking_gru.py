"""
I will be adding a few improvements - packed padded sequences and masking -
to the model from the previous notebook. Packed padded sequences are used to
tell our RNN to skip over padding tokens in our encoder. Masking explicitly
forces the model to ignore certain values, such as attention over padded
elements.

Coded by Roni Abraham on 28/08/2020
"""

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
# import numpy as np

import random
# import math

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def de_tokenizer(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def en_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


german = Field(tokenize=de_tokenizer, lower=True, init_token="<sos>",
               eos_token="<eos>", include_lengths=True)
english = Field(
    tokenize=en_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# One quirk about packed padded sequences is that all elements in the batch
# need to be sorted by their non-padded lengths in descending order, i.e. the
# first sentence in the batch needs to be the longest. We use two arguments of
# the iterator to handle this, sort_within_batch which tells the iterator that
# the contents of the batch need to be sorted, and sort_key a function which
# tells the iterator how to sort the elements in the batch

batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src_len, batch_size]

        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch size, emb_dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.gru(packed_embedded)
        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        # when the input is a pad token are all zeros

        # outputs = [src_len, batch_size, hid_dim * num directions]
        # hidden = [n_layers * num_directions, batch_size, hid_dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and
        # backwards
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :],
                                               hidden[-1, :, :]), dim=1)))
        # outputs = [src_len, batch_size, enc_hid_dim * 2]
        # hidden = [batch_size, dec_hid_dim]
        return outputs, hidden

# The attention module is where we calculate the attention values over the
# source sentence.
# Previously, we allowed this module to "pay attention" to padding tokens within
# the source sentence. However, using "masking", we can force the attention to
# only be over non-padding elements
# The forward method now takes a mask input. This is a [batch size, src_len]
# tensor that is 1 when the source sentence token is not a padding token, and 0
# when it is a padding token. For example, if the source sentence is:
# ["hello", "how", "are", "you", "?", <pad>, <pad>], then the mask would
# be [1, 1, 1, 1, 1, 0, 0].
# We apply the mask after the attention has been calculated, but before it has
# been normalized by the softmax function. It is applied using masked_fill. This
# fills the tensor at each element where the first argument (mask == 0) is true,
# with the value given by the second argument (-1e10). In other words, it will
# take the un-normalized attention values, and change the attention values over
# padded elements to be -1e10. As these numbers will be miniscule compared to
# the other values they will become zero when passed through the softmax layer,
# ensuring no attention is payed to padding tokens in the source sentence.


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()

        self.attention = nn.Linear((enc_hid_dim * 2) + dec_hid_dim,
                                   dec_hid_dim)
        self.fc = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        #hidden = [batch_size, dec_hid_dim]
        #encoder_outputs = [src_len, batch_size, enc_hid_dim * 2]

        # batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch_size, src_len, dec_hid_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]

        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs),
                                                     dim=2)))
        # energy = [batch_size, src len, dec_hid_dim]

        attention = self.fc(energy).squeeze(2)
        # attention = [batch_size, src_len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)

# needs to accept a mask over the source sentence and pass this to the attention
# module. As we want to view the values of attention during inference, we also
# return the attention tensor.


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout, attention):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim,
                            output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, encoder_outputs, mask):
        # trg = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [src_len, batch_size, enc_hid_dim * 2]
        # mask = [batch_size, src_len]

        trg = trg.unsqueeze(0)
        # trg = [1, batch_size]

        embedded = self.dropout(self.embedding(trg))
        # embedded = [1, batch_size, emb_dim]

        attn = self.attention(hidden, encoder_outputs, mask)
        # attn = [batch_size, src_len]
        attn = attn.unsqueeze(1)
        #  attn = [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(attn, encoder_outputs)
        # weighted = [batch_size, 1, enc_hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch_size, enc_hid_dim * 2]

        lstm_input = torch.cat((embedded, weighted), dim=2)
        # lstm_input = [1, batch_size, (enc_hid_dim * 2) + emb dim]

        output, hidden = self.rnn(lstm_input, hidden.unsqueeze(0))
        # hidden = [1, batch_size, dec_hid_dim] after unsqueeze

        # new output and hidden
        # output = [1, batch_size, dec_hid_dim * 1]
        # hidden = [1 * 1, batch_size, dec_hid_dim]
        # this also means that output == hidden

        embedded = embedded.squeeze(0)
        # [batch_size, embed_dim]
        output = output.squeeze(0)
        # [batch_size, dec_hid_dim]
        weighted = weighted.squeeze(0)
        # [batch_size, enc_hid_dim * 2]

        prediction = self.fc(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch_size, output_dim]

        # hidden.squeeze(0) = [batch_size, dec_hid_dim]
        # attn.squeeze(1) = [[batch_size, src_len]]
        # attn for inference
        return prediction, hidden.squeeze(0), attn.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # src_len = [batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of
        # the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.input_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)
        # outputs = [trg_len, batch_size, trg_vocab_size]

        # encoder_outputs is all hidden states of the input sequence, back and
        # forwards
        # hidden is the final forward and backward hidden states, passed
        # through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
        # outputs = [src_len, batch_size, enc_hid_dim * 2]
        # hidden = [batch_size, dec_hid_dim]

        # first input to the decoder is the <sos> tokens
        x = trg[0, :]  # x = [1, batch_size]

        mask = self.create_mask(src)
        # mask = [batch size, src_len]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all
            # encoder hidden states and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(x, hidden, encoder_outputs,
                                             mask)
            # output = [batch_size, output_dim]
            # hidden = [batch_size, dec_hid_dim]

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next
            # word otherwise we take the word that the Decoder predicted it to
            # be. Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing
            # is 1 then inputs at test time might be completely different than
            # what the network is used to. This was a long comment.
            x = trg[t] if random.random(
            ) < teacher_forcing_ratio else best_guess

        return outputs
