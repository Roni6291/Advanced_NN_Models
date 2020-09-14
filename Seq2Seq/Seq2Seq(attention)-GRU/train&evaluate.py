"""
File includes the training and evaluate

Coded by Roni Abraham on 28/08/2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
# import numpy as np
import time
# import random
import math

from seq2seq_padding_masking_gru import Encoder, Decoder, Attention, Seq2Seq

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

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 10
CLIP = 1
INPUT_DIM = len(german.vocab)
OUTPUT_DIM = len(english.vocab)
BATCH_SIZE = 128
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = german.vocab.stoi[german.pad_token]

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM,
                  DEC_DROPOUT, attn)
model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

# model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = english.vocab.stoi[english.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# Training Loop
# As we are using include_lengths=True for our source field, batch.src is now
# a tuple with the first element being the numericalized tensor representing the
# sentence and the second element being the lengths of each sentence within the
# batch


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for idx, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, src_len, trg)
        # trg = [trg_len, batch_size]
        # output = [trg_len, batch_size, output_dim]

        output_dim = output.shape[-1]  # trg_vocab_size
        # Output shape is (trg_len, batch_size, output_dim) but Cross Entropy
        # Loss doesn't take input in that form. For example if we have MNIST
        # we want to have output to be: (N, 10) and targets just (N).
        # Here we can view it in a similar way that we have
        # output_words * batch_size that we want to send in into our cost
        # function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# evaluation loop


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)  # turn off teacher forcing
            # trg = [trg_len, batch_size]
            # output = [trg_len, batch_size, output_dim]
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg_len - 1) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, validation_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(
        f'\tTrain Loss: {train_loss: .3f} | Train PPL: {math.exp(train_loss): 7.3f}')
    print(
        f'\t Val. Loss: {valid_loss: .3f} | Val. PPL: {math.exp(valid_loss): 7.3f}')


# Inference
def translate_sentence(sentence, src_field, trg_field, model, device,
                       max_len=50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden,
                                                      encoder_outputs, mask)

        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]
