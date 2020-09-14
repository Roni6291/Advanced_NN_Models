import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenizer_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


german = Field(tokenize=tokenizer_de, lower=True,
               init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenizer_en, lower=True,
                init_token="<sos>", eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Transformer(nn.Module):
    def __init__(self, embed_size, src_vocab_size, trg_vocab_size, src_pad_idx,
                 heads, num_encoder_layers, num_decoder_layers,
                 forward_expansion, dropout, max_length, device):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Linear(src_vocab_size, embed_size)
        self.src_positional_embedding = nn.Linear(max_length, embed_size)
        self.trg_word_embedding = nn.Linear(trg_vocab_size, embed_size)
        self.trg_positional_embedding = nn.Linear(max_length, embed_size)
        self.device = device

        self.transformer = nn.Transformer(
            embed_size, heads, num_encoder_layers, num_decoder_layers,
            forward_expansion, dropout)

        self.fc = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask

    def forward(self, src, trg):
        src_len, N = src.shape
        trg_len, N = trg.shape

        src_positions = torch.arange(0, src_len).unsqueeze(
            1).expand(src_len, N).to(self.device)
        trg_positions = torch.arange(0, trg_len).unsqueeze(
            1).expand(trg_len, N).to(self.device)

        embed_src = self.dropout(
            self.src_word_embedding(
                src) + self.src_positional_embedding(src_positions)
        )

        embed_trg = self.dropout(
            self.trg_word_embedding(
                trg) + self.trg_positional_embedding(trg_positions)
        )

        src_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(
            trg_len).to(self.device)

        out = self.transformer(embed_src, embed_trg,
                               src_key_padding_mask=src_mask, tgt_mask=trg_mask)

        return out


# Training phase
device = ("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training Hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

# Model Hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
max_length = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard Parameters
writer = SummaryWriter("runs/loss_plots")
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
)

model = Transformer(embed_size, src_vocab_size, trg_vocab_size, src_pad_idx,
                    heads, num_encoder_layers, num_decoder_layers,
                    forward_expansion, dropout, max_length, device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(input_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy
        # Loss doesn't take input in that form. For example if we have MNIST
        # we want to have output to be: (N, 10) and targets just (N). Here we
        # can view it in a similar way that we have output_words * batch_size
        # that we want to send in into our cost function, so we need to do
        # some reshapin. Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")