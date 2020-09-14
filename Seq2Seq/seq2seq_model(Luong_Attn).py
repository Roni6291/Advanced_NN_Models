import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=dropout, batch_first=True)
        self.bn1d = nn.BatchNorm1d(hidden_size, momentum=0.6)

    def forward(self, x):
        # x: [src_len, features]
        x = torch.unsqueeze(x, 0)
        # x: [1, sec_len, features]

        outputs, (hidden, cell) = self.lstm(x)
        # outputs: [1, src_len, hidden_size]
        # hidden: [1, 1, hidden_size]
        # cell: [1, 1, hidden_size]

        hidden = self.bn1d(hidden)
        hidden = self.bn1d(hidden)
        # hidden: [1, 1, hidden_size]
        # cell: [1, 1, hidden_size]

        return outputs, hidden, cell


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers, dropout=dropout, batch_first=True)
        self.softmax = nn.Softmax(dim=0)
        self.bn1d = nn.BatchNorm1d(hidden_size, momentum=0.6)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.relu = nn.ReLU()

    def forward(self, trg_len, outputs, hidden, cell):

        h_reshaped = hidden.repeat(1, trg_len, 1)
        # h_reshaped: [1, trg_len, hidden_size]

        decoder_states, _ = self.lstm(h_reshaped, (hidden, cell))
        # outputs: [1, trg_len, hidden_size]

        attention = torch.einsum("bth,bsh->bts", decoder_states, outputs)
        # attention: [1, trg_len, src_len]

        attention = self.softmax(attention)
        # attention: [1, trg_len, src_len]

        context = torch.einsum("bts,bsh->bth", attention, outputs)
        # context: [1, trg_len, hidden_size]
        context = self.bn1d(context)
        # context: [1, trg_len, hidden_size]

        out = self.relu(self.fc(torch.cat((context, decoder_states), dim=2)))

        return out


class Seq2Seq_Luong(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq_Luong, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, trg_len):

        outputs, hidden, cell = self.encoder(x)
        out = self.decoder(trg_len, outputs, hidden, cell)

        return out


if __name__ == "__main__":

    x = torch.randint(1, 100, (200, 2), dtype=torch.float)
    encoder = Encoder(2, 100, 1, 0)
    decoder = Decoder(100, 2, 1, 0)
    model = Seq2Seq_Luong(encoder, decoder)
    out = model.forward(x, 20)
    print(out.shape)
