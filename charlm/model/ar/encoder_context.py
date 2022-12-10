import torch
import torch.nn as nn
from survae.nn.layers.autoregressive import AutoregressiveShift


class IdxContextNet(nn.Sequential):
    def __init__(self, num_classes, context_size, num_layers, hidden_size, dropout, maxlen=400):
        super(IdxContextNet, self).__init__(
            SqueezeLayer(),  # (B,1,L) -> (B,L)
            nn.Embedding(num_classes, hidden_size),  # (B,L,H)
            PermuteLayer((1, 0, 2)),  # (B,L,H) -> (L,B,H)
            LayerLSTM(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            ),  # (L,B,H) -> (L,B,2*H)
            # Transformer(hidden_size, num_layers=num_layers, dropout=dropout, maxlen=maxlen),
            nn.Linear(hidden_size * 2, context_size),  # (L,B,2*H) -> (L,B,P)
            # AutoregressiveShift(context_size),
            PermuteLayer((1, 2, 0)),  # (L,B,P) -> (B,P,L)
        )


class Transformer(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, maxlen):
        super().__init__()
        self.embed_positions = nn.Embedding(maxlen, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(hidden_size, nhead=8, dropout=dropout, norm_first=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        seqlen = x.shape[0]
        positions = self.embed_positions(torch.arange(seqlen).to(x.device)).reshape(seqlen, 1, -1)
        x = x + positions
        for mod in self.layers:
            x = mod(x)
        return x


class LayerLSTM(nn.LSTM):
    def forward(self, x):
        output, _ = super(LayerLSTM, self).forward(x)  # output, (c_n, h_n)
        return output


class SqueezeLayer(nn.Module):
    def forward(self, x):
        return x.squeeze(1)


class PermuteLayer(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)
