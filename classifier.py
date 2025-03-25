import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            # nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
            # nn.Linear(hid_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.main(x)
