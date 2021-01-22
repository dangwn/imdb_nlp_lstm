'''
Author: Dan Gawne
Date: 2021-01-22
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class lstm(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        lstm_blocks = 1, lstm_channels = 64,
        hidden_layers = 1, hidden_channels = 64,
    ):
        super().__init__()

        assert lstm_blocks > 0 and hidden_layers >= 0

        self.lstm_blocks, self.lstm_channels = lstm_blocks, lstm_channels
        self.hidden_layers = hidden_layers

        self.lstm = nn.LSTM(
            in_channels, lstm_channels, lstm_blocks, batch_first = True
        )

        self.fc_in = nn.Linear(lstm_channels, hidden_channels)
        self.fc_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        h0 = torch.zeros(self.lstm_blocks, x.size(0), self.lstm_channels).to(x.device)
        c0 = torch.zeros(self.lstm_blocks, x.size(0), self.lstm_channels).to(x.device)
        
        out, _ = self.lstm(x, (h0,c0))

        out = out[:,-1,:]
        
        out = F.relu(self.fc_in(out))
        
        for _ in range(self.hidden_layers):
            out = F.relu(self.fc_hidden(out))
        
        return self.fc_out(out)

if __name__ == '__main__':
    model = lstm(60,2).to('cuda')

    x = torch.randn(32,140,60).to('cuda')
    print(x.shape)
    print(model(x).shape)
