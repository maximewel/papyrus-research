import torch.nn as nn

class HwLstm(nn.Module):
    lstm_layer: nn.LSTM
    output_linear_layer: nn.Linear
    
    last_hidden_state: None
    last_cell_state: None

    keep_state: bool
    
    #X,Y coordinates
    OUTPUT_SIZE = 2

    def __init__(self, input_size, hidden_size, num_layers, keep_state: bool = False):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output_linear_layer = nn.Linear(2*hidden_size, HwLstm.OUTPUT_SIZE)

        #Keep state can be used when working to avoid re-computation of the whole sequence, just keeping the state for a single signal
        self.set_keep_state(keep_state)
    
    def set_keep_state(self, keep_state : bool):
        """Set the keep state value"""
        self.keep_state = keep_state
        if self.keep_state:
            self.clear()

    def clear(self):
        """Reset this LSTM state when working on new signal"""
        self.last_cell_state = ()
        self.last_hidden_state = ()

    def forward(self, x):
        # Pass through LSTM (L,N,Hin​)
        # output, (hn, cn) = rnn(input, (h0, c0))
        if self.keep_state:
            x, (self.last_hidden_state, self.last_cell_state) = self.lstm_layer(x, (self.last_hidden_state, self.last_cell_state))
        else:
            x, _ = self.lstm_layer(x)

        #Return linear layer prediction for next value based on the last very last hidden output of every batched input
        x = self.output_linear_layer(x[:, -1, :])
        return x