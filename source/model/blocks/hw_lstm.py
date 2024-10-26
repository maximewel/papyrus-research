import torch.nn as nn

class HwLstm(nn.Module):
    lstm_layer: nn.LSTM
    output_linear_layer: nn.Linear
        
    #X,Y coordinates
    OUTPUT_SIZE_COORDINATES = 2

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        Create a LSTM layer

        args
        -----   
            input_size: int     - size of input
            hidden_size: int    - Hidden size of the LSTM, dictates output size if last_layer_mlp is false
            num_layers: int     - Layers of the LSTM
        """
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output_linear_layer = nn.Linear(hidden_size, HwLstm.OUTPUT_SIZE_COORDINATES)

    def forward(self, x, last_layer_mlp: bool):
        """
            Forward call

            args
            -----
                last_layer_mlp: bool - Whether to return a coordinate or the last hidden layer
                    MLP output - Useful for training LSTM
                    Hidden output - Useful for transformer generation
        """
        # Pass through LSTM (L,N,Hin​)
        x, (hn, cn) = self.lstm_layer(x)

        if last_layer_mlp:
            # Return linear layer prediction for next value based on the last very last hidden output of every batched input
            # Useful for training, where loss can be computed between predicted and expected coordinates
            output = self.output_linear_layer(hn[-1])
        else:
            # Return the last hidden layer, which has hidden_dimensions as dim
            output = hn[-1]

        return output