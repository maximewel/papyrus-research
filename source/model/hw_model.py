"""
Contains the handwriting transformer, binding all the submodels together and offering a single interface for predictions

Credits for the encoder-decoder creation can be attributed to multiple internet ressources, such as
    https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
"""

import torch.nn as nn
from .blocks.hw_encoder import HwEncoder, FFActivationFunction
from .blocks.hw_decoder import HwDecoder

class HwTransformer(nn.Module):
    #Models composing this transformer (high level blocks)
    encoder_layers: HwEncoder
    decoder_layers: HwDecoder
    output_mlp: nn.Module

    # Define the output of the model's prediction (it predicts a single point).s
    # Default: Generate (x,y) coordinates with dim=2
    output_dim: int
    #Define the configuration of the encoder-decoder blocks
    ##Encoder
    n_encoder_layers: int
    hidden_dim: int
    n_encoder_heads: int
    ff_expension_ratio: int
    encoder_ff_activation_Function: FFActivationFunction
    encoder_dropout_ratio: float

    ##Decoder
    n_decoder_layers: int
    n_decoder_heads: int

    def __init__(self, 
                    n_encoder_layers: int = 2, hidden_dim:int = 20, n_encoder_heads: int = 4, ff_expension_ratio: int = 4, encoder_ff_activation_Function: FFActivationFunction = FFActivationFunction.GELU, encoder_dropout_ratio: float = 0.1,
                    n_decoder_layers: int = 4, n_decoder_heads: int = 4,
                    output_dim: int = 2) -> None:

        self.output_dim = output_dim

        #Enc config
        self.n_encoder_layers = n_encoder_layers
        self.hidden_dim = hidden_dim
        self.ff_expension_ratio = ff_expension_ratio
        self.encoder_ff_activation_Function = encoder_ff_activation_Function
        self.n_encoder_heads = n_encoder_heads
        self.encoder_dropout_ratio = encoder_dropout_ratio

        #Dec config
        self.n_decoder_layers = n_decoder_layers
        self.n_decoder_heads = n_decoder_heads

        self.init_layers()
    
    def init_layers(self):
        """Initialize the different layers of this model"""
        self.encoder_layers = nn.ModuleList([HwEncoder(self.hidden_dim, self.n_encoder_heads, self.ff_expension_ratio, 
                                                       self.encoder_ff_activation_Function, self.encoder_dropout_ratio) for _ in range(self.n_encoder_layers)])        

        self.output_mlp = nn.Linear(d_model, output_dim)

    def forward(self, image, sub_sequence):
        """Generate the next prediction according to this """