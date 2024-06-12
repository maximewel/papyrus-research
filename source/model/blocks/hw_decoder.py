import torch.nn as nn
from activation.activation_func import ActivationHelper, FFActivationFunction

class HwDecoder(nn.Module):
    ### Init variables ###
    # Hidden dimension of the encoder's heads
    hidden_dim: int
    
    # Number of heads of this encoder. In an idea case, each head looks for a different feature in the data
    n_heads: int
    
    # Expension ratio for the feed-forward network. The FFN expends the features dimension and applies an activation function on them.
    # This parameter allows for moving the ffn expension.
    ff_expension_ratio: int
    
    # FF activation: Dictate activation function for the FF network
    ff_activation_function: FFActivationFunction

    # Drop-out ratio: If >0, dropout will be applied to relevant layers (nn layers that can benefit from dropout)
    dropout_ratio: float

    ### Layers ##
    # Multi-head self attention on the encoder target sequence
    target_sequence_mhsa: nn.MultiheadAttention
    # Second MSA layer between result of self-attention on target sequence as query and encoder output as key, values
    encoder_decoder_mha: nn.MultiheadAttention
    
    # Norm layers
    norm_layer_1: nn.LayerNorm
    norm_layer_2: nn.LayerNorm
    norm_layer_3: nn.LayerNorm

    #Feed-forward layer that expends the dim of the features and apply a simple activation function over it
    feed_forward: nn.ModuleList

    # Single dropout layer (no operation is performed / weights learned, acts as a mask, can be used on both activations)
    dropout_layer: nn.Dropout


    def __init__(self, hidden_dim: int, n_heads: int, dropout_ratio: float = 0.1, ff_expension_ratio: int = 2, 
                 ff_activation_function: FFActivationFunction = FFActivationFunction.RELU) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.ff_expension_ratio = ff_expension_ratio
        self.ff_activation_function = ff_activation_function

    def init_layers(self) -> None:
        """Init the layers of this decoder"""
        self.target_sequence_mhsa = nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)
        self.norm_layer_1 = nn.LayerNorm(self.hidden_dim)

        self.target_sequence_mhsa = nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)
        self.norm_layer_2 = nn.LayerNorm(self.hidden_dim)

        activation_function = ActivationHelper.activation_from_enum(self.ff_activation_function)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_expension_ratio * self.hidden_dim),
            activation_function,
            nn.Linear(self.ff_expension_ratio * self.hidden_dim, self.hidden_dim),
        )
        self.norm_layer_3 = nn.LayerNorm(self.hidden_dim)

        self.dropout_layer = nn.Dropout(self.dropout_ratio)

    def forward(self, embedded_input, target_sequence, source_mask, target_mask):
        """Forward the values from the embedded encoder and the shifted target sequence to generate the next prediction"""
        msa_target_out = self.target_sequence_mhsa(target_sequence, target_sequence, target_sequence, source_mask)[0]
        msa_target_out = self.norm_layer_1(embedded_input + self.dropout_layer(msa_target_out))

        input_and_target_attention = self.encoder_decoder_mha(msa_target_out, embedded_input, embedded_input, target_mask)[0]
        input_and_target_attention = self.norm_layer_2(msa_target_out + self.dropout_layer(input_and_target_attention))

        ffn_output = self.feed_forward(input_and_target_attention)
        ffn_output = self.norm_layer_3(input_and_target_attention + self.dropout_layer(ffn_output))

        return ffn_output