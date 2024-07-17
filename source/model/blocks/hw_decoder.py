import torch
from torch import Tensor
import torch.nn as nn
from source.model.blocks.constants.activation_func import ActivationHelper, FFActivationFunction

from source.logging.log import logger, LogChannels

from source.model.blocks.constants.device_helper import device

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

    causal_mask: Tensor

    def __init__(self, hidden_dim: int, n_heads: int, target_sequence_length: int, dropout_ratio: float = 0.1, ff_expension_ratio: int = 2, 
                 ff_activation_function: FFActivationFunction = FFActivationFunction.RELU) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.ff_expension_ratio = ff_expension_ratio
        self.ff_activation_function = ff_activation_function

        self.causal_mask = self.generate_causal_mask(target_sequence_length)

        self.init_layers()

    def init_layers(self) -> None:
        """Init the layers of this decoder"""
        self.target_sequence_mhsa = nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)
        self.norm_layer_1 = nn.LayerNorm(self.hidden_dim)

        self.encoder_decoder_mha = nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)
        self.norm_layer_2 = nn.LayerNorm(self.hidden_dim)

        activation_function = ActivationHelper.activation_from_enum(self.ff_activation_function)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_expension_ratio * self.hidden_dim),
            activation_function,
            nn.Linear(self.ff_expension_ratio * self.hidden_dim, self.hidden_dim),
        )
        self.norm_layer_3 = nn.LayerNorm(self.hidden_dim)

        self.dropout_layer = nn.Dropout(self.dropout_ratio)

    def generate_causal_mask(self, target_seq_length: Tensor) -> Tensor:
        """Compute the triangular non-look-ahead mask for the target self-attention
        This mask is used to prevent tokens attending to future tokens that are not generated yet
        """
        # Create a square matrix of shape target_seq_length*target_seq_length 
        # as self attention is [N*N] for input N, and the input is the sequence.
        # Make it triangular with true on the upper part (mask subsequent tokens for each token generation)
        trig_matrix = torch.triu(torch.ones(target_seq_length, target_seq_length, device=device), diagonal=1)

        # Reverse 1-1->0, 1-0->1 to obtain a mask with true on the lower triangular part, false on everything else
        # This effectively forms a mask that only allows each token to look to tokens before itself.
        causal_mask = trig_matrix.bool()
        logger.log(LogChannels.MASKS, f"Causal mask: {causal_mask}")
        return causal_mask
    
    def forward(self, encoder_output: Tensor, target_sequence: Tensor, encoder_padding_mask: Tensor, target_padding_mask: Tensor):
        """Forward the values from the embedded encoder and the shifted target sequence to generate the next prediction
        
        Args
        -----
            encoder_output:         Embedded output of the encoder
            target_sequence:        Target sequence for auto-regression
            encoder_padding_mask:   Padding mask for the encoder, flagging uninteresting padding embeddings   
            target_padding_mask:    Padding mask for the target sequence, flagging uninteresting coordinates
        """
        logger.log(LogChannels.DIMENSIONS, f"Decoder - Dimensions in decoder input (target sequence): {target_sequence.shape}")

        msa_target_out, _ = self.target_sequence_mhsa(target_sequence, target_sequence, target_sequence,
                                                      key_padding_mask=target_padding_mask, attn_mask=self.causal_mask, need_weights=False)
        msa_target_out = self.norm_layer_1(target_sequence + self.dropout_layer(msa_target_out))
        logger.log(LogChannels.DIMENSIONS, f"Decoder - Dimensions output from self-attention: {msa_target_out.shape}")

        logger.log(LogChannels.DIMENSIONS, f"Decoder - Dimensions output from encoder: {encoder_output.shape}")
        input_and_target_attention, _ = self.encoder_decoder_mha(msa_target_out, encoder_output, encoder_output, 
                                                                key_padding_mask=encoder_padding_mask, need_weights=False)
        input_and_target_attention = self.norm_layer_2(msa_target_out + self.dropout_layer(input_and_target_attention))
        logger.log(LogChannels.DIMENSIONS, f"Decoder - after cross-attention: {input_and_target_attention.shape}")

        ffn_output = self.feed_forward(input_and_target_attention)
        ffn_output = self.norm_layer_3(input_and_target_attention + self.dropout_layer(ffn_output))
        logger.log(LogChannels.DIMENSIONS, f"Decoder - Final output dimension after ffn: {ffn_output.shape}")

        return ffn_output