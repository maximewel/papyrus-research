import torch.nn as nn
import torch

from source.model.blocks.constants.activation_func import FFActivationFunction, ActivationHelper

from source.logging.log import logger, LogChannels

class HwEncoder(nn.Module):
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
    #multi-head self-attention (single layer in encoder)
    mhsa: nn.MultiheadAttention
    #Feed-forward layer that expends the dim of the features and apply a simple activation function over it
    feed_forward: nn.ModuleList
    #Norm layers
    norm_layer_1: nn.LayerNorm
    norm_layer_2: nn.LayerNorm
    #Single dropout layer (no operation is performed / weights learned, acts as a mask, can be used on both activations)
    dropout_layer: nn.Dropout

    def __init__(self, hidden_dim: int, n_heads: int, ff_expension_ratio: int = 2, 
                 ff_activation_function: FFActivationFunction = FFActivationFunction.RELU, dropout_ratio: float = 0.1)  -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.ff_expension_ratio = ff_expension_ratio
        self.ff_activation_function = ff_activation_function
        if dropout_ratio < 0 or dropout_ratio >= 1:
            raise Exception(f"Dropout ratio {dropout_ratio} invalid. Please select droupout ratio c [0; 1[")
        self.dropout_ratio = dropout_ratio
        
        self.init_layers()

    def init_layers(self):
        """Init the layers of this encoder for further use"""
        self.mhsa = nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)
        self.norm_layer_1 = nn.LayerNorm(self.hidden_dim)

        activation_function = ActivationHelper.activation_from_enum(self.ff_activation_function)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_expension_ratio * self.hidden_dim),
            activation_function,
            nn.Linear(self.ff_expension_ratio * self.hidden_dim, self.hidden_dim),
        )
        self.norm_layer_2 = nn.LayerNorm(self.hidden_dim)

        self.dropout_layer = nn.Dropout(self.dropout_ratio)
    
    def forward(self, x:torch.Tensor, source_padding_mask: torch.Tensor):
        """Forward pass of the HW encoder.
        Args
        -----
            x: Batch of patchified images of dimensions [batch_size, n_patches, hidden_d]
                With batch_size the number of images in a batch
                With n_patches the number of patches per image
                With hidden_d the flattened dimensions of the patches
            source_padding_mask: Mask of dimensions [batch_size, n_patches] where true indicates that the patch is 
        """
        ## P1 ##
        #Do MHA over input
        #Pre-layer norm before msa
        x_norm = self.norm_layer_1(x) 
        msa_out, _ = self.mhsa(x_norm, x_norm, x_norm, key_padding_mask=source_padding_mask, need_weights=False)
        msa_out = self.dropout_layer(msa_out)

        #Add
        x = x + msa_out

        ## P2 ##
        #Do feed forward over input
        # Pre-LayerNorm before feedforward
        x_norm = self.norm_layer_2(x)  
        ff_out = self.feed_forward(x_norm)
        ff_out = self.dropout_layer(ff_out)
        x = x + ff_out 

        return x