"""
Contains the handwriting transformer, binding all the submodels together and offering a single interface for predictions

Credits for the encoder-decoder creation can be attributed to multiple internet ressources, such as
    https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
"""
from __future__ import annotations 

import torch.nn as nn
import numpy as np
import torch
from torch import Tensor

from source.model.blocks.hw_encoder import HwEncoder, FFActivationFunction
from source.model.blocks.hw_decoder import HwDecoder

from source.logging.log import logger, LogChannels
from source.model.blocks.constants.tokens import Tokens

class HwTransformer(nn.Module):
    #Models composing this transformer (high level blocks)
    encoder_embedding_layer: nn.Linear
    encoder_layers: nn.ModuleList[HwEncoder]
    decoder_layers: nn.ModuleList[HwDecoder]
    output_mlp: nn.Module
    stop_signal_output: nn.Linear

    #Additional config
    # Generation token used to store 'results', analog to classification token in VIT
    use_prediction_token: bool
    prediction_token: nn.Parameter

    use_lstm: bool

    # Define the output of the model's prediction (it predicts a single point).s
    # Default: Generate (x,y) coordinates with dim=2
    output_dim: int
    #Define the configuration of the encoder-decoder blocks
    ##Common enc/dec
    enc_dec_dropout_ratio: float
    hidden_dim: int
    encoder_patch_dimension: tuple
    fixed_size_image_dimension: tuple

    ##Encoder
    n_encoder_layers: int
    n_encoder_heads: int
    enc_ff_expension_ratio: int
    encoder_ff_activation_Function: FFActivationFunction

    ##Decoder
    n_decoder_layers: int
    n_decoder_heads: int
    dec_ff_expension_ratio: int
    decoder_ff_activation_Function: FFActivationFunction
    autoregressive_target_seq_len: int

    #Usefull components
    unfolder: torch.nn.Unfold
    encoder_positional_embeddings: nn.Parameter
    decoder_positional_embeddings: nn.Parameter

    #Private variables
    n_patches: int

    def __init__(self, 
                    use_prediction_token: bool, use_lstm: bool,
                    hidden_dim: int = 20, enc_dec_dropout_ratio: float = 0.0,
                    encoder_patch_dimension: tuple = (20, 20), fixed_size_image_dimension: tuple = (500, 200),
                    n_encoder_layers: int = 2, n_encoder_heads: int = 4, enc_ff_expension_ratio: int = 4, encoder_ff_activation_Function: FFActivationFunction = FFActivationFunction.RELU,
                    n_decoder_layers: int = 4, n_decoder_heads: int = 4, dec_ff_expension_ratio: int = 4, decoder_ff_activation_Function: FFActivationFunction = FFActivationFunction.RELU, autoregressive_target_seq_len: int = 50,
                    output_dim: int = 2) -> None:
        
        super().__init__()

        #Transformer config
        self.output_dim = output_dim
        self.use_lstm = use_lstm
        self.use_prediction_token = use_prediction_token
        self.encoder_patch_dimension = encoder_patch_dimension
        self.fixed_size_image_dimension = fixed_size_image_dimension

        #Enc/Dec config
        self.hidden_dim = hidden_dim
        self.enc_dec_dropout_ratio = enc_dec_dropout_ratio

        #Enc config
        self.n_encoder_layers = n_encoder_layers
        self.enc_ff_expension_ratio = enc_ff_expension_ratio
        self.encoder_ff_activation_Function = encoder_ff_activation_Function
        self.n_encoder_heads = n_encoder_heads

        #Dec config
        self.n_decoder_layers = n_decoder_layers
        self.n_decoder_heads = n_decoder_heads
        self.dec_ff_expension_ratio = dec_ff_expension_ratio
        self.decoder_ff_activation_Function = decoder_ff_activation_Function
        self.autoregressive_target_seq_len = autoregressive_target_seq_len

        #Init all necessary components of the transformer
        self.init_layers()
        self.init_unfolder()
        self.init_positional_embeddings()

    def init_positional_embeddings(self):
        """Create the fixed positional embeddings for encoder and decoder input"""
        # The encoder positional inputs are added to the image patch embeddings
        # Hence their dimension is relative to the number of patches per image
        # As we have the fixed dimensions of images and patches, we can simply compute
        # The number of patches and pre-compute the embeddings.
        w, h = self.fixed_size_image_dimension
        p_w, p_h = self.encoder_patch_dimension
        n_patches = int(w/p_w * h/p_h)

        self.encoder_positional_embeddings = nn.Parameter(self.get_positional_embeddings(n_patches, self.hidden_dim))
        self.encoder_positional_embeddings.requires_grad = False
        
        #The decoder positional embeddings are added to the target sequence embeddings
        decoder_dim = self.autoregressive_target_seq_len
        if self.use_prediction_token:
            decoder_dim += 1
        if self.use_lstm:
            decoder_dim += 1
        self.decoder_positional_embeddings = nn.Parameter(self.get_positional_embeddings(decoder_dim, self.hidden_dim))
        self.decoder_positional_embeddings.requires_grad = False
    
    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Patchify an image - either square or rectangle"""
        return self.unfolder(images).permute(0, 2, 1)
    
    def init_unfolder(self):
        """Init the unfolder kernel"""
        self.unfolder = torch.nn.Unfold(self.encoder_patch_dimension, stride=self.encoder_patch_dimension)

    def get_positional_embeddings(self, sequence_length: int, dimension: int) -> torch.Tensor:
        result = torch.ones(sequence_length, dimension)
        for i in range(sequence_length):
            for j in range(dimension):
                result[i][j] = np.sin(i / (10000 ** (j / dimension))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / dimension)))
        return result

    def init_layers(self):
        """Initialize the different layers of this model"""
        #Compute the size of the patches
        p_w, p_h = self.encoder_patch_dimension
        patch_dim = int(p_w * p_h)
        self.encoder_embedding_layer = nn.Linear(patch_dim, self.hidden_dim)

        #Decoder 'target sequences' input dimension is the decoder's output dimension
        self.decoder_embedding_layer = nn.Linear(self.output_dim, self.hidden_dim)

        self.encoder_layers = nn.ModuleList([HwEncoder(self.hidden_dim, self.n_encoder_heads, self.enc_ff_expension_ratio, 
                                                       self.encoder_ff_activation_Function, self.enc_dec_dropout_ratio) for _ in range(self.n_encoder_layers)])        

        self.decoder_layers = nn.ModuleList([HwDecoder(self.hidden_dim, self.n_decoder_heads, self.autoregressive_target_seq_len, 
                                                       self.enc_dec_dropout_ratio, self.dec_ff_expension_ratio, self.decoder_ff_activation_Function) for _ in range(self.n_decoder_layers)])        

        #Output MLP has the full flattened sequence embeddings as input and create an output token
        self.output_mlp = nn.Linear(self.autoregressive_target_seq_len * self.hidden_dim , self.output_dim)

        #Output signal indicating whether to end the signal on the next prediction. Result in a single value
        self.stop_signal_output = nn.Linear(self.autoregressive_target_seq_len * self.hidden_dim, 1)

    def normalize_target_sequences(self, target_sequences: Tensor) -> tuple[Tensor, Tensor]:
        """Normalize the target sequences and generate the relevant padding mask
        
        Args
        -----
            target_sequences: Tensor - A tensor of shape [batch, seq_len, seq_dim]
            
        Returns
        -----
            normalized_target_sequences - Tensor: The normalized sequence of tensor of shape [batch, autoregressive_len, seq_dim]
            padding_mask - Tensor: The mask, a boolean tensor with a shape of [batch, autoregressive_len]. Filled with true for padding, false for values
        """
        # Reshape the target sequences so that each of them correspond to the target length. There can be padding (l<fixed_l) or clipping(l>fixed_l).
        # Target is of shape [batch, sequence_len, sequences_dim]
        batch_size, target_len, target_dim = target_sequences.shape
        device = target_sequences.device
        if target_len < self.autoregressive_target_seq_len:
            #Pad to autoregressive len
            padded_targets = torch.full((batch_size, self.autoregressive_target_seq_len, target_dim), Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value, device=device)
            padded_targets[:, :target_len, :] = target_sequences

            #Created associated mask for the encoder, a 2D mask of batch, target_len that identifies the embeddings that we just padded
            source_padding_mask = torch.ones((batch_size, self.autoregressive_target_seq_len), device=device)
            source_padding_mask[:, :target_len] = 0
            source_padding_mask = source_padding_mask.bool()
            return padded_targets.float(), source_padding_mask

        #In case target_len >= self.autoregressive_target_seq_len, no padding is involved
        no_padding_mask = torch.zeros((batch_size, self.autoregressive_target_seq_len), device=device).bool()

        #Rare case (once every training) where length is exactly right
        if target_len == self.autoregressive_target_seq_len:
            return target_sequences, no_padding_mask
    
        #Otherwise, clip and take last X elements
        clipped_targets = target_sequences[:, -self.autoregressive_target_seq_len:]
        return clipped_targets, no_padding_mask

    def forward(self, patchified_images: Tensor, images_padding_masks: Tensor, target_sequences: Tensor):
        """Generate the next predictions
        
        Args:
            images: A batch of images
            target_sequences: A batch of target sequences

        Returns:
            Tensor - (x,y) coordinate 
            Tensor - (bool) stop token 
        """
        logger.log(LogChannels.DIMENSIONS, f"Transformer - images dim: {patchified_images.dtype} {patchified_images.shape}")
        logger.log(LogChannels.DIMENSIONS, f"Transformer - masks dim: {images_padding_masks.dtype} {images_padding_masks.shape}")
        logger.log(LogChannels.DIMENSIONS, f"Transformer - target sequences dim: {target_sequences.dtype} {target_sequences.shape}")

        ## Encoder ##
        #Pass patches through linear layer to obtain embeddings
        embeding_patch_vectors = self.encoder_embedding_layer(patchified_images)
        logger.log(LogChannels.DIMENSIONS, f"Transformer - Embedded images dim: {embeding_patch_vectors.shape}")

        #Add positional embeddings
        n = embeding_patch_vectors.shape[0]
        encoder_positional_encoding = self.encoder_positional_embeddings.repeat(n, 1, 1)
        logger.log(LogChannels.DIMENSIONS, f"Transformer - positional embeddings for patch images: {encoder_positional_encoding.shape}")
        embeding_patch_vectors = embeding_patch_vectors + encoder_positional_encoding

        #Send patchified images to encoder, retrieving embeddings
        encoder_out = embeding_patch_vectors
        for encoder in self.encoder_layers:
            #self, x:torch.Tensor, source_padding_mask: torch.Tensor
            encoder_out = encoder(x=encoder_out, source_padding_mask=images_padding_masks)
        
        ## Decoder ##
        #Normalize all target sequences to the autoregressive length (pad/clip), retrieve associated mask
        normalized_target_sequences, target_sequences_padding_masks = self.normalize_target_sequences(target_sequences)
        #Pass through embedding layer
        embeding_target_sequences = self.decoder_embedding_layer(normalized_target_sequences)
        #Add positional embedding
        n = embeding_target_sequences.shape[0]
        decoder_positional_encoding = self.decoder_positional_embeddings.repeat(n, 1, 1)
        embeding_target_sequences = embeding_target_sequences + decoder_positional_encoding
        logger.log(LogChannels.DIMENSIONS, f"Transformer - Embedded target sequence dim: {embeding_target_sequences.shape}")

        #Send target, images through decoder, receiving final outputs
        decoder_out = embeding_target_sequences
        for decoder in self.decoder_layers:
            decoder_out = decoder(encoder_output=encoder_out, target_sequence=decoder_out, 
                                  encoder_padding_mask=images_padding_masks, target_padding_mask=target_sequences_padding_masks)

        #logger.log(LogChannels.DEBUG, f"Transformer - LAST DIM: {decoder_out.shape}")

        # Pass output through final layer to obtain correct length
        # Reshape to [batch_size, seq_len * seq_dim] and send to final MLP
        flattened_decoder_out = torch.flatten(decoder_out, start_dim=1)  
        #Final MLP will obtain a coordinate (x,y) for each batched input
        final_output = self.output_mlp(flattened_decoder_out)

        # Signal output takes the same flattened decoder output and transforms it into a boolean value
        stop_signal_output = self.stop_signal_output(flattened_decoder_out)

        return final_output, stop_signal_output