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
from source.model.blocks.hw_lstm import HwLstm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence
from source.model.blocks.constants.device_helper import device

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
    lstm_module: HwLstm

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
    decoder_dim: int

    #Usefull components
    encoder_positional_embeddings: nn.Parameter
    decoder_positional_embeddings: nn.Parameter

    #Private variables
    n_patches: int

    def __init__(self, 
                    use_prediction_token: bool, use_lstm: bool, lstm_module: HwLstm = None,
                    hidden_dim: int = 20, enc_dec_dropout_ratio: float = 0.0,
                    encoder_patch_dimension: tuple = (20, 20), fixed_size_image_dimension: tuple = (500, 200),
                    n_encoder_layers: int = 2, n_encoder_heads: int = 4, enc_ff_expension_ratio: int = 2, encoder_ff_activation_Function: FFActivationFunction = FFActivationFunction.LEAKYRELU,
                    n_decoder_layers: int = 4, n_decoder_heads: int = 4, dec_ff_expension_ratio: int = 2, decoder_ff_activation_Function: FFActivationFunction = FFActivationFunction.LEAKYRELU, autoregressive_target_seq_len: int = 50,
                    output_dim: int = 2) -> None:
        
        super().__init__()

        #Transformer config
        self.output_dim = output_dim
        self.use_lstm = use_lstm
        self.lstm_module = lstm_module
        if use_lstm and lstm_module is None:
            raise Exception("If use_LSTM property is activated on HWTransformer, a LSTM_Module is required")
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
        self.init_positional_embeddings()
        self.init_layers()

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
        self.decoder_dim = self.autoregressive_target_seq_len
        if self.use_prediction_token:
            logger.log(LogChannels.DIMENSIONS, f"Use prediction token, going from {self.decoder_dim} to {self.decoder_dim+1}")
            self.decoder_dim += 1
        if self.use_lstm:
            logger.log(LogChannels.DIMENSIONS, f"Use LSTM, going from {self.decoder_dim} to {self.decoder_dim+1}")
            self.decoder_dim += 1
        self.decoder_positional_embeddings = nn.Parameter(self.get_positional_embeddings(self.decoder_dim, self.hidden_dim))
        self.decoder_positional_embeddings.requires_grad = False

    def get_positional_embeddings(self, sequence_length: int, dimension: int) -> torch.Tensor:
        result = torch.ones(sequence_length, dimension)
        for i in range(sequence_length):
            for j in range(dimension):
                if j % 2 == 0:
                    result[i][j] = np.sin(i / (10000 ** (j / dimension))) 
                else:
                    result[i][j] = np.cos(i / (10000 ** ((j - 1) / dimension)))
        return result

    def init_layers(self):
        """Initialize the different layers of this model"""
        #Compute the size of the patches
        p_w, p_h = self.encoder_patch_dimension
        patch_dim = int(p_w * p_h)

        self.encoder_embedding_layer = nn.Linear(patch_dim, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.encoder_embedding_layer.weight)
        if self.encoder_embedding_layer.bias is not None:
            torch.nn.init.zeros_(self.encoder_embedding_layer.bias)

        #Decoder 'target sequences' input dimension is the decoder's output dimension
        self.decoder_embedding_layer = nn.Linear(self.output_dim, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.decoder_embedding_layer.weight)
        if self.decoder_embedding_layer.bias is not None:
            torch.nn.init.zeros_(self.decoder_embedding_layer.bias)

        self.encoder_layers = nn.ModuleList([HwEncoder(self.hidden_dim, self.n_encoder_heads, self.enc_ff_expension_ratio, 
                                                       self.encoder_ff_activation_Function, self.enc_dec_dropout_ratio) for _ in range(self.n_encoder_layers)])        

        self.decoder_layers = nn.ModuleList([HwDecoder(self.hidden_dim, self.n_decoder_heads, self.decoder_dim, 
                                                       self.enc_dec_dropout_ratio, self.dec_ff_expension_ratio, self.decoder_ff_activation_Function, 
                                                       use_prediction_token=self.use_prediction_token, use_lstm=self.use_lstm) for _ in range(self.n_decoder_layers)])        

        #Output MLP has the full flattened sequence embeddings as input and create an output token
        if self.use_prediction_token:
            self.prediction_token = nn.Parameter(torch.rand(1, self.hidden_dim))
            self.output_mlp = nn.Linear(self.hidden_dim, self.output_dim)
            self.stop_signal_output = nn.Linear(self.hidden_dim, 1)
        else:
            decoder_dim = self.autoregressive_target_seq_len
            if self.use_lstm:
                decoder_dim += 1
            self.output_mlp = nn.Linear(decoder_dim * self.hidden_dim , self.output_dim)
            #Output signal indicating whether to end the signal on the next prediction. Result in a single value
            self.stop_signal_output = nn.Linear(decoder_dim * self.hidden_dim, 1)

    def normalize_target_sequences(self, target_sequences: PackedSequence) -> tuple[Tensor, Tensor]:
        """Normalize the target sequences and generate the relevant padding mask
        
        Args
        -----
            target_sequences: PackedSequence - A packet sequence containing all sequences for this batch
            
        Returns
        -----
            normalized_target_sequences - Tensor: The normalized sequence of tensor of shape [batch, autoregressive_len, seq_dim]
            padding_mask - Tensor: The mask, a boolean tensor with a shape of [batch, autoregressive_len]. Filled with true for padding, false for values
        """
        # Reshape the target sequences so that each of them correspond to the target length. There can be padding (l<fixed_l) or clipping(l>fixed_l).
        normalized_sequences = []
        padding_sequences = []

        sequences, original_lengths = pad_packed_sequence(target_sequences, batch_first=True, padding_value=0)
        target_dim = target_sequences.data[0].shape[0]

        for i, original_length in enumerate(original_lengths):
            normalized_sequence: Tensor
            mask: Tensor

            if original_length <= self.autoregressive_target_seq_len:
                #Case: Padding needed
                padding_count = self.autoregressive_target_seq_len - original_length
                sequence_padding = torch.full((padding_count, target_dim), 
                                              fill_value=Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value, device=device)
                normalized_sequence = torch.cat([sequences[i, :original_length], sequence_padding])
                mask = torch.BoolTensor([i >= original_length for i in range(self.autoregressive_target_seq_len)])
            else:
                #Clip sequence to keep autoregressive length
                normalized_sequence = (sequences[i][:original_length])[-self.autoregressive_target_seq_len:]

                #Nothing is padded
                mask = torch.BoolTensor([False for _ in range(self.autoregressive_target_seq_len)])
            
            if self.use_prediction_token:
                #In case of a prediction token, add always-non-padded item at start of mask
                mask = torch.cat([torch.BoolTensor([False]), mask])
            if self.use_lstm:
                #In case of a LSTM, add always-non-padded item at end of mask
                mask = torch.cat([mask, torch.BoolTensor([False])])

            normalized_sequences.append(normalized_sequence)
            padding_sequences.append(mask.to(device))

        logger.log(LogChannels.MASKS, f"Sequences len: {original_lengths}, Padding masks:\n{padding_sequences}")
        return torch.stack(normalized_sequences), torch.stack(padding_sequences)

    def forward(self, patchified_images: Tensor, images_padding_masks: Tensor, target_sequences: PackedSequence):
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
            encoder_out = encoder(x=encoder_out, source_padding_mask=images_padding_masks)
        
        ## Decoder ##
        #Normalize all target sequences to the autoregressive length (pad/clip), retrieve associated mask
        #Normalizing to autoregression ensure correct dimensions in later MLPs, at the end of the transformer.
        normalized_target_sequences, target_sequences_padding_masks = self.normalize_target_sequences(target_sequences)
        logger.log(LogChannels.MASKS, f"Transformer - Target sequence padding: {target_sequences_padding_masks[0]}")
        logger.log(LogChannels.INTERNAL_SEQUENCE_TRACE, f"Transformer - Normalized target sequence : {normalized_target_sequences}")

        #Pass through embedding layer
        embeding_target_sequences = self.decoder_embedding_layer(normalized_target_sequences)

        # When using classification token, Add classification token to the tokens
        if self.use_prediction_token:
            #Add prediction token
            embeding_target_sequences = torch.cat((self.prediction_token.expand(embeding_target_sequences.shape[0], 1, -1), embeding_target_sequences), dim=1)
            logger.log(LogChannels.DIMENSIONS, f"Transformer - Embedded images dim with prediction token: {embeding_target_sequences.shape}")

        #When using LSTM, Add LSTM last hidden input at end of sequence
        if self.use_lstm:
            #Add LSTM hidden layer
            #LSTM input layer can accept target sequences un-normalized
            lstm_output = self.lstm_module.forward(target_sequences, last_layer_mlp=False).unsqueeze(1)
            embeding_target_sequences = torch.cat([embeding_target_sequences, lstm_output], dim=1)
            logger.log(LogChannels.DIMENSIONS, f"Transformer - Embedded images dim with LSTM hidden layer: {embeding_target_sequences.shape}")

        logger.log(LogChannels.INTERNAL_SEQUENCE_TRACE, f"Transformer - Embedding target sequences : {embeding_target_sequences}")

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
        logger.log(LogChannels.DEBUG, f"Transformer - LAST DIM: {decoder_out.shape}")

        #Final MLP will obtain a coordinate (x,y) for each batched input
        if self.use_prediction_token:
            #In the case of the prediction token, extract said token from decoder and do all computations on it
            prediction_token = decoder_out[:, 0]
            final_output = self.output_mlp(prediction_token)
            stop_signal_output = self.stop_signal_output(prediction_token)
        else:
            # Otherwise, Pass output through final layer to obtain correct length
            # Reshape to [batch_size, seq_len * seq_dim] and send to final MLP
            flattened_decoder_out = torch.flatten(decoder_out, start_dim=1)
            final_output = self.output_mlp(flattened_decoder_out)
            # Signal output takes the same flattened decoder output and transforms it into a boolean value
            stop_signal_output = self.stop_signal_output(flattened_decoder_out)

        return final_output, stop_signal_output
    
    def prepare_sequences_for_lstm(self, sequences: Tensor) -> PackedSequence:
        """
            Remove padding from a sequence in order to better ingest it into the LSTM. As a result of
            inhomogeneous input, return a packedSequence
        """
        filtered_sequences = []

        for sequence in sequences:
            # Create a mask to filter out padding token values
            mask = ~torch.all(sequence == Tokens.PADDING_TENSOR.value, dim=1)
            filtered_sequences.append(sequence[mask])

        return pack_sequence(filtered_sequences, enforce_sorted=False)