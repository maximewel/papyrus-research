"""
Unified Handwriting dataset
Used to apply the same transformation to multiple data sources and obtain a unified dataset containing 
uniformily processed HW data.
"""

from torch.utils.data import Dataset
from source.model.blocks.helper.patches import Patchificator
from source.model.blocks.helper.tensor_utils import TensorUtils
import torch
import numpy as np
from source.model.blocks.constants.tokens import Tokens
from source.logging.log import logger, LogChannels
from source.model.blocks.constants.sequence_to_image import ImageHelper

from random import shuffle

from abc import ABC, abstractmethod

class HandWrittingDataset(Dataset, ABC):
    patchified_images: torch.Tensor
    patches_padding_masks: torch.Tensor
    signals_as_tensor: list[torch.Tensor]

    #Base information that underlying classes must define
    signals: list[list]
    images: list[np.ndarray]

    #Used in Transformer mode
    batchified_patchified_images: list[torch.Tensor]
    batchified_patches_padding_masks: list[torch.Tensor]
    batchified_sequences: list[torch.Tensor]

    #Used in LSTM mode
    coordinate_to_predict: list[torch.Tensor]

    patchificator: Patchificator
    normalize_pixel_values: bool
    normalize_coordinate_sequences: bool

    patches_dim: tuple
    target_image_shape: tuple

    window_size: int
    lstm_forecast_length: int

    samples_to_take: int|float

    def __init__(self, patches_dim: tuple, normalize_pixel_values: bool = True, normalize_coordinate_sequences: bool = True, 
                 window_size: int = None, lstm_forecast_length: int = None, samples_to_take: int | float = None):
        super().__init__()

        self.coordinate_to_predict = None
        self.lstm_forecast_length = lstm_forecast_length
        self.samples_to_take = samples_to_take

        self.patches_dim = patches_dim
        self.window_size = window_size

        self.normalize_pixel_values = normalize_pixel_values
        self.normalize_coordinate_sequences = normalize_coordinate_sequences

        #Ask the implementationt to load the signals/images data
        self.signals = []
        self._load_data()
        logger.log(LogChannels.DATA,f"Loaded {len(self.signals)} Sequences for a total of {sum([len(signal) for signal in self.signals])} data points")

    @abstractmethod
    def _load_data(self):
        """
        Load_data: Private function used only by the dataset itself.
        Classes implementing handwriting datasets must simply fill the self.signals 
        and self.images lists from their data sources.
        """
        raise NotImplementedError()
    
    def build_images(self):
        """
        Build all the images corresponding to this dataset signal
        """
        logger.log(LogChannels.DATA, f"Creating images...")
        self.images = [ImageHelper.create_image(signal) for signal in self.signals]
        logger.log(LogChannels.DATA, f"Images created")
    
    def take_samples_of_dataset(self, number: int|float):
        """
        Restrict dataset to a sample of it.
        If provided an integer, take this number of samples.
        If provided a float, take this % of the total dataset length. 
        """
        samples_count_to_take = None
        signal_len = len(self.signals)

        if isinstance(number, int):
            if number < 0 or number > signal_len:
                raise TypeError(f"Impossible to take {number} samples in dataset of {signal_len} values")
            samples_count_to_take = number
        elif isinstance(number, float):
            if number < 0 or number > 1:
                raise TypeError(f"Keep ratio between 0 and 1, impossible to take {number}")
            samples_count_to_take = int(number * signal_len)
        else:
            raise TypeError(f"Pass an integer or float to sampling function, unable to process {type(number)}")
        
        logger.log(LogChannels.DATA, f"Restricting dataset to {samples_count_to_take} samples")
        shuffle(self.signals)
        self.signals = self.signals[0:samples_count_to_take]
    
    ### Implementation of dataset ###    
    #Override
    def __len__(self):
        return len(self.signals) if self.coordinate_to_predict is None else len(self.coordinate_to_predict)

    #Override
    def __getitem__(self, idx):
        if idx >= len(self):
            raise Exception(f"Invalid index: Dataset of size {len(self)} has no item at index {idx}")
        
        image, mask, sequence = self.batchified_patchified_images[idx], self.batchified_patches_padding_masks[idx], self.batchified_sequences[idx]
        label = self.coordinate_to_predict[idx]
        
        return (image, mask, sequence), label
    
    
    ### Implementation of methods to go from numpy signals to workable tensors ###
    def transform_to_batch(self):
        """Transform the data to homogeneous tensors"""
        self.images_to_tensor()
        self.sequences_to_tensor()
        
        self.extract_all_predictable_from_tensor()
            
    def images_to_tensor(self):
        """
        Transform all inhomogeneous images into an homogeneous tensor of padded images with its associated padding masks.
        Store both patched images and masks into class.
        """
        #Calculate the strict maximum image shape of the dataset
        images_widths, images_heigth = [image.shape[0] for image in self.images], [image.shape[1] for image in self.images]
        max_image_in_dataset = (np.max(images_widths), np.max(images_heigth))

        #In order to have nice patching, adjust this max to be a multiple of the patch size
        self.target_image_shape = self.next_multiple_of_patch(max_image_in_dataset, self.patches_dim)

        logger.log(LogChannels.DATA, f"The maximum shape of data is {self.target_image_shape}")

        patchificator = Patchificator(self.patches_dim, self.target_image_shape)

        self.patchified_images, self.patches_padding_masks = patchificator.normalize_patchify_images(self.images, normalize_value=self.normalize_pixel_values)
    
    def sequences_to_tensor(self):
        """"
        Transform all inhomogeneous into an homogeneous sequence by adding an EOS token as well as padding to the maximum length sequence
        """
        sequences_len = [len(signal) for signal in self.signals]
        max_sequences_len = np.max(sequences_len)  + 1 #+1: Used to account for the added EOS on the lengthiest signal

        self.signals_as_tensor = torch.full((len(self.signals), max_sequences_len, 2), Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value).float()

        logger.log(LogChannels.DATA, f"Converting {len(self.signals)} signals into a tensor of {self.signals_as_tensor.shape}")
        
        EOS_TOKEN = [Tokens.COORDINATE_SEQUENCE_EOS.value for _ in range(2)]

        for i in range(len(self.signals)):
            signal_to_copy = self.signals[i][:, :2].astype(float)
            if self.normalize_coordinate_sequences:
                signal_to_copy /= self.target_image_shape
            signal_to_copy = np.vstack([signal_to_copy, EOS_TOKEN])
            self.signals_as_tensor[i, :len(signal_to_copy), :] = torch.FloatTensor(signal_to_copy[:, :2])

    def extract_all_predictable_from_tensor(self) -> torch.Tensor:
        """Extract all the predictable values from a tensor
        ie: for a tensor of length i, generate i-1 sequences of [0:i] where the goal is to generate sequence i+1"""
        #We do not want the data to be on GPU
        coordinates_to_predict = []
        signals_to_predict = []
        batchified_images = []
        batchified_masks = []

        #For a signal of size i, as we always give the first 
        n_points_to_predict = sum([TensorUtils.true_seq_lengths_of_tensor(signal)-1 for signal in self.signals_as_tensor])
        logger.log(LogChannels.DATA, f"Computing signals: from {self.signals_as_tensor.shape[0]} signals, we have {n_points_to_predict} points to predict")

        padding_token = torch.FloatTensor([Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value, Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value], device='cpu')
        
        number_of_signals, signal_max_len, _ = self.signals_as_tensor.shape
        for i in range(number_of_signals):
            signal = self.signals_as_tensor[i]
            len_of_signal = TensorUtils.true_seq_lengths_of_tensor(signal)

            if self.lstm_forecast_length is not None:
                if len_of_signal < self.lstm_forecast_length + 1:
                    continue
                for j in range(0, len_of_signal - self.lstm_forecast_length - 1):
                    subsequence = signal[j : self.lstm_forecast_length + j]

                    signals_to_predict.append(subsequence)
                    coordinates_to_predict.append(signal[self.lstm_forecast_length + j])
                    batchified_images.append(self.patchified_images[i])
                    batchified_masks.append(self.patches_padding_masks[i])
            else:
                for j in range(1, len_of_signal-1):
                    subsequence = signal[:j]
                    padding_length = signal_max_len - j
                    padding_tensor = padding_token.unsqueeze(0).repeat(padding_length, 1)
                    padded_subsequence = torch.cat((subsequence, padding_tensor), dim=0)

                    signals_to_predict.append(padded_subsequence)
                    coordinates_to_predict.append(signal[j])
                    batchified_images.append(self.patchified_images[i])
                    batchified_masks.append(self.patches_padding_masks[i])

        self.batchified_sequences = signals_to_predict
        self.batchified_patchified_images = batchified_images
        self.batchified_patches_padding_masks = batchified_masks
        self.coordinate_to_predict = coordinates_to_predict

    def next_multiple_of_patch(self, size: tuple, patch_dim: tuple):
        """
        Very simple util method - retrieve the width/height corresponding to the next whole multiple of the patch size
        Usefull to resize images to dimensions compatible with patch
        """
        whole_multiple_width = int(np.ceil(size[0] / patch_dim[0]))
        whole_multiple_height = int(np.ceil(size[1] / patch_dim[1]))

        return (patch_dim[0] * whole_multiple_width, patch_dim[1] * whole_multiple_height)
