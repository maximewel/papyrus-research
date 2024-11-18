"""
Unified Handwriting dataset
Used to apply the same transformation to multiple data sources and obtain a unified dataset containing 
uniformily processed HW data.
"""
from __future__ import annotations
from torch.utils.data import Dataset
from source.model.blocks.helper.patches import Patchificator
import torch
import numpy as np
from source.model.blocks.constants.tokens import Tokens
from source.logging.log import logger, LogChannels
from source.model.blocks.constants.sequence_to_image import ImageHelper
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch import Tensor
from source.model.blocks.constants.files import *
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from random import shuffle

class HandWrittingDataset(Dataset):
    #Dataset variables
    lstm_mode: bool
    size: int
    dataset_folder_name: str

    #Parameter to protect RAM while still being efficient when batch-transformer data on the saving method
    PREPARE_TRAINING_DATA_WINDOW_SIZE = 1000

    #Gaussian noise parameters
    GAUSS_MEAN = 0.0
    GAUSS_STD = 1.0
    #DEBUG ONLY
    DISPLAY_GAUSS_AUGMENT = False

    def __init__(self, dataset_folder_name: str, lstm_mode: bool = False):
        super().__init__()

        self.coordinate_to_predict = None
        self.lstm_mode = lstm_mode
        self.dataset_folder_name = dataset_folder_name

        subsequences_path = self.datafolder_subsequence_path(dataset_folder_name)
        self.size = len(os.listdir(subsequences_path))
        logger.log(LogChannels.DATA, f"Detected {self.size} datapoints on datasetfolder {subsequences_path}")
    
    ### Implementation of dataset ###    
    #Override
    def __len__(self):
        return self.size

    #Override
    def __getitem__(self, idx):
        if idx >= len(self):
            raise Exception(f"Invalid index: Dataset of size {len(self)} has no item at index {idx}")
        
        #Retrieve subsequence
        subsequence_folder = self.subsequence_path_at_index(self.dataset_folder_name, idx)
        data = np.load(subsequence_folder)
        subsequence, label = torch.Tensor(data["subsequence"]), torch.Tensor(data["label"])
        if self.lstm_mode:
            return subsequence, label
        else:
            sequence_index = data["image_id"]
            sequence_folder = self.sequence_path_at_index(self.dataset_folder_name, sequence_index)
            sequence_data = np.load(sequence_folder)
            image, patchified_image, patchified_mask = sequence_data["image"], torch.Tensor(sequence_data["patchified_image"]), torch.Tensor(sequence_data["patchified_masks"])
            return image, patchified_image, patchified_mask, subsequence, label

    @staticmethod
    def collate_batch_transformer(batch_data: list[tuple[np.ndarray, Tensor, Tensor, Tensor, Tensor]]) -> tuple[list[np.ndarray], Tensor, Tensor, PackedSequence, Tensor]:
        """
            Collate a HW batch into merged return values
            Use as collate_fn in datasets using HW datasets
            Responds to the data return in __getitem__
            Use in tranformer mode

            Returnss
            -----
                List[np.ndarray], Tensor, Tensor, PackedSequence, Tensor
                    * Images as a list of ndarray - contains the original images
                    * Images_patches as a tensor of homogeneous, padded, patches representing images
                    * Masks corresponding to the image patches
                    * Sequences as PackedSequence of non homogeneous signals
                    * Tensor: Labels as a tensor of size [batch, 2]
        """
        images, patchified_images, masks, sequences, labels = zip(*batch_data)
        return images, torch.stack(patchified_images), torch.stack(masks), pack_sequence(sequences, enforce_sorted=False), torch.stack(labels)
    
    @staticmethod
    def collate_batch_lstm(batch_data: list[tuple[Tensor, Tensor]]) -> tuple[PackedSequence, Tensor]:
        """
            Collate a HW batch into merged return values
            Use as collate_fn in datasets using HW datasets
            Responds to the data return in __getitem__
            Use in LSTM mode

            Returnss
            -----
                PackedSequence, Tensor
                    * Sequences as PackedSequence
                    * Tensor: Labels as a tensor
        """
        sequences, labels = zip(*batch_data)
        
        return pack_sequence(sequences, enforce_sorted=False), torch.stack(labels)
    
    def get_collate_function(self) -> callable:
        """
        Get the collate function adapted to this dataset

        Returns
        -----
            The adapted collate function
        """
        if self.lstm_mode:
            return self.collate_batch_lstm
        else:
            return self.collate_batch_transformer

    ### Implementation of methods to go from numpy signals to workable tensors ###
    @classmethod
    def prepare_and_save_training_data(cls, signals: list[list], save_to_folder: str, patches_dim: tuple, lstm_mode: bool, 
                                       target_image_shape: tuple[int, int], normalize_coordinate_sequences: bool = True,
                                       apply_data_augment_gaussian: bool = False):
        """
        Transform the data to homogeneous tensors on a rolling window
        Save tensors on disk to be loaded on demand

        Args
        -----
            signals: list[list] - List of signals to save
            save_to_folder: str - Folder to save to
        """
        sequence_index = 0
        subsequence_index = 0
        sequences_bundles_to_save = []
        subsequences_bundles_to_save = []

        Path(cls.datafolder_sequence_path(save_to_folder)).mkdir(parents=True, exist_ok=False)
        Path(cls.datafolder_subsequence_path(save_to_folder)).mkdir(parents=False, exist_ok=False)

        with ProcessPoolExecutor() as executor:
            for i in range(0, len(signals), cls.PREPARE_TRAINING_DATA_WINDOW_SIZE):
                upper_bound = min(i+cls.PREPARE_TRAINING_DATA_WINDOW_SIZE, len(signals)-1)
                logger.log(LogChannels.DATA, f"Preparing data {i}:{upper_bound}/{len(signals)}")

                subsequence = signals[i:upper_bound]
                
                #Build tensors training data
                sequences_as_tensor = cls.sequences_to_tensor(subsequence, target_image_shape, normalize_coordinate_sequences)
                images = cls.build_images(subsequence)
                patchified_images, patchified_masks = cls.images_to_tensor(images, patches_dim, target_image_shape)
                signal_subsequences, signal_labels = cls.extract_all_predictable_from_tensor(sequences_as_tensor, lstm_mode, apply_data_augment_gaussian)

                #Save sequences to disk
                for i in range(len(sequences_as_tensor)):
                    sequence_datafolder = cls.sequence_path_at_index(save_to_folder, sequence_index)
                    sequence_bundle = [sequences_as_tensor[i], images[i], patchified_images[i].numpy(), patchified_masks[i].numpy()]
                    sequences_bundles_to_save.append((sequence_datafolder, sequence_bundle))

                    #Save subsequences to disk
                    current_signal_subsequences, current_signal_labels = signal_subsequences[i], signal_labels[i]
                    for (current_signal_subsequence, current_signal_label) in zip(current_signal_subsequences, current_signal_labels):
                        subsequence_datafolder = cls.subsequence_path_at_index(save_to_folder, subsequence_index)
                        #Add reference to the sequence so that each subsequence has a direct link to its sequence
                        subsequence_bundle = [np.array(sequence_index), current_signal_subsequence.numpy(), current_signal_label.numpy()]
                        subsequences_bundles_to_save.append((subsequence_datafolder, subsequence_bundle))

                        subsequence_index += 1
                    
                    sequence_index += 1

                # Save datapoints using multiprocessing
                logger.log(LogChannels.DATA, f"Saving all sequences and subsequences...")
                print(f"Saving {len(sequences_bundles_to_save)} sequences")
                for a in sequences_bundles_to_save:
                    cls.save_sequence_bundle(a)
                map(cls.save_sequence_bundle, sequences_bundles_to_save)
                map(cls.save_subsequence_bundle, subsequences_bundles_to_save)
                # seq_futures = executor.map()
                # subseq_futures = executor.map()
                # list(seq_futures)
                # list(subseq_futures)
                logger.log(LogChannels.DATA, f"Done")

        logger.log(LogChannels.DATA, f"Saved {subsequence_index} datapoints to {save_to_folder}")

    @classmethod
    def save_sequence_bundle(cls, filepath_and_bundle: tuple[str, list]):
        """
        Save a single datapoint to disk
        """
        filepath, (sequence, image, patchified_image, patchified_masks) = filepath_and_bundle
        print(f"Saving sequence to {filepath}")

        with open(filepath, 'wb') as f:
            np.savez_compressed(f, 
                                sequence=sequence,
                                image=image,
                                patchified_image=patchified_image,
                                patchified_masks=patchified_masks)
        
    @classmethod
    def save_subsequence_bundle(cls, filepath_and_bundle: tuple[str, list]):
        """
        Save a single datapoint to disk
        """
        filepath, [image_id, subsequence, label] = filepath_and_bundle
        print(f"Saving subsequence to {filepath}")

        with open(filepath, 'wb') as f:
            np.savez_compressed(f,
                                image_id=image_id,
                                subsequence=subsequence,
                                label=label)

    @classmethod
    def datafolder_path(cls, save_folder: str):
        return os.path.join(DATA_ROOT, DATASET_FOLDER, save_folder)
    
    @classmethod
    def datafolder_sequence_path(cls, save_folder: str):
        return os.path.join(cls.datafolder_path(save_folder), SEQUENCES_FOLDER)

    @classmethod
    def datafolder_subsequence_path(cls, save_folder: str):
        return os.path.join(cls.datafolder_path(save_folder), SUBSEQUENCES_FOLDER)

    @classmethod
    def sequence_path_at_index(cls, save_folder: str, index: int):
        return os.path.join(cls.datafolder_sequence_path(save_folder), f'sequence_{index}.npz')

    @classmethod
    def subsequence_path_at_index(cls, save_folder: str, index: int):
        return os.path.join(cls.datafolder_subsequence_path(save_folder), f'subsequence_{index}.npz')

    @classmethod
    def build_images(cls, signals: list[list[tuple]]):
        """
        Build all the images corresponding to the given signals list
        """
        return [ImageHelper.create_image(signal) for signal in signals]
    
    @classmethod
    def images_to_tensor(cls, images: list[np.ndarray], patches_dim: tuple, target_image_shape: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transform all inhomogeneous images into an homogeneous tensor of padded images with its associated padding masks.
        Store both patched images and masks into class.

        Args
        -----
            images: list[np.ndarray] - The images to transform
        
        Returns
        -----
            patchified_images, patchified_masks: tuple[torch.Tensor, torch.Tensor]
            patchified_images - The images as a tensor of patches
            patchified_masks - The padding masks corresponding to the patches
        """
        #In order to have nice patching, adjust this max to be a multiple of the patch size
        target_image_shape = cls.next_multiple_of_patch(target_image_shape, patches_dim)

        logger.log(LogChannels.DATA, f"The maximum shape of data is {target_image_shape}")
        patchificator = Patchificator(patches_dim, target_image_shape)

        patchified_images, patches_padding_masks = patchificator.normalize_patchify_images(images)

        return patchified_images, patches_padding_masks
    
    @classmethod
    def sequences_to_tensor(cls, signals: list[list[tuple]], target_image_shape: tuple, normalize_coordinate_sequences: bool) -> list[torch.Tensor]:
        """"
        Transform all inhomogeneous into an homogeneous sequence by adding an EOS token as well as padding to the maximum length sequence

        Args
        ----
            signals: list[list[tuple]] - The list of signals to transform to tensors

        Returns
        -----
            list[torch.Tensor] - The list of tensors, with EOS token added at the end
        """
        signals_as_tensor = []

        logger.log(LogChannels.DATA, f"Converting {len(signals)} into tensors")
        
        EOS_TOKEN = [Tokens.COORDINATE_SEQUENCE_EOS.value, Tokens.COORDINATE_SEQUENCE_EOS.value]

        for i in range(len(signals)):
            signal_to_copy = signals[i][:, :2].astype(float)
            if normalize_coordinate_sequences:
                signal_to_copy /= (target_image_shape[1], target_image_shape[0])
            signal_to_copy = np.vstack([signal_to_copy, EOS_TOKEN])
            signals_as_tensor.append(signal_to_copy)
        
        return signals_as_tensor

    @classmethod
    def extract_all_predictable_from_tensor(cls, signals_as_tensor: list[torch.Tensor], lstm_mode: bool, apply_gaussian_data_augmentation: bool) -> torch.Tensor:
        """Extract all the predictable values (datapoints) from a tensor
        ie: for a tensor of length i, generate i-1 sequences of [0:i] where the goal is to generate sequence i+1
        
        """
        #We do not want the data to be on GPU
        signals_subsequences = []
        signals_labels = []

        #For a signal of size i, as we always give the first 
        n_points_to_predict = sum([len(signal)-1 for signal in signals_as_tensor])
        logger.log(LogChannels.DATA, f"Computing signals: from {len(signals_as_tensor)} signals, we have {n_points_to_predict} points to predict")

        number_of_signals = len(signals_as_tensor)
        for i in range(number_of_signals):
            signal = signals_as_tensor[i]
            len_of_signal = signal.shape[0]

            subsequences_current_signal = []
            labels_current_signal = []

            #If necessary, apply gaussian noise to signal (generated for each sequence)
            if apply_gaussian_data_augmentation:
                gaussian_noise = torch.normal(mean=cls.GAUSS_MEAN, std=cls.GAUSS_STD, size=signal.shape)
                #Important: Remove EOS from being noised
                gaussian_noise[-1, :] = 0

                noised_signal = gaussian_noise + signal
                if cls.DISPLAY_GAUSS_AUGMENT:
                    cls.display_gaussian_augment(signal, noised_signal)

            #LSTM does not want to predict [-1,-1]
            max_predict_length = len_of_signal-1 if lstm_mode else len_of_signal
            for j in range(1, max_predict_length):
                #J Starts at 1 as we expect to always have at least 1 data (The starting point) to predict
                subsequence = torch.Tensor(signal[:j])
                label = torch.Tensor(signal[j])
                subsequences_current_signal.append(subsequence)
                labels_current_signal.append(label)

                #Take the noised sub-sequence and the original label to avoid learning to predict 'out-of-skeleton' samples
                if apply_gaussian_data_augmentation:
                    noised_subsequence = torch.Tensor(noised_signal[:j])
                    orig_label = torch.Tensor(signal[j])
                    subsequences_current_signal.append(noised_subsequence)
                    labels_current_signal.append(orig_label)
        
            signals_subsequences.append(subsequences_current_signal)
            signals_labels.append(labels_current_signal)

        return signals_subsequences, signals_labels

    @classmethod
    def next_multiple_of_patch(cls, size: tuple, patch_dim: tuple):
        """
        Very simple util method - retrieve the width/height corresponding to the next whole multiple of the patch size
        Usefull to resize images to dimensions compatible with patch
        """
        whole_multiple_width = int(np.ceil(size[0] / patch_dim[0]))
        whole_multiple_height = int(np.ceil(size[1] / patch_dim[1]))

        return (patch_dim[0] * whole_multiple_width, patch_dim[1] * whole_multiple_height)

    @classmethod
    def display_gaussian_augment(cls, pre_signal, post_signal):
        import matplotlib.pyplot as plt
        
        img_pre_augment = ImageHelper.create_image(np.pad(pre_signal, ((0, 0), (0, 1)), mode='constant', constant_values=0).astype(int))
        img_post_augment = ImageHelper.create_image(np.pad(post_signal, ((0, 0), (0, 1)), mode='constant', constant_values=0).astype(int))
        
        # Create the figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the pre-augmentation image
        axs[0].imshow(img_pre_augment, cmap="gray")
        axs[0].set_title("Original Skeleton")
        axs[0].axis("off")  # Hide axes for better visual clarity

        # Plot the post-augmentation image
        axs[1].imshow(img_post_augment, cmap="gray")
        axs[1].set_title("Noised Skeleton")
        axs[1].axis("off")  # Hide axes for better visual clarity

        print(f"Original:\n{pre_signal}")
        print(f"Noised:\n{post_signal}")

        # Show the plots
        plt.tight_layout()
        plt.show()

    #Not used anymore, as datasources can more simply be separated at the signal level
    @DeprecationWarning
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
