import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import math
import pickle
from enum import Enum, auto
from pathlib import Path

from source.logging.log import logger, LogChannels
from source.model.blocks.constants.tokens import Tokens
from source.model.blocks.helper.patches import Patchificator
from source.model.blocks.helper.tensor_utils import TensorUtils
from source.model.blocks.constants.sequence_to_image import ImageHelper


import torch

class StrokeMode(Enum):
    STROKES = auto()
    SUBSTROKES = auto()

class BrushDataset(Dataset):
    """The BRUSH dataset object is used to retrieve all samples from the BRUSH dataset
    This dataset retrives them as:
    Sample  = Offline image
    Label   = Online signal
    The loading of the dataset is special, as we retrieve only the label in memory and build
    the images at runtime.
    """
    brush_root: str
    save_to_file: bool

    DRAW_COLOR_WHITE = 0
    DRAW_COLOR_BLACK = 255
    DRAW_COLOR_SIZE = 1

    RAW_DIR = "raw"
    GROUPED_STROKES_DIR = "FULL_STROKES"
    GROUPED_SUBSTROKES_DIR = "FULL_SUBSTROKES"
    FOLDER_IMAGES = "offline"
    FOLDER_SIGNALS = "online"
    FILE_IMAGES = "images.npy"
    FILE_SIGNALS = "images.npy"

    strokemode: StrokeMode

    images: list[np.ndarray]
    signals: list[list]

    patchified_images: torch.Tensor
    patches_padding_masks: torch.Tensor
    signals_as_tensor: list[torch.Tensor]

    batchified_patchified_images: list[torch.Tensor]
    batchified_patches_padding_masks: list[torch.Tensor]
    batchified_sequences: list[torch.Tensor]
    coordinate_to_predict: list[torch.Tensor]

    patchificator: Patchificator
    normalize_pixel_values: bool
    normalize_coordinate_sequences: bool

    size: int

    patches_dim: tuple
    target_image_shape: tuple

    window_size: int
    lstm_forecast: int

    def __init__(self, brush_root, patches_dim: tuple, save_to_file: bool = True, strokemode: StrokeMode = StrokeMode.STROKES,
                 normalize_pixel_values: bool = True, normalize_coordinate_sequences: bool = True, window_size: int = None, lstm_forecast: int = None):
        self.brush_root = brush_root
        self.save_to_file = save_to_file
        self.strokemode = strokemode

        self.signals = []
        self.images = []

        self.coordinate_to_predict = None
        self.lstm_forecast = lstm_forecast

        self.patches_dim = patches_dim
        self.window_size = window_size

        self.normalize_pixel_values = normalize_pixel_values
        self.normalize_coordinate_sequences = normalize_coordinate_sequences
        
        self.load_samples()

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
    
    def load_samples(self):
        """Function that tries to retrieve samples form single file. If it cannot, retrieve samples from individual files on disk"""
        try:
            self.load_images_labels()
        except Exception as e:
            logger.log(LogChannels.DATA, f"Impossible to retrieve single file, retrieving samples individually")
            self.search_load_samples()
        finally:
            self.size = len(self.images)
            logger.log(LogChannels.DATA,f"Loaded {self.size} data points")

    def search_load_samples(self):
        """This function loads the samples forn disk, creatin the offline image in the process"""
        raw_root = os.path.join(self.brush_root, self.RAW_DIR)
        try:
            writer_ids = os.listdir(raw_root)
            total_writers = len(writer_ids)
            logger.log(LogChannels.DATA, f"Loading {total_writers} writers")
        except Exception as e:
            logger.log(LogChannels.DATA, f"Impossible to read root folder {self.brush_root}")
            raise e
        
        i = 0
        for writer_id in writer_ids:
            i += 1
            if i >= 2:
                break
            writer_path = os.path.join(raw_root, writer_id)
            #Each drawin is present in three examplaries: n, n_resample20 and n_resample25
            #base dataloader selects default (10ms)
            drawing_ids = [name for name in os.listdir(writer_path) if "_" not in name and ".npy" not in name]

            logger.log(LogChannels.DATA, f"{i}/{total_writers}: Detected {len(drawing_ids)} drawings")

            for drawing_id in drawing_ids:
                signal_path = os.path.join(writer_path, drawing_id)
                sentence, signal, char_label = self.load_signal(signal_path)

                if self.strokemode is StrokeMode.SUBSTROKES:
                    signals = self.stroke_signal_to_substroke(signal, align_substrokes=True, left_padding=5)
                else:
                    signals = [signal]

                #If window size, each signal has to be passed through the apply window function that splits signals that are too long
                if self.window_size:
                    window_signals = []
                    for raw_signal in signals:
                        window_signals.extend(self.apply_window_to_signal(raw_signal))
                    signals = window_signals

                self.signals.extend(signals)
                self.images.extend([ImageHelper.create_image(signal) for signal in signals])

        if self.save_to_file:
            self.save_images_labels_single_file()

    def apply_window_to_signal(self, signal: list[tuple]):
        """Cut the signal into sub-signals
        Args
        -----
            signal: list[tuple] - The signal, as a list of tuples (x,y,penUp)
        """
        if len(signal) <= self.window_size:
            return [signal]
        
        signal_numbers = int(math.ceil(len(signal)/self.window_size))
        
        logger.log(LogChannels.DATA, f"Cutting signal of len {len(signal)} in {signal_numbers}")

        cut_signals = []

        for i in range(signal_numbers):
            cut_signals.append(signal[i*self.window_size : (i+1)*self.window_size])

        logger.log(LogChannels.DATA, f"returning {len(cut_signals)} signals of len {[len(sig) for sig in cut_signals]}")

        return cut_signals

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

            if self.lstm_forecast is not None:
                if len_of_signal < self.lstm_forecast + 1:
                    continue
                for j in range(0, len_of_signal - self.lstm_forecast - 1):
                    subsequence = signal[j : self.lstm_forecast + j]

                    signals_to_predict.append(subsequence)
                    coordinates_to_predict.append(signal[self.lstm_forecast + j])
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
        whole_multiple_width = int(math.ceil(size[0] / patch_dim[0]))
        whole_multiple_height = int(math.ceil(size[1] / patch_dim[1]))

        return (patch_dim[0] * whole_multiple_width, patch_dim[1] * whole_multiple_height)
        
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

    def transform_to_batch(self):
        """Transform the data to homogeneous tensors"""
        self.images_to_tensor()
        self.sequences_to_tensor()
        
        self.extract_all_predictable_from_tensor()

    def load_signal(self, filepath: str) -> tuple[str, list, list]:
        """Load an online sinal from a filepath
        Args
        -----
            Filepath: The name of the file to load from
            
        Returns
        -----
            - str: Written sentence as string
            - list: Signal of x, y, penUp
            - list: List of one-hot vectors with same length as signal identifying charachter of point
        """
        with open(filepath, 'rb') as f:
            [sentence, signal, label] = pickle.load(f)

        signal = (np.rint(signal)).astype(int)

        return sentence, signal, label


    @classmethod
    def stroke_signal_to_substroke(cls, signal: list, align_substrokes: bool = True, left_padding: int = 5) -> list[list]:
        """Provided an online signal, return a list of its substrokes separated by PENUP
        
        args
        -----
            signal: list - The signal as (x,y,penup) array
            align_substrokes: bool - Whether to align the substrokes at the left
            h_padding: 
        """
        substrokes = []

        start_idx = 0
        end_idx = 0

        #Cut the stroke each time the pen is up to obtain substrokes
        for x, y, penup in signal:
            end_idx += 1
            if penup:
                substroke = signal[start_idx:end_idx]

                # If a substroke starts at x=40, it can be re-placed at 0. 
                # This is be done via substracting the min X value to each value of the signal
                if align_substrokes:
                    min_x = min(substroke[:, 0])
                    substroke[:, 0] -= min_x

                # Add a left padding
                if left_padding > 0:
                    substroke[:, 0] += left_padding
                
                substrokes.append(substroke)
                start_idx = end_idx

        return substrokes

    def load_images_labels(self):
        """Load all images and labels at once"""
        strokemode_folder = self.GROUPED_STROKES_DIR if self.strokemode is StrokeMode.STROKES else self.GROUPED_SUBSTROKES_DIR
        
        image_path = os.path.join(self.brush_root, strokemode_folder, self.FOLDER_IMAGES, self.FILE_IMAGES)
        signal_path = os.path.join(self.brush_root, strokemode_folder, self.FOLDER_SIGNALS, self.FILE_SIGNALS)
        logger.log(LogChannels.DATA, f"Trying to retrive files at {image_path}, {signal_path}")

        with open(image_path, "rb") as f:
            images = np.load(f, allow_pickle=True)
        with open(signal_path, "rb") as f:
            signals = np.load(f, allow_pickle=True)
        
        self.images = images
        self.signals = signals

    def save_images_labels_single_file(self):
        """Save all images and labels at once"""
        strokemode_folder = self.GROUPED_STROKES_DIR if self.strokemode is StrokeMode.STROKES else self.GROUPED_SUBSTROKES_DIR
        
        image_path = os.path.join(self.brush_root, strokemode_folder, self.FOLDER_IMAGES)
        signal_path = os.path.join(self.brush_root, strokemode_folder, self.FOLDER_SIGNALS)

        Path(image_path).mkdir(parents=True, exist_ok=True)
        Path(signal_path).mkdir(parents=True, exist_ok=True)

        image_folder = os.path.join(image_path, self.FILE_IMAGES)
        signal_folder = os.path.join(signal_path, self.FILE_SIGNALS)

        logger.log(LogChannels.DATA, f"Saving images to {image_path}, signals to {signal_path}")

        images_to_save = np.array(self.images, dtype="object")
        signale_to_save = np.array(self.signals, dtype="object")

        with open(image_folder, "wb") as f:
            np.save(f, images_to_save)
        with open(signal_folder, "wb") as f:
            np.save(f, signale_to_save)