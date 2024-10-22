import os
import numpy as np
import math
import pickle
from pathlib import Path

from source.logging.log import logger, LogChannels
from source.model.blocks.constants.sequence_to_image import ImageHelper

from source.data_management.common.stroked_handwriting_dataset import StrokedHandwrittingDataset

class BrushDataset(StrokedHandwrittingDataset):
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

    def __init__(self, brush_root, patches_dim: tuple, save_to_file: bool = True, strokemode: bool = True,
                 normalize_pixel_values: bool = True, normalize_coordinate_sequences: bool = True, window_size: int = None, lstm_forecast_length: int = None):
        self.brush_root = brush_root
        self.save_to_file = save_to_file

        super().__init__(patches_dim, strokemode, normalize_pixel_values, normalize_coordinate_sequences, window_size, lstm_forecast_length)

    def _load_data(self):
        """Function that tries to retrieve samples form single file. If it cannot, retrieve samples from individual files on disk"""
        try:
            self.load_from_memory()
        except Exception as e:
            logger.log(LogChannels.DATA, f"Impossible to retrieve single file, retrieving samples individually")
            self.load_raw_data()

    def load_from_memory(self):
        """Load all images and labels at once"""
        strokemode_folder = self.GROUPED_STROKES_DIR if self.separate_strokes else self.GROUPED_SUBSTROKES_DIR
        
        image_path = os.path.join(self.brush_root, strokemode_folder, self.FOLDER_IMAGES, self.FILE_IMAGES)
        signal_path = os.path.join(self.brush_root, strokemode_folder, self.FOLDER_SIGNALS, self.FILE_SIGNALS)
        logger.log(LogChannels.DATA, f"Trying to retrive files at {image_path}, {signal_path}")

        with open(image_path, "rb") as f:
            images = np.load(f, allow_pickle=True)
        with open(signal_path, "rb") as f:
            signals = np.load(f, allow_pickle=True)
        
        self.images = images
        self.signals = signals

    def load_raw_data(self):
        """This function loads the samples from disk, creating the offline image in the process"""
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
                self.signals.append(signal)

        self.apply_all_preprocess_to_signals()

        if self.samples_to_take is not None:
            self.take_samples_of_dataset(self.samples_to_take)

        self.build_images()

        if self.save_to_file:
            self.save_images_labels_single_file()

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

    def save_images_labels_single_file(self):
        """Save all images and labels at once"""
        strokemode_folder = self.GROUPED_STROKES_DIR if self.separate_strokes else self.GROUPED_SUBSTROKES_DIR
        
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