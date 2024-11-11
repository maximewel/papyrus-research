import os
import numpy as np
import pickle
from pathlib import Path

from source.logging.log import logger, LogChannels
from source.model.blocks.constants.files import *

from source.data_management.common.stroke_handwriting_dataset import StrokeHandwrittingDataset

class BrushDataset(StrokeHandwrittingDataset):
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

    restrict_id: int | None

    def __init__(self, brush_root, separate_strokes: bool = True, save_to_file: bool = True, 
                 image_max_shape: tuple[int, int] = None, window_size: int = None, restrict_id: int|None = None):
        self.brush_root = brush_root
        self.save_to_file = save_to_file
        self.window_size = window_size

        self.restrict_id = restrict_id

        super().__init__(separate_strokes, image_max_shape)

    def _load_data(self):
        """Function that tries to retrieve samples form single file. If it cannot, retrieve samples from individual files on disk"""
        try:
            self.load_from_memory()
        except Exception as e:
            logger.log(LogChannels.DATA, f"Impossible to retrieve single file, retrieving samples individually")
            self.load_raw_data()

    def load_from_memory(self):
        """Load all images and labels at once"""
        strokemode_folder = GROUPED_ORIGINAL_DIR if self.separate_strokes else GROUPED_STROKES_DIR
        signal_path = os.path.join(self.brush_root, strokemode_folder, FOLDER_SIGNALS, FILE_SIGNALS)
        logger.log(LogChannels.DATA, f"Trying to retrieve BRUSH files at {signal_path}")

        with open(signal_path, "rb") as f:
            signals = np.load(f, allow_pickle=True)
        
        self.signals = signals

    def load_raw_data(self):
        """This function loads the samples from disk, creating the offline image in the process"""
        raw_root = os.path.join(self.brush_root, RAW_DIR)
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
            if self.restrict_id is not None and i != self.restrict_id:
                continue
            writer_path = os.path.join(raw_root, writer_id)
            #Each drawin is present in three examplaries: n, n_resample20 and n_resample25
            #base dataloader selects default (10ms)
            drawing_ids = [name for name in os.listdir(writer_path) if "_" not in name and ".npy" not in name]

            logger.log(LogChannels.DATA, f"{i}/{total_writers}: Detected {len(drawing_ids)} drawings")

            for drawing_id in drawing_ids:
                signal_path = os.path.join(writer_path, drawing_id)
                sentence, signal, char_label = self.load_signal(signal_path)
                self.signals.append(signal)

        if self.save_to_file:
            self.save_signals_single_file()

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

    def save_signals_single_file(self):
        """Save all images and labels at once"""
        strokemode_folder = GROUPED_ORIGINAL_DIR if self.separate_strokes else GROUPED_STROKES_DIR
        
        signal_path = os.path.join(self.brush_root, strokemode_folder, FOLDER_SIGNALS)

        Path(signal_path).mkdir(parents=True, exist_ok=True)

        signal_folder = os.path.join(signal_path, FILE_SIGNALS)

        logger.log(LogChannels.DATA, f"Saving signals to {signal_path}")

        signale_to_save = np.array(self.signals, dtype="object")

        with open(signal_folder, "wb") as f:
            np.save(f, signale_to_save)