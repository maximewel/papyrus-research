import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset, GaussianAugmentationMode
from source.model.blocks.constants.files import *
from source.model.blocks.constants.datasets_library import *
from source.logging.log import logger, LogChannels
import random

PATCHES_DIM = (16, 16)
LSTM_MODE = False
TARGET_IMAGE_SHAPE = (96, 96)
NORMALIZE__COORDS = False
S_RATIO = 0.2
TRAIN_RATIO = 0.80

WORKING_SIGNAL_LOW_THRESHOLD = 15
VALID_SIGNAL_MEDIUM_THRESHOLD = 50

def prepare_datasets():
    print("Preparing small datasets...")
    brush_datasource = BrushDataset(brush_root=BRUSH_ROOT, separate_strokes=True, image_max_shape=TARGET_IMAGE_SHAPE)

    train, test, valid = train_test_valid_signals(brush_datasource.signals)

    prepare_dataset(train, BRUSH_96_96_TRAIN_M_MIXED, GaussianAugmentationMode.MIXED_AUGMENTED_NON_AUGMENTED)
    prepare_dataset(test, BRUSH_96_96_TEST_M_MIXED, GaussianAugmentationMode.MIXED_AUGMENTED_NON_AUGMENTED)
    prepare_dataset(valid, BRUSH_96_96_VALID_M, GaussianAugmentationMode.UNAUGMENTED, only_last=True)

def train_test_valid_signals(signals: list) -> tuple[list, list, list]:
    l = int(S_RATIO*len(signals))

    #Cut very little information signals
    signals = list(filter(lambda sig: len(sig) >= WORKING_SIGNAL_LOW_THRESHOLD, signals))
    random.shuffle(signals)

    signals_train_test = signals[:l]
    signals_valid = signals[l:]

    split_index = int(len(signals_train_test) * TRAIN_RATIO)

    train = signals_train_test[:split_index]
    test = signals_train_test[split_index:]

    valid = random.choices([signal for signal in signals_valid if len(signal) <= VALID_SIGNAL_MEDIUM_THRESHOLD], k=1000)
    valid += random.choices([signal for signal in signals_valid if len(signal) > VALID_SIGNAL_MEDIUM_THRESHOLD], k=1000)

    return train, test, valid

def prepare_dataset(signals, dataset_folder, gaussian_augment_mode, only_last = False):
    print(F"Saving dataset of {len(signals)} signals on disk at {dataset_folder}")
    HandWrittingDataset.prepare_and_save_training_data(signals, dataset_folder, PATCHES_DIM, LSTM_MODE, TARGET_IMAGE_SHAPE, NORMALIZE__COORDS, gaussian_augment_mode, only_last)

if __name__ == "__main__":
    logger.add_log_channel(LogChannels.DATA)

    prepare_datasets()