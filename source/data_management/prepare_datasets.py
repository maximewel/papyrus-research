import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.blocks.constants.files import *
from source.model.blocks.constants.datasets_library import *
from source.logging.log import logger, LogChannels
import random

PATCHES_DIM = (16, 16)
LSTM_MODE = False
TARGET_IMAGE_SHAPE = (100, 100)
NORMALIZE__COORDS = False
APPLY_GAUSSIAN_DATA_AUGMENT = True

TEST_DATASET_FOLDER_NAME = "TEST_100_100_BRUSH"

S_RATIO = 0.2
M_RATIO = 0.5
L_RATIO = 1.0

TRAIN_RATIO = 0.8

def test_prepare_dataset():
    datasource_from_file = BrushDataset(brush_root=BRUSH_ROOT, separate_strokes=True, image_max_shape=(100,100))
    print(F"Loading Done, number of signals: {len(datasource_from_file.signals)}")

    signals = random.choices(datasource_from_file.signals, k=int(0.001*len(datasource_from_file.signals)))

    print(F"Saving dataset of {len(signals)} signals on disk at {TEST_DATASET_FOLDER_NAME}")
    HandWrittingDataset.prepare_and_save_training_data(signals, TEST_DATASET_FOLDER_NAME, PATCHES_DIM, LSTM_MODE, 
                                                       TARGET_IMAGE_SHAPE, NORMALIZE__COORDS, APPLY_GAUSSIAN_DATA_AUGMENT)

def test_load_dataset():
    print(f"Creating dataset...")
    dataset = HandWrittingDataset(TEST_DATASET_FOLDER_NAME, LSTM_MODE)

    first_data = dataset[0]

    import random
    r_ind = random.randrange(0, len(dataset))
    r_data = dataset[r_ind]
    for i in range(10):
        image, patchified_image, patchified_mask, subsequence, label = dataset[r_ind + i]
        print(f"Seq, label {i}: \n{subsequence}\n{label}\n")

def prepare_datasets():
    print("Preparing BRUSH medium...")
    brush_datasource = BrushDataset(brush_root=BRUSH_ROOT, separate_strokes=True, image_max_shape=(100,100))
    train, test = train_test_signals(brush_datasource.signals, M_RATIO)

    prepare_dataset(train, BRUSH_100_100_TRAIN_M, apply_gaussian_augment=False)
    prepare_dataset(test, BRUSH_100_100_TEST_M, apply_gaussian_augment=False)
    prepare_dataset(train, BRUSH_100_100_TRAIN_M_AUGMENTED, apply_gaussian_augment=True)
    prepare_dataset(test, BRUSH_100_100_TEST_M_AUGMENTED, apply_gaussian_augment=True)

    print("Preparing UNIPEN medium...")
    unipen_datasource = BrushDataset(brush_root=BRUSH_ROOT, separate_strokes=True, image_max_shape=(100,100))
    train, test = train_test_signals(unipen_datasource.signals, M_RATIO)

    prepare_dataset(train, UNIPEN_100_100_TRAIN_M, apply_gaussian_augment=False)
    prepare_dataset(test, UNIPEN_100_100_TEST_M, apply_gaussian_augment=False)
    prepare_dataset(train, UNIPEN_100_100_TRAIN_M_AUGMENTED, apply_gaussian_augment=True)
    prepare_dataset(test, UNIPEN_100_100_TEST_M_AUGMENTED, apply_gaussian_augment=True)

def train_test_signals(signals: list, ratio: float):
    signals_sampled = random.choices(signals, k=int(ratio*len(signals)))

    split_index = int(len(signals_sampled) * TRAIN_RATIO)

    train = signals_sampled[:split_index]
    test = signals_sampled[split_index:]

    return train, test


def prepare_dataset(signals, dataset_folder, apply_gaussian_augment):
    print(F"Saving dataset of {len(signals)} signals on disk at {dataset_folder}")
    HandWrittingDataset.prepare_and_save_training_data(signals, dataset_folder, PATCHES_DIM, LSTM_MODE, TARGET_IMAGE_SHAPE, NORMALIZE__COORDS, apply_gaussian_augment)
    
def prepare_inference_dataset():
    pass

if __name__ == "__main__":
    logger.add_log_channel(LogChannels.DATA)

    prepare_datasets()

    #prepare_inference_dataset()