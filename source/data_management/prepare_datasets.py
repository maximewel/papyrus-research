import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.blocks.constants.files import *
from source.logging.log import logger, LogChannels

DATASET_FOLDER_NAME = "BRUSH_100.100_FULL"
PATCHES_DIM = (16, 16)
LSTM_MODE = False
TARGET_IMAGE_SHAPE = (100, 100)
NORMALIZE__COORDS = False
APPLY_GAUSSIAN_DATA_AUGMENT = True

def prepare_dataset():
    datasource_from_file = BrushDataset(brush_root=BRUSH_ROOT, separate_strokes=True, image_max_shape=(100,100))
    print(F"Loading Done, number of signals: {len(datasource_from_file.signals)}")

    import random
    signals = random.choices(datasource_from_file.signals, k=int(0.001*len(datasource_from_file.signals)))

    print(F"Saving dataset of {len(signals)} signals on disk at {DATASET_FOLDER_NAME}")
    HandWrittingDataset.prepare_and_save_training_data(signals, DATASET_FOLDER_NAME, PATCHES_DIM, LSTM_MODE, 
                                                       TARGET_IMAGE_SHAPE, NORMALIZE__COORDS, APPLY_GAUSSIAN_DATA_AUGMENT)

def load_dataset():
    print(f"Creating dataset...")
    dataset = HandWrittingDataset(DATASET_FOLDER_NAME, LSTM_MODE)

    first_data = dataset[0]

    import random
    r_ind = random.randrange(0, len(dataset))
    r_data = dataset[r_ind]
    for i in range(10):
        image, patchified_image, patchified_mask, subsequence, label = dataset[r_ind + i]
        print(f"Seq, label {i}: \n{subsequence}\n{label}\n")
    
if __name__ == "__main__":
    logger.add_log_channel(LogChannels.DATA)

    prepare_dataset()

    load_dataset()