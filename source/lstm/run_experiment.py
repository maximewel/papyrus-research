
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.lstm.lstm_loop import do_training
from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from source.model.blocks.hw_lstm import HwLstm
from source.logging.log import logger, LogChannels
from datetime import datetime
from source.model.blocks.constants.files import *
from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset

import torch

BATCH_SIZE = 256

LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 12
LSTM_INPUT_SIZE = 2

PATCHES_DIM = (1,1)

N_EPOCHS = 50
LR = 0.001

TARGET_IMAGE_SIZE = (100, 150)

NORMALIZE_PIXEL_VALUES = False
NORMALIZE_COORDS = False

TRAIN_SIZE = 0.8

USE_BRUSH = True

from source.model.blocks.constants.device_helper import device

if __name__ == "__main__":
    #Set logging
    # for channel in LogChannels:
    #     logger.add_log_channel(channel)
    #logger.add_log_channel(LogChannels.TRAINING)
    #logger.add_log_channel(LogChannels.DEBUG)
    #logger.add_log_channel(LogChannels.LOSSES)
    # logger.add_log_channel(LogChannels.INIT)
    # logger.add_log_channel(LogChannels.PARAMS)
    # logger.add_log_channel(LogChannels.DATA)

    #print(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''})")

    # Init data
    if USE_BRUSH:
        datasource = BrushDataset(brush_root=BRUSH_ROOT, separate_strokes=True, image_max_shape=TARGET_IMAGE_SIZE, save_to_file=False)
    else:
        datasource = UnipenDataset(unipen_root=UNIPEN_ROOT, separate_strokes=True, image_max_shape=TARGET_IMAGE_SIZE)
    
    #Separate signal in appropriate train, test, split
    train_signals, test_signals = train_test_split(datasource.signals, train_size=TRAIN_SIZE)

    do_pin_memory = device != 'cpu'

    image_max_shape = tuple(reversed(datasource.signals_max_shape))

    train_dataset = HandWrittingDataset(train_signals, image_max_shape, PATCHES_DIM, NORMALIZE_PIXEL_VALUES, NORMALIZE_COORDS, True)
    train_dataset.prepare_training_data()
    test_dataset = HandWrittingDataset(test_signals, image_max_shape, PATCHES_DIM, NORMALIZE_PIXEL_VALUES, NORMALIZE_COORDS, True)
    test_dataset.prepare_training_data()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())

    logger.log(LogChannels.INIT, f"Using n° points to predict: Train={len(train_dataset)}, Test={len(test_dataset)}")

    logger.log(LogChannels.INIT, f"Loading {len(train_loader)} batches as train, {len(test_loader)} batches as test")

    #Create model
    model = HwLstm(input_size=2, hidden_size=LSTM_HIDDEN_DIM, num_layers=LSTM_LAYERS)
    
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(LogChannels.PARAMS, f"Number of model parameters: {n_model_params}")

    return_figures = None
    
    #Start trainings
    try:
        return_figures = do_training(model, train_loader, test_loader, device, N_EPOCHS, LR)
    except Exception as e:
        print(f"Encountered exception while training model: {e}")
        raise e
    except KeyboardInterrupt:
        print(f"Training interrupted")
    finally:
        date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, LSTM_FOLDER, f"{date}")
        os.makedirs(folderPath, exist_ok=True)

        filepath = os.path.join(folderPath, MODEL_FILENAME)
        print(f"Saving model to: {filepath}")
        torch.save(model, filepath)

        if return_figures is not None:
            for fig_name, figure in return_figures:
                filepath = os.path.join(folderPath, f"{fig_name}.png")
                print(f"Saving figure {fig_name} to {filepath}")
                figure.savefig(filepath)