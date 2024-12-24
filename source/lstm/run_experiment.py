
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.lstm.lstm_loop import do_training
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from source.model.blocks.hw_lstm import HwLstm
from source.logging.log import logger, LogChannels
from datetime import datetime
from source.model.blocks.constants.files import *
from source.data_management.common.handwritting_dataset import HandWrittingDataset

from source.model.blocks.constants.datasets_library import *

import torch

BATCH_SIZE = 256

LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 6
LSTM_INPUT_SIZE = 2

PATCHES_DIM = (1,1)

N_EPOCHS = 10
LR = 0.001

TARGET_IMAGE_SIZE = (96, 96)

NORMALIZE_COORDS = False

TRAIN_DATASET = BRUSH_96_96_TRAIN_S_AUGMENTED
TEST_DATASET = BRUSH_96_96_TEST_S_AUGMENTED

OUTPUT_NAME = "lstm_96.96_augmented"

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
    
    #Separate signal in appropriate train, test, split
    do_pin_memory = device != 'cpu'

    train_dataset = HandWrittingDataset(TRAIN_DATASET, lstm_mode=True)
    test_dataset = HandWrittingDataset(TEST_DATASET, lstm_mode=True)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())

    logger.log(LogChannels.INIT, f"Using n° points to predict: Train={len(train_dataset)}, Test={len(test_dataset)}")

    logger.log(LogChannels.INIT, f"Loading {len(train_loader)} batches as train, {len(test_loader)} batches as test")

    #Create model
    model = HwLstm(input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_DIM, num_layers=LSTM_LAYERS)
    
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
        folder_name = OUTPUT_NAME if OUTPUT_NAME else datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, LSTM_FOLDER, f"{folder_name}")
        os.makedirs(folderPath, exist_ok=True)

        filepath = os.path.join(folderPath, MODEL_FILENAME)
        print(f"Saving model to: {filepath}")
        torch.save(model, filepath)

        if return_figures is not None:
            for fig_name, figure in return_figures:
                filepath = os.path.join(folderPath, f"{fig_name}.png")
                print(f"Saving figure {fig_name} to {filepath}")
                figure.savefig(filepath)