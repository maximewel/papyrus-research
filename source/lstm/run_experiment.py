
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.lstm.lstm_loop import do_training
from source.data_management.brush.brush_dataset import BrushDataset, StrokeMode
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from source.model.blocks.hw_lstm import HwLstm
from source.logging.log import logger, LogChannels
from datetime import datetime
from source.model.blocks.constants.files import *

import torch

BATCH_SIZE = 256

from source.model.blocks.constants.device_helper import device

if __name__ == "__main__":
    #Set logging
    # for channel in LogChannels:
    #     logger.add_log_channel(channel)
    #logger.add_log_channel(LogChannels.TRAINING)
    #logger.add_log_channel(LogChannels.DEBUG)
    #logger.add_log_channel(LogChannels.LOSSES)
    logger.add_log_channel(LogChannels.INIT)
    logger.add_log_channel(LogChannels.PARAMS)

    #print(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''})")

    # Init data
    dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=(1,1), save_to_file=False, strokemode=StrokeMode.SUBSTROKES, lstm_forecast=25)
    dataset.transform_to_batch()

    train_size = int(0.7 * len(dataset))
    test_size = int(0.2 * len(dataset))
    validation_size = len(dataset) - (train_size + test_size)

    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

    do_pin_memory = device != 'cpu'

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory)

    logger.log(LogChannels.INIT, f"Using n° points to predict: Train={len(train_dataset)}, Test={len(test_dataset)}, Valid={len(validation_dataset)}")

    logger.log(LogChannels.INIT, f"Loading {len(train_loader)} sub-strokes batches as train, {len(test_loader)} sub-strokes batches as test")

    #Create model
    model = HwLstm(input_size=2, hidden_size=10, num_layers=8)
    logger.log(LogChannels.INIT, f"Loaded {len(dataset.signals_as_tensor)} sub-strokes")
    
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(LogChannels.PARAMS, f"Number of model parameters: {n_model_params}")

    logger.log(LogChannels.INIT, f"Loaded {len(dataset.signals_as_tensor)} sub-strokes")
    
    #Start trainings
    try:
        do_training(model, train_loader, test_loader, device)
    except Exception as e:
        print(f"Encountered exception while training model: {e}")
        raise e
    except KeyboardInterrupt:
        print(f"Training interrupted")
    finally:
        date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, f"{date}")
        os.makedirs(folderPath, exist_ok=True)

        filepath = os.path.join(folderPath, MODEL_FILENAME)
        print(f"Saving model to: {filepath}")
        torch.save(model, filepath)