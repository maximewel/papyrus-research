
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.loops import do_training
from source.data_management.brush.brush_dataset import BrushDataset, StrokeMode
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels
from datetime import datetime
from source.model.blocks.constants.files import *

import torch

ENCODER_HEADS = 6
DECODER_HEADS = 6

ENCODER_LAYERS = 6
DECODER_LAYERS = 6

DROPOUT_RATIO = 0.1

BATCH_SIZE = 32
PATCHES_DIM = (6, 6)
EMBEDDING_DIMS = 18

NORMALIZE_COORDS = True
NORMALIZE_PIXEL_VALUES = True

TRAIN_SIZE = 0.7
TEST_SIZE = 0.2

LR = 0.001
N_EPOCHS = 1

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
    logger.add_log_channel(LogChannels.DATA)

    #print(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''})")

    # Init data management
    dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=PATCHES_DIM, save_to_file=False, 
                           strokemode=StrokeMode.SUBSTROKES, window_size=50,
                           normalize_pixel_values=NORMALIZE_PIXEL_VALUES, normalize_coordinate_sequences=NORMALIZE_COORDS)
    dataset.transform_to_batch()

    train_size = int(TRAIN_SIZE* len(dataset))
    test_size = int(TEST_SIZE * len(dataset))
    validation_size = len(dataset) - (train_size + test_size)

    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

    do_pin_memory = device != 'cpu'

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory)

    logger.log(LogChannels.INIT, f"Using n° points to predict: Train={len(train_dataset)}, Test={len(test_dataset)}, Valid={len(validation_dataset)}")

    logger.log(LogChannels.INIT, f"Loading {len(train_loader)} sub-strokes batches as train, {len(test_loader)} sub-strokes batches as test")

    #Init the transformer model
    model = HwTransformer(False, False, hidden_dim=EMBEDDING_DIMS, enc_dec_dropout_ratio=DROPOUT_RATIO,
                          n_encoder_layers=ENCODER_LAYERS, n_encoder_heads=ENCODER_HEADS,
                          n_decoder_layers=DECODER_LAYERS, n_decoder_heads=DECODER_HEADS,
                          encoder_patch_dimension=PATCHES_DIM, fixed_size_image_dimension=dataset.target_image_shape)
        
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(LogChannels.PARAMS, f"Number of model parameters: {n_model_params}")
    
    #Start training
    try:
        do_training(model, train_loader, test_loader, device, N_EPOCHS, LR)
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