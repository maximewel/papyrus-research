
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.loops import do_training
from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.model.blocks.hw_lstm import HwLstm
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels
from datetime import datetime
from source.model.blocks.constants.files import *
from source.model.blocks.helper.id_card_creator import IdCardCreator
from source.model.blocks.constants.device_helper import device
from source.criterions.losses_weights import LossesWeights

import torch

ENCODER_HEADS = 8
DECODER_HEADS = 8

ENCODER_LAYERS = 10
DECODER_LAYERS = 10

AUTOREGRESS_TARGET_LEN = 500

DROPOUT_RATIO = 0.0
BATCH_SIZE = 8

PATCHES_DIM = (8, 8)
EMBEDDING_DIMS = 256

NORMALIZE_COORDS = True
NORMALIZE_PIXEL_VALUES = False

USE_PREDICTION_TOKEN = False
USE_LSTM = False
LSTM_MODEL_PATH = "2024-10-24 22-39-02"

TRAIN_SIZE = 0.7
TEST_SIZE = 0.2

LR = 0.001
N_EPOCHS = 5

USE_BRUSH = True

WEIGHT_EOS = 1
WEIGHT_COORD = 5
WEIGHT_SKELETON = 2

if __name__ == "__main__":
    #Set logging
    # for channel in LogChannels:
    #     logger.add_log_channel(channel)
    #logger.add_log_channel(LogChannels.TRAINING)
    #logger.add_log_channel(LogChannels.DEBUG)
    # logger.add_log_channel(LogChannels.INIT)
    #logger.add_log_channel(LogChannels.PARAMS)
    #logger.add_log_channel(LogChannels.DIMENSIONS)
    # logger.add_log_channel(LogChannels.DATA)
    # logger.add_log_channel(LogChannels.PADDING)
    #logger.add_log_channel(LogChannels.MASKS)

    logger.add_log_channel(LogChannels.LOSSES)
    logger.add_log_channel(LogChannels.LOSS_DETAILED)

    #print(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''})")

    if USE_LSTM:
        #Load pre-trained LSTM model
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, LSTM_FOLDER, LSTM_MODEL_PATH)
        filepath = os.path.join(folderPath, MODEL_FILENAME)
        print(f"Loading LSTM model from: {filepath}")
        lstm_model: HwLstm = torch.load(filepath)
        #Freeze model as we have a pre-trained LSTM model that doesnt need to learn in this step
        for param in lstm_model.parameters():
            param.requires_grad = False
    else:
        lstm_model = None

    #Init data management
    if USE_BRUSH:
        dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=PATCHES_DIM, save_to_file=False, 
                            strokemode=True, window_size=0,
                            normalize_pixel_values=NORMALIZE_PIXEL_VALUES, normalize_coordinate_sequences=NORMALIZE_COORDS)
    else:
        dataset = UnipenDataset(unipen_root=UNIPEN_ROOT, patches_dim=PATCHES_DIM,
                            strokemode=True, window_size=0,
                            normalize_pixel_values=NORMALIZE_PIXEL_VALUES, normalize_coordinate_sequences=NORMALIZE_COORDS,
                            samples_to_take=5000)

    dataset.transform_to_batch()

    train_size = int(TRAIN_SIZE * len(dataset))
    test_size = int(TEST_SIZE * len(dataset))
    validation_size = len(dataset) - (train_size + test_size)

    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

    do_pin_memory = device != 'cpu'

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=dataset.get_collate_function())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=dataset.get_collate_function())

    losses_weights = LossesWeights(WEIGHT_EOS, WEIGHT_COORD, WEIGHT_SKELETON)

    logger.log(LogChannels.INIT, f"Using n° points to predict: Train={len(train_dataset)}, Test={len(test_dataset)}, Valid={len(validation_dataset)}")

    logger.log(LogChannels.INIT, f"Loading {len(train_loader)} sub-strokes batches as train, {len(test_loader)} sub-strokes batches as test")

    #Init the transformer model
    model = HwTransformer(use_prediction_token=USE_PREDICTION_TOKEN, hidden_dim=EMBEDDING_DIMS,
                          use_lstm=USE_LSTM, lstm_module=lstm_model,
                          n_encoder_layers=ENCODER_LAYERS, n_encoder_heads=ENCODER_HEADS, enc_dec_dropout_ratio=DROPOUT_RATIO,
                          n_decoder_layers=DECODER_LAYERS, n_decoder_heads=DECODER_HEADS,
                          encoder_patch_dimension=PATCHES_DIM, fixed_size_image_dimension=dataset.target_image_shape,
                          autoregressive_target_seq_len=AUTOREGRESS_TARGET_LEN)

    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(LogChannels.PARAMS, f"Number of model parameters: {n_model_params}")
    
    #Start training
    try:
        return_figures = do_training(model, train_loader, test_loader, device, N_EPOCHS, LR, NORMALIZE_COORDS, dataset.target_image_shape, losses_weights)
    except Exception as e:
        print(f"Encountered exception while training model: {e}")
        raise e
    except KeyboardInterrupt:
        print(f"Training interrupted")
    finally:
        date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, f"{date}")
        os.makedirs(folderPath, exist_ok=True)

        filepath = os.path.join(folderPath, MODEL_FILENAME)
        print(f"Saving model to: {filepath}")
        torch.save(model, filepath)

        if return_figures is not None:
            for fig_name, figure in return_figures:
                filepath = os.path.join(folderPath, f"{fig_name}.png")
                print(f"Saving figure {fig_name} to {filepath}")
                figure.savefig(filepath)
        
        filepath = os.path.join(folderPath, ID_CARD_FILE)
        id_card = IdCardCreator.create_transfo_id_card(USE_BRUSH, LR, N_EPOCHS, BATCH_SIZE, ENCODER_LAYERS, DECODER_LAYERS, ENCODER_HEADS, DECODER_HEADS, DROPOUT_RATIO, AUTOREGRESS_TARGET_LEN, PATCHES_DIM, EMBEDDING_DIMS, USE_PREDICTION_TOKEN, USE_LSTM, LSTM_MODEL_PATH)
        with open(filepath, "w+") as f:
            f.write(id_card)