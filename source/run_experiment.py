
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.loops import do_training
from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.blocks.hw_lstm import HwLstm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels
from datetime import datetime
from source.model.blocks.constants.files import *
from source.model.blocks.constants.datasets_library import *
from source.model.blocks.helper.id_card_creator import IdCardCreator
from source.model.blocks.constants.device_helper import device
from source.criterions.losses_weights import LossesWeights

import pickle
import torch

ENCODER_HEADS = 8
DECODER_HEADS = 8

ENCODER_LAYERS = 12
DECODER_LAYERS = 12

AUTOREGRESS_TARGET_LEN = 500

MAKE_POSITIONAL_ENCODING_LEARNABLE = False

DROPOUT_RATIO = 0.1
BATCH_SIZE = 8

PATCHES_DIM = (16, 16)
EMBEDDING_DIMS = 256

NORMALIZE_COORDS = False

TRAIN_DATASET_NAME = ""
TEST_DATASET_NAME = ""

USE_PREDICTION_TOKEN = False
USE_LSTM = False
LSTM_MODEL_PATH = "2024-10-24 22-39-02"

TRAIN_DATASET_NAME = BRUSH_100_100_TRAIN_M_AUGMENTED
TEST_DATASET_NAME = BRUSH_100_100_TEST_M_AUGMENTED
IMAGE_MAX_SHAPE = (112, 112)

LR = 0.001
N_EPOCHS = 5
TRAIN_SIZE = 0.8

WEIGHT_COORD = 1
WEIGHT_SKELETON = 1

def save_model_and_figures(encoder_heads, decoder_heads, encoder_layers, decoder_layers, autoregress_target_len, dropout_ratio, batch_size, patches_dim, embedding_dims, use_prediction_token, use_lstm, lstm_model_path, lr, n_epochs, dataset_name, model, return_figures):
    date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, f"{date}")
    os.makedirs(folderPath, exist_ok=True)

    filepath = os.path.join(folderPath, MODEL_FILENAME)
    print(f"Saving model to: {filepath}")
    torch.save(model, filepath)

    if return_figures is not None:
        for fig_name, figure in return_figures:
            imagepath = os.path.join(folderPath, f"{fig_name}.png")
            print(f"Saving figure image {fig_name} to {imagepath}")
            figure.savefig(imagepath)
            
            filepath = os.path.join(folderPath, f"{fig_name}.pickle")
            with open(filepath, 'wb') as f:
                pickle.dump(figure, f)
        
    filepath = os.path.join(folderPath, ID_CARD_FILE)
    id_card = IdCardCreator.create_transfo_id_card(dataset_name, lr, n_epochs, batch_size, 
                                                       encoder_layers, decoder_layers, encoder_heads, decoder_heads, 
                                                       dropout_ratio, autoregress_target_len, 
                                                       patches_dim, embedding_dims, use_prediction_token, use_lstm, lstm_model_path)
    with open(filepath, "w+") as f:
        f.write(id_card)

if __name__ == "__main__":
    #Set logging
    # for channel in LogChannels:
    #     logger.add_log_channel(channel)
    # logger.add_log_channel(LogChannels.TRAINING)
    # logger.add_log_channel(LogChannels.DEBUG)
    # logger.add_log_channel(LogChannels.INIT)
    # logger.add_log_channel(LogChannels.PARAMS)
    # logger.add_log_channel(LogChannels.DIMENSIONS)
    # logger.add_log_channel(LogChannels.PADDING)
    # logger.add_log_channel(LogChannels.MASKS)
    logger.add_log_channel(LogChannels.DATA)
    # logger.add_log_channel(LogChannels.LOSSES)
    # logger.add_log_channel(LogChannels.LOSS_DETAILED)
    logger.add_log_channel(LogChannels.DOCKER_TRACE)
    # logger.add_log_channel(LogChannels.INTERNAL_SEQUENCE_TRACE)

    print(f"Using device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''})")

    training_dataset_name = os.getenv('TRAIN_DATASET_NAME', TRAIN_DATASET_NAME)
    test_dataset_name = os.getenv('TEST_DATASET_NAME', TEST_DATASET_NAME)

    encoder_heads = int(os.getenv('ENCODER_HEADS', ENCODER_HEADS))
    decoder_heads = int(os.getenv('DECODER_HEADS', DECODER_HEADS))
    encoder_layers = int(os.getenv('ENCODER_LAYERS', ENCODER_LAYERS))
    decoder_layers = int(os.getenv('DECODER_LAYERS', DECODER_LAYERS))
    autoregress_target_len = int(os.getenv('AUTOREGRESS_TARGET_LEN', AUTOREGRESS_TARGET_LEN))
    
    image_max_shape = tuple(map(int, os.getenv('IMAGE_MAX_SHAPE', ','.join(map(str, IMAGE_MAX_SHAPE))).split(',')))

    dropout_ratio = float(os.getenv('DROPOUT_RATIO', DROPOUT_RATIO))
    batch_size = int(os.getenv('BATCH_SIZE', BATCH_SIZE))
    
    patches_dim = tuple(map(int, os.getenv('PATCHES_DIM', ','.join(map(str, PATCHES_DIM))).split(',')))
    embedding_dims = int(os.getenv('EMBEDDING_DIMS', EMBEDDING_DIMS))

    normalize_coords = bool(int(os.getenv('NORMALIZE_COORDS', int(NORMALIZE_COORDS))))

    use_prediction_token = bool(int(os.getenv('USE_PREDICTION_TOKEN', int(USE_PREDICTION_TOKEN))))
    use_lstm = bool(int(os.getenv('USE_LSTM', int(USE_LSTM))))
    lstm_model_path = os.getenv('LSTM_MODEL_PATH', LSTM_MODEL_PATH)

    train_size = float(os.getenv('TRAIN_SIZE', TRAIN_SIZE))

    lr = float(os.getenv('LR', LR))
    n_epochs = int(os.getenv('N_EPOCHS', N_EPOCHS))

    weight_coord = float(os.getenv('WEIGHT_COORD', WEIGHT_COORD))
    weight_skeleton = float(os.getenv('WEIGHT_SKELETON', WEIGHT_SKELETON))

    make_positional_encoding_learnable = bool(int(os.getenv('MAKE_POSITIONAL_ENCODING_LEARNABLE', int(MAKE_POSITIONAL_ENCODING_LEARNABLE))))

    if use_lstm:
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
    
    do_pin_memory = device != 'cpu'

    train_dataset = HandWrittingDataset(training_dataset_name, False)
    test_dataset = HandWrittingDataset(test_dataset_name, False)

    logger.log(LogChannels.INIT, f"Using n° points to predict: Train={len(train_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=do_pin_memory, collate_fn=test_dataset.get_collate_function())

    logger.log(LogChannels.INIT, f"Loading {len(train_loader)} subsequences batches as train, {len(test_loader)} subsequences batches as test")
    
    losses_weights = LossesWeights(weight_coord, weight_skeleton)
    #Init the transformer model
    model = HwTransformer(use_prediction_token=use_prediction_token, hidden_dim=embedding_dims,
                          use_lstm=use_lstm, lstm_module=lstm_model,
                          n_encoder_layers=encoder_layers, n_encoder_heads=encoder_heads, enc_dec_dropout_ratio=dropout_ratio,
                          n_decoder_layers=decoder_layers, n_decoder_heads=decoder_heads,
                          encoder_patch_dimension=patches_dim, fixed_size_image_dimension=image_max_shape,
                          autoregressive_target_seq_len=autoregress_target_len,
                          make_positional_encodings_trainable=make_positional_encoding_learnable)

    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(LogChannels.PARAMS, f"Number of model parameters: {n_model_params}")

    #Start training
    try:
        return_figures = do_training(model, train_loader, test_loader, device, n_epochs, lr, normalize_coords, image_max_shape, losses_weights)
    except Exception as e:
        print(f"Encountered exception while training model: {e}")
        raise e
    except KeyboardInterrupt:
        print(f"Training interrupted")
    finally:
        save_model_and_figures(encoder_heads, decoder_heads, encoder_layers, decoder_layers, autoregress_target_len, dropout_ratio, batch_size, patches_dim, embedding_dims, use_prediction_token, use_lstm, lstm_model_path, lr, n_epochs, (training_dataset_name, test_dataset_name), model, return_figures)