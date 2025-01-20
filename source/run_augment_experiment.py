
"""
This is a slight modification of the very generalist 'run_experiment'.
It start an experiment on 6 epochs with the same model using the augmented, unaugmented, and mixed datasets in order to compare their results
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.loops import do_training
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from torch.utils.data import DataLoader

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

ENCODER_LAYERS = 6
DECODER_LAYERS = 6

AUTOREGRESS_TARGET_LEN = 100

MAKE_POSITIONAL_ENCODING_LEARNABLE = False

DROPOUT_RATIO = 0.1
BATCH_SIZE = 512

PATCHES_DIM = (16, 16)
EMBEDDING_DIMS = 256

NORMALIZE_COORDS = False

USE_PREDICTION_TOKEN = True
USE_LSTM = False

IMAGE_MAX_SHAPE = (96, 96)

LR = 0.001
N_EPOCHS = 6

WEIGHT_COORD = 1
WEIGHT_SKELETON = 1

def save_model_and_figures(model_name, encoder_heads, decoder_heads, encoder_layers, decoder_layers, autoregress_target_len, dropout_ratio, batch_size, patches_dim, embedding_dims, use_prediction_token, use_lstm, lstm_model_path, lr, n_epochs, dataset_name, model, return_figures):
    folder_name = model_name if model_name else datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, folder_name)
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

def experiment_with_dataset(model_name : str, training_dataset_name : str, test_dataset_name : str) -> None:
    """Start the experiment with the given train dataset and test dataset names. Expect dataset to have been created beforehand"""
    lstm_model = None
    
    do_pin_memory = device != 'cpu'

    train_dataset = HandWrittingDataset(training_dataset_name, False)
    test_dataset = HandWrittingDataset(test_dataset_name, False)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=test_dataset.get_collate_function())
    
    losses_weights = LossesWeights(WEIGHT_COORD, WEIGHT_SKELETON)
    #Init the transformer model
    model = HwTransformer(use_prediction_token=USE_PREDICTION_TOKEN, hidden_dim=EMBEDDING_DIMS,
                          use_lstm=USE_LSTM, lstm_module=lstm_model,
                          n_encoder_layers=ENCODER_LAYERS, n_encoder_heads=ENCODER_HEADS, enc_dec_dropout_ratio=DROPOUT_RATIO,
                          n_decoder_layers=DECODER_LAYERS, n_decoder_heads=DECODER_HEADS,
                          encoder_patch_dimension=PATCHES_DIM, fixed_size_image_dimension=IMAGE_MAX_SHAPE,
                          autoregressive_target_seq_len=AUTOREGRESS_TARGET_LEN,
                          make_positional_encodings_trainable=MAKE_POSITIONAL_ENCODING_LEARNABLE)
    #Start training
    try:
        return_figures = do_training(model, train_loader, test_loader, device, N_EPOCHS, LR, NORMALIZE_COORDS, IMAGE_MAX_SHAPE, losses_weights)
    except Exception as e:
        print(f"Encountered exception while training model: {e}")
        raise e
    except KeyboardInterrupt:
        print(f"Training interrupted")
    finally:
        save_model_and_figures(model_name, ENCODER_HEADS, DECODER_HEADS, ENCODER_LAYERS, DECODER_LAYERS, AUTOREGRESS_TARGET_LEN, DROPOUT_RATIO, BATCH_SIZE, PATCHES_DIM, EMBEDDING_DIMS, USE_PREDICTION_TOKEN, USE_LSTM, "", LR, N_EPOCHS, (training_dataset_name, test_dataset_name), model, return_figures)

if __name__ == "__main__":
    logger.add_log_channel(LogChannels.DATA)
    logger.add_log_channel(LogChannels.DOCKER_TRACE)

    #Get the model names and dataset's names to test.
    tuples_to_test = [
        ("unaugmented", BRUSH_96_96_TRAIN_S_UNAUGMENTED, BRUSH_96_96_TEST_S_UNAUGMENTED),
        ("augmented", BRUSH_96_96_TRAIN_S_AUGMENTED, BRUSH_96_96_TEST_S_AUGMENTED),
        ("mixed", BRUSH_96_96_TRAIN_S_MIXED, BRUSH_96_96_TEST_S_MIXED)
    ]

    for tuple_to_test in tuples_to_test:
        experiment_with_dataset(*tuple_to_test)
