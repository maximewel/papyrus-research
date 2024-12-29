import os
import sys
import tempfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.blocks.hw_lstm import HwLstm
from source.model.hw_model import HwTransformer
from source.model.blocks.constants.files import *
from source.model.blocks.constants.device_helper import device
from source.criterions.losses_weights import LossesWeights
from source.criterions.euclidian_distance import EuclideanDistanceLoss
from source.criterions.skeleton_loss import SkeletonLoss, SkeletonLossMode
from source.criterions.losses_weights import LossesWeights
from source.model.blocks.constants.tokens import Tokens
from source.model.blocks.constants.datasets_library import *
import json

import numpy as np
from enum import Enum
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import MedianStoppingRule

#Fixed constants for the structural hyper-parameter search
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LSTM_MODEL = "lstm_96.96_unaugmented"
EMBEDDING_DIMS = 256
AUTOREGRESS_TARGET_LENGTH= 100
N_LAYERS = 6
N_HEADS = 8
DROPOUT_RATIO= 0.1
PATCH_DIM = (16, 16)
IMAGE_SHAPE = (96, 96)
W_LOSS = 1
BATCH_SIZE = 512
LR = 1e-3

TRAIN_DATASET = HandWrittingDataset(BRUSH_96_96_TRAIN_S_AUGMENTED)
TEST_DATASET = HandWrittingDataset(BRUSH_96_96_TEST_S_AUGMENTED)

MAX_ITER = 4

#Keys used in configuration dict
class StructuralParameters(Enum):
    USE_PRED_TOKEN = "pred_token"
    USE_LSTM = "use_lstm"
    IS_POSITION_LEARNABLE = "positional_learnable"

def data_from_batch(batch, device) -> tuple[list[np.ndarray], Tensor, Tensor, PackedSequence, Tensor, Tensor]:
    """
    Get all data from batch. Specifically.
    """
    #Get images, coord sequences as batch
    images, patched_images, masks, sequences, labels = batch
    patched_images, masks, sequences, labels = patched_images.to(device), masks.to(device), sequences.to(device), labels.to(device)

    return images, patched_images, masks, sequences, labels

def retrieve_last_values(sequences: PackedSequence) -> Tensor:
    """
        Return the last values from the packed sequences
    """
    padded_data, unpacked_lengths = pad_packed_sequence(sequences, batch_first=True)
    last_values = [padded_data[i, seq_len-1] for i, seq_len in enumerate(unpacked_lengths)]

    return torch.stack(last_values)

def train_loop(config: dict):
    """Loop used to train a model under the given parameters and report results to raytune's engine"""
    losses_weights = LossesWeights(W_LOSS, W_LOSS)

    use_lstm = config[StructuralParameters.USE_LSTM.value]
    if use_lstm:
        #Load pre-trained LSTM model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_dir = os.path.join(MODEL_FOLDER, LSTM_FOLDER)
        filepath = os.path.join(current_dir, relative_dir, LSTM_MODEL, MODEL_FILENAME)
        lstm_model: HwLstm = torch.load(filepath)
        #Freeze model as we have a pre-trained LSTM model that doesnt need to learn in this step
        for param in lstm_model.parameters():
            param.requires_grad = False
    else:
        lstm_model = None

    do_pin_memory = True

    train_loader = DataLoader(TRAIN_DATASET, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=TRAIN_DATASET.get_collate_function())
    test_loader = DataLoader(TEST_DATASET, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=TEST_DATASET.get_collate_function())
    
    #Init the transformer model
    model = HwTransformer(use_lstm=use_lstm, lstm_module=lstm_model,
                            use_prediction_token=config[StructuralParameters.USE_PRED_TOKEN.value], make_positional_encodings_trainable=config[StructuralParameters.IS_POSITION_LEARNABLE.value],
                            hidden_dim=EMBEDDING_DIMS, autoregressive_target_seq_len=AUTOREGRESS_TARGET_LENGTH,
                            n_encoder_layers=N_LAYERS, n_encoder_heads=N_HEADS, enc_dec_dropout_ratio=DROPOUT_RATIO,
                            n_decoder_layers=N_LAYERS, n_decoder_heads=N_HEADS,
                            encoder_patch_dimension=PATCH_DIM, fixed_size_image_dimension=IMAGE_SHAPE,
                            )

    model = model.to(device)

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    coord_criterion = EuclideanDistanceLoss()
    Skeleton_criterion = SkeletonLoss(normalized_sequences=False, dataset_image_shape=IMAGE_SHAPE, mode=SkeletonLossMode.DIST_LAST_PIX)
    
    #If checkpointed, retrieve last values
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            start_epoch = checkpoint["epoch"] + 1
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    train_losses = []
    test_losses = []
    for epoch in range(start_epoch, MAX_ITER):
        train_loss = 0.0
        for batch in train_loader:
            original_images, images_patches, masks, sequences, labels = data_from_batch(batch, device)

            # Iterate over the sequences untill all are over. 
            y_pred = model.forward(images_patches, masks, sequences)

            ### COORD loss ###
            coord_loss = coord_criterion(y_pred, labels) * losses_weights.coord_weight

            ### Skeleton loss ###
            #Apply skeletton loss only on generated tensors that are not EOS
            label_eos_mask = ~(labels == Tokens.EOS_TENSOR.value).all(dim=1)
            last_coordinates = retrieve_last_values(sequences)
            Skeleton_loss = Skeleton_criterion(last_coordinates[label_eos_mask], y_pred.detach()[label_eos_mask], original_images) * losses_weights.skeleton_weight

            loss = (coord_loss + Skeleton_loss) / losses_weights.total_weights

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            
            train_loss += loss.detach().cpu().item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Test loop
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_loader:
                original_images, images_patches, masks, sequences, labels = data_from_batch(batch, device)

                y_pred = model.forward(images_patches, masks, sequences)

                ### COORD loss ###
                coord_loss = coord_criterion(y_pred, labels) * losses_weights.coord_weight

                ### Skeleton loss ###
                #Apply skeletton loss only on generated tensors that are not EOS
                label_eos_mask = ~(labels == Tokens.EOS_TENSOR.value).all(dim=1)
                last_coordinates = retrieve_last_values(sequences)
                Skeleton_loss = Skeleton_criterion(last_coordinates[label_eos_mask], y_pred.detach()[label_eos_mask], original_images) * losses_weights.skeleton_weight

                loss = (coord_loss + Skeleton_loss) / losses_weights.total_weights
                test_loss += loss.detach().cpu().item()

            test_loss /= len(test_loader)
            test_losses.append(test_loss)

            with tempfile.TemporaryDirectory() as save_checkpoint_dir:
                checkpoint_path = os.path.join(save_checkpoint_dir, "checkpoint.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)

                train.report(
                    {
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "training_iteration": epoch + 1,
                        #Usefull for end reports
                        "train_losses": train_losses,
                        "test_losses": test_losses,
                    },
                    checkpoint=Checkpoint.from_directory(save_checkpoint_dir)
                )

if __name__ == "__main__":
    # Search space for hyperparameters
    search_space = {
        StructuralParameters.USE_PRED_TOKEN.value: tune.grid_search([True, False]),
        StructuralParameters.USE_LSTM.value: tune.grid_search([True, False]),
        StructuralParameters.IS_POSITION_LEARNABLE.value: tune.grid_search([True, False])
    }

    scheduler = MedianStoppingRule(
        time_attr="training_iteration",
        metric="test_loss", 
        mode="min",
        grace_period=2,
        min_samples_required=3
    )

    analysis = tune.run(
        train_loop,
        config=search_space,
        scheduler=scheduler,
        num_samples=1,
        resources_per_trial = { "gpu": 1.0 },
        max_concurrent_trials = 1
    )

    # Print the best hyperparameters
    print("Best config: ", analysis.get_best_config(metric="test_loss", mode="min"))

    # Save results
    df = analysis.results_df
    os.makedirs("/home/ubuntu/ray_results", exist_ok=True)
    df.to_csv("/home/ubuntu/ray_results/results_df.csv")
    with open('/home/ubuntu/ray_results/best_results.txt', 'wt+') as f:
        f.write(json.dumps(analysis.get_best_config(metric="test_loss", mode="min"), indent=4))