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

import numpy as np
from enum import Enum
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import MedianStoppingRule, ASHAScheduler

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
BATCH_SIZE = 64
LR = 1e-3

MAX_ITER = 6

#Keys used in configuration dict
class StructuralParameters(Enum):
    DATASETS = "datasets"
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

    train_dataset_name, test_dataset_name = config[StructuralParameters.DATASETS.value]

    train_dataset = HandWrittingDataset(train_dataset_name, lstm_mode=False)
    test_dataset = HandWrittingDataset(test_dataset_name, lstm_mode=False)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=train_dataset.get_collate_function())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=do_pin_memory, collate_fn=test_dataset.get_collate_function())
    
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

            #TODO DEL test
            break
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

                #TODO DEL test
                break
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

def trial_name_creator(trial):
    config: dict = trial.config
    augment_mode = config[StructuralParameters.DATASETS.value][0].split('_')[-1]

    return f"augment_mode={augment_mode}_lstm={config[StructuralParameters.USE_LSTM.value]}_pred_tok={config[StructuralParameters.USE_PRED_TOKEN.value]}_learn-pos={config[StructuralParameters.IS_POSITION_LEARNABLE.value]}"

if __name__ == "__main__":
    # Search space for hyperparameters
    augmented_datasets = (BRUSH_96_96_TRAIN_S_AUGMENTED, BRUSH_96_96_TEST_S_AUGMENTED)
    unaugmented_datasets = (BRUSH_96_96_TRAIN_S_UNAUGMENTED, BRUSH_96_96_TEST_S_UNAUGMENTED)
    mixed_datasets = (BRUSH_96_96_TRAIN_S_MIXED, BRUSH_96_96_TEST_S_MIXED)

    search_space = {
        StructuralParameters.DATASETS.value: tune.choice([augmented_datasets, unaugmented_datasets, mixed_datasets]),
        StructuralParameters.USE_PRED_TOKEN.value: tune.choice([True, False]),
        StructuralParameters.USE_LSTM.value: tune.choice([True, False]),
        StructuralParameters.IS_POSITION_LEARNABLE.value: tune.choice([True, False])
    }

    # In case of median stopping, but ashas can be favored (less overall runs)
    # scheduler = MedianStoppingRule(
    #     time_attr="training_iteration",
    #     metric="test_loss", 
    #     mode="min", 
    #     grace_period=2
    # )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="test_loss", 
        mode="min", 
        max_t=MAX_ITER,
        grace_period=2,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_loop,
        trial_name_creator=trial_name_creator,
        config=search_space,
        scheduler=scheduler,
        num_samples=24,
        resources_per_trial = { "gpu": 1.0 },
        max_concurrent_trials = 1
    )

    # Print the best hyperparameters
    print("Best config: ", analysis.get_best_config(metric="test_loss", mode="min"))

    # Save results
    df = analysis.results_df
    df.to_csv("~/ray_results/results_df.csv")    

    #TODO del
    os.makedirs("/home/ubuntu/ray_results", exist_ok=True)
    df.to_csv("/home/ubuntu/ray_results/results_df.csv")