"""Contains the train and test loops for the model(s)"""
import torch
from torch import Tensor
from torch.nn import MSELoss, BCEWithLogitsLoss, BCELoss
from source.criterions.euclidian_distance import EuclideanDistanceLoss
from source.criterions.skeleton_loss import SkeletonLoss, SkeletonLossMode
from source.criterions.losses_weights import LossesWeights
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeElapsedColumn, BarColumn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels
from source.model.blocks.constants.tokens import Tokens
import matplotlib.pyplot as plt
import numpy as np

LOSS_FIGURE_NAME = "train_test_losses"
DETAILED_LOSS_FIGURE_NAME = "detailed_train_test_losses"

#Only for debug
DISPLAY_Skeleton_LOSS = False

def create_loss_figure(train_losses: list, test_losses: list, n_epochs: int):
    losses_figure = plt.figure()
    ax = losses_figure.add_subplot(111)

    ax.set_title(f'Training of a LSTM model on {n_epochs} epochs')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.plot(train_losses, 'g', label="Train loss")
    ax.plot(test_losses, 'b', label="Test loss")

    ax.legend()

    return losses_figure

def create_detailed_loss_figure(distance_losses: list, Skeleton_losses: list, n_epochs: int, n_batches: int):
    losses_figure = plt.figure()
    ax = losses_figure.add_subplot(111)

    ax.set_title(f'Detailed of the losses, transformer model, {n_epochs} epochs')
    ax.set_xlabel("Batch")
    ax.set_ylabel("Skeleton Loss")

    ax.plot(Skeleton_losses, 'r', label="Skeleton loss")
    twin_ax = ax.twinx()
    twin_ax.set_ylabel("Coordinate euclidian loss")
    twin_ax.plot(distance_losses, 'b', label="euclidian distance loss")

    total_batches_processed = len(distance_losses)
    ax.set_xlim([0, total_batches_processed])

    #Plot every epoch up untill last registered loss (If training is stopped before end, there will be less epochs than expected)
    for i in range(1, n_epochs+1):
        if (i * n_batches) > total_batches_processed:
            break
        ax.axvline(x = i * n_batches, color = 'm', linestyle='--', label = 'Epochs' if i == 1 else None)

    ax.legend()
    twin_ax.legend()

    return losses_figure

def create_figures(train_losses: list, test_losses: list, epochs: int, batch_per_epoch: int,
                   distance_losses: list, Skeleton_losses: list) -> list[tuple]:
    figures: list[tuple[str, any]] = []

    figures.append((LOSS_FIGURE_NAME, create_loss_figure(train_losses, test_losses, epochs)))
    figures.append((DETAILED_LOSS_FIGURE_NAME, create_detailed_loss_figure(distance_losses, Skeleton_losses, epochs, batch_per_epoch)))

    return figures

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

def do_training(model: HwTransformer, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, 
                n_epochs: int, lr: float, normalized_sequences: bool, dataset_image_shape: tuple, losses_weights: LossesWeights):
    """Do the training and return the given model"""
    model = model.to(device)

    train_losses = []
    test_losses = []
    distances_losses, Skeleton_losses = [], []

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn()
            ) as progress:

        # Training loop
        optimizer = Adam(model.parameters(), lr=lr)
        coord_criterion = EuclideanDistanceLoss()
        Skeleton_criterion = SkeletonLoss(normalized_sequences=normalized_sequences, dataset_image_shape=dataset_image_shape, 
                                          display=DISPLAY_Skeleton_LOSS, mode=SkeletonLossMode.DIST_PIX)

        epoch_progress_bar = progress.add_task("[blue]Epoch...", total=n_epochs)
        batch_progress_bar = progress.add_task("[red]Batch...", total=len(train_loader))
        sep = '\n\t'

        try:
            for epoch in range(n_epochs):

                train_loss = 0.0

                for batch in train_loader:
                    original_images, images_patches, masks, sequences, labels = data_from_batch(batch, device)

                    # Iterate over the sequences untill all are over. 
                    y_pred = model.forward(images_patches, masks, sequences)
                    if not normalized_sequences:
                        y_pred = torch.round(y_pred)

                    ### COORD loss ###
                    if LogChannels.LOSS_DETAILED in logger.channels:
                        coordLossesAsStr = [f"{labels[i].cpu().detach().numpy()}-{y_pred[i].cpu().detach().numpy()}" for i in range(len(labels))]
                        logger.log(LogChannels.LOSS_DETAILED, f"Coordinates labels-pred:{sep}{sep.join(coordLossesAsStr)}")

                    coord_loss = coord_criterion(y_pred, labels) * losses_weights.coord_weight

                    ### Skeleton loss ###
                    last_coordinates = retrieve_last_values(sequences)
                    Skeleton_loss = Skeleton_criterion(last_coordinates, y_pred.detach(), original_images) * losses_weights.skeleton_weight

                    coord_loss_as_nbr, Skeleton_loss_as_nbr = coord_loss.detach().cpu().item(), Skeleton_loss.detach().cpu().item()

                    distances_losses.append(coord_loss_as_nbr)
                    Skeleton_losses.append(Skeleton_loss_as_nbr)

                    loss = (coord_loss + Skeleton_loss) / losses_weights.total_weights

                    logger.log(LogChannels.LOSSES, f"TRAIN LOOP")
                    logger.log(LogChannels.LOSSES, f"coord_loss = {coord_loss_as_nbr}")
                    logger.log(LogChannels.LOSSES, f"Skeleton_loss = {Skeleton_loss_as_nbr}")
                    logger.log(LogChannels.LOSSES, f"total_loss = {loss.detach().cpu().item()}\n")

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                    optimizer.step()

                    # Check for vanishing or exploding gradients. This helps detecting two well-known issues with gradiants.
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm() if param.grad is not None else None
                            if grad_norm is None:
                                continue
                            if grad_norm < 1e-6:
                                print(f"Warning: Vanishing gradients in layer {name}")
                            if grad_norm > 1e2:
                                print(f"Warning: Exploding gradients in layer {name}")
                    
                    train_loss += loss.detach().cpu().item()
                    progress.advance(batch_progress_bar)

                    del images_patches, masks, sequences, labels, y_pred, loss

                train_loss /= len(train_loader)
                print(f"Epoch {epoch + 1}/{n_epochs} Train loss: {train_loss:.2f}")
                train_losses.append(train_loss)

                # Test loop
                logger.log(LogChannels.LOSSES, f"TEST LOOP...")
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in test_loader:
                        original_images, images_patches, masks, sequences, labels = data_from_batch(batch, device)

                        y_pred = model.forward(images_patches, masks, sequences)                    
                        if not normalized_sequences:
                            y_pred = torch.round(y_pred)

                        ### COORD loss ###
                        coord_loss = coord_criterion(y_pred, labels)

                        ### Skeleton loss ###
                        last_coordinates = retrieve_last_values(sequences)
                        Skeleton_loss = Skeleton_criterion(last_coordinates, y_pred.detach(), original_images)

                        loss = (coord_loss + Skeleton_loss) / losses_weights.total_weights
                        test_loss += loss.detach().cpu().item()

                        logger.log(LogChannels.LOSSES, f"coord_loss = {coord_loss.detach().cpu().item()}")
                        logger.log(LogChannels.LOSSES, f"Skeleton_loss = {Skeleton_loss.detach().cpu().item()}")
                        logger.log(LogChannels.LOSSES, f"total_loss = {loss.detach().cpu().item()}")

                        del images_patches, masks, sequences, labels, y_pred, loss
                    
                    test_loss /= len(test_loader)
                    print(f"Epoch {epoch + 1}/{n_epochs} Test loss: {test_loss:.2f}")
                    test_losses.append(test_loss)
                
                progress.advance(epoch_progress_bar)
                progress.reset(batch_progress_bar)

        except Exception as e:
            print(f"Stopping due to {e}")
        
        finally:
            return create_figures(train_losses, test_losses, n_epochs, len(train_loader), distances_losses, Skeleton_losses)