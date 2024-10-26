"""Contains the train and test loops for the model(s)"""
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss, BCELoss
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeElapsedColumn, BarColumn

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels
from source.model.blocks.constants.tokens import Tokens
from source.criterions.euclidian_distance import EuclideanDistanceLoss
import matplotlib.pyplot as plt

LOSS_FIGURE_NAME = "train_test_losses"
DETAILED_LOSS_FIGURE_NAME = "detailed_train_test_losses"

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

def create_detailed_loss_figure(eos_losses: list, distance_losses: list, skeletton_losses: list, n_epochs: int):
    losses_figure = plt.figure()
    ax = losses_figure.add_subplot(111)

    ax.set_title(f'Detailed of the losses, transformer model, {n_epochs} epochs')
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")

    ax.plot(eos_losses, 'g', label="EOS loss")
    ax.plot(distance_losses, 'b', label="euclidian distance loss")
    ax.plot(skeletton_losses, 'r', label="skeletton loss")
    ax.legend()

    return losses_figure

def create_figures(train_losses: list, test_losses: list, epochs: int,
                   eos_losses: list, distance_losses: list, skeletton_losses: list) -> list[tuple]:
    figures: list[tuple[str, any]] = []

    figures.append((LOSS_FIGURE_NAME, create_loss_figure(train_losses, test_losses, epochs)))
    figures.append((DETAILED_LOSS_FIGURE_NAME, create_detailed_loss_figure(eos_losses, distance_losses, skeletton_losses, epochs)))

    return figures

def data_from_batch(batch, device):
    #Get images, coord sequences as batch
    images, masks, sequences, labels = batch
    images, masks, sequences, labels = images.to(device), masks.to(device), sequences.to(device), labels.to(device)
    #Infer the stop labels - if the token to predict is EOS, then the output should be true.
    stop_labels = torch.all(labels == Tokens.EOS_TENSOR.value, dim=1).float().unsqueeze(-1)

    return images, masks, sequences, labels, stop_labels

def do_training(model: HwTransformer, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, 
                n_epochs: int, lr: float):
    """Do the training and return the given model"""
    model = model.to(device)

    train_losses = []
    test_losses = []
    eos_losses, distances_losses, skeletton_losses = [], [], []

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn()
            ) as progress:

        # Training loop
        optimizer = Adam(model.parameters(), lr=lr)
        coord_criterion = EuclideanDistanceLoss()
        #SUM because MEAN will mean EOS is ignored, as there are almost no stop point (1/signal) compared to regular coords
        eos_criterion = BCEWithLogitsLoss()

        epoch_progress_bar = progress.add_task("[blue]Epoch...", total=n_epochs)
        batch_progress_bar = progress.add_task("[green]Batch...", total=len(train_loader))

        try:
            for epoch in range(n_epochs):
                train_loss = 0.0

                progress.reset(batch_progress_bar)

                for batch in train_loader:
                    images, masks, sequences, labels, stop_labels = data_from_batch(batch, device)

                    # Iterate over the sequences untill all are over. 
                    y_pred, eos_output = model.forward(images, masks, sequences)

                    #Mask coordinates when EOS is reached to avoid double-guessing EOS signal + EOS coordinates of -1,-1
                    mask = (stop_labels == 0).squeeze(-1)
                    y_pred = y_pred[mask]
                    labels = labels[mask]

                    coordLossesAsStr = [f"{labels[i].cpu().detach().numpy()}-{y_pred[i].cpu().detach().numpy()}" for i in range(len(labels))]
                    sep = '\n\t'
                    logger.log(LogChannels.LOSS_DETAILED, f"Coordinates labels-pred:{sep}{sep.join(coordLossesAsStr)}")

                    eosLossesAsStr = [f"{stop_labels[i].cpu().detach().numpy()}-{eos_output[i].cpu().detach().numpy()}" for i in range(len(stop_labels))]
                    logger.log(LogChannels.LOSS_DETAILED, f"EOS labels-pred:{sep}{sep.join(eosLossesAsStr)}")

                    eos_loss = eos_criterion(eos_output, stop_labels)
                    coord_loss = coord_criterion(y_pred, labels)
                    skeletton_loss = torch.tensor([0.0], device='cuda')

                    eos_loss_as_nbr, coord_loss_as_nbr, skeletton_loss_as_nbr = eos_loss.detach().cpu().item(), coord_loss.detach().cpu().item(), skeletton_loss.detach().cpu().item()

                    eos_losses.append(eos_loss_as_nbr)
                    distances_losses.append(coord_loss_as_nbr)
                    skeletton_losses.append(skeletton_loss_as_nbr)

                    loss = coord_loss #+ eos_loss + skeletton_loss

                    logger.log(LogChannels.LOSSES, f"stop_loss = {eos_loss_as_nbr}")
                    logger.log(LogChannels.LOSSES, f"coord_loss = {coord_loss_as_nbr}")
                    logger.log(LogChannels.LOSSES, f"skeletton_loss = {skeletton_loss_as_nbr}")
                    logger.log(LogChannels.LOSSES, f"total_loss = {loss.detach().cpu().item()}\n")

                    train_loss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                    optimizer.step()

                    # Check for vanishing or exploding gradients. This helps detecting two well-known issues with gradiants.
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            grad_norm = param.grad.norm() if param.grad is not None else None  # Check for None
                            if grad_norm is None:
                                continue
                            if grad_norm < 1e-6:
                                print(f"Warning: Vanishing gradients in layer {name}")
                            if grad_norm > 1e2:
                                print(f"Warning: Exploding gradients in layer {name}")

                    progress.advance(batch_progress_bar)

                    del images, masks, sequences, labels, y_pred, loss
                    #torch.cuda.empty_cache() if torch.cuda.is_available() else None

                progress.advance(epoch_progress_bar)

                train_loss /= len(train_loader)
                print(f"Epoch {epoch + 1}/{n_epochs} Train loss: {train_loss:.2f}")
                train_losses.append(train_loss)

                # Test loop
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in test_loader:
                        images, masks, sequences, labels, stop_labels = data_from_batch(batch, device)

                        y_pred, eos_output = model.forward(images, masks, sequences)

                        coord_loss = coord_criterion(y_pred, labels)
                        eos_loss = eos_criterion(eos_output, stop_labels)

                        loss = coord_loss + eos_loss                    
                        test_loss += loss.detach().cpu().item()

                        logger.log(LogChannels.LOSSES, f"coord_loss = {coord_loss.detach().cpu().item()}")
                        logger.log(LogChannels.LOSSES, f"stop_loss = {eos_loss.detach().cpu().item()}")
                        logger.log(LogChannels.LOSSES, f"total_loss = {loss.detach().cpu().item()}")

                        del images, masks, sequences, labels, y_pred, loss
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    test_loss /= len(test_loader)
                    print(f"Epoch {epoch + 1}/{n_epochs} Test loss: {test_loss:.2f}")
                test_losses.append(test_loss)
        except Exception as e:
            print(f"Stopping due to {e}")
            raise e
        finally:
            return create_figures(train_losses, test_losses, n_epochs,
                                  eos_losses, distances_losses, skeletton_losses)