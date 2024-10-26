"""Contains the train and test loops for the model(s)"""
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeElapsedColumn, BarColumn

from source.model.blocks.hw_lstm import HwLstm
from source.logging.log import logger, LogChannels

from source.criterions.euclidian_distance import EuclideanDistanceLoss
import matplotlib.pyplot as plt

LOSS_FIGURE_NAME = "train_test_losses"

def create_loss_figure(train_losses: list, test_losses: list, n_epochs: int):
    losses_figure = plt.figure()
    ax = losses_figure.add_subplot(111)

    ax.set_title(f'Training of a LSTM model on {n_epochs} epochs')
    ax.plot(train_losses, 'g', label="Train loss")
    ax.plot(test_losses, 'b', label="Test loss")
    ax.legend()

    return losses_figure

def create_figures(train_losses: list, test_losses: list, epochs: int) -> list[tuple]:
    figures: list[tuple[str, any]] = []

    figures.append((LOSS_FIGURE_NAME, create_loss_figure(train_losses, test_losses, epochs)))

    return figures

def do_training(model: HwLstm, train_loader: DataLoader, test_loader: DataLoader, device: torch.device,
                n_epochs: int, lr: float):
    """Do the training and return the given model"""
    model = model.to(device)

    train_losses = []
    test_losses = []

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn()
            ) as progress:

        # Training loop
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = EuclideanDistanceLoss()

        epoch_progress_bar = progress.add_task("[blue]Epoch...", total=n_epochs)
        batch_progress_bar = progress.add_task("[green]Batch...", total=len(train_loader))

        try:
            for epoch in range(n_epochs):
                train_loss = 0.0

                progress.reset(batch_progress_bar)

                for batch in train_loader:
                    #Extract labels from data
                    packed_sequences, labels = batch
                    packed_sequences, labels = packed_sequences.to(device), labels.to(device)

                    # Iterate over the sequences untill all are over.
                    y_pred = model.forward(packed_sequences, last_layer_mlp=True)
                    asStr = [f"|{labels[i].cpu().detach().numpy()}-{y_pred[i].cpu().detach().numpy()}|" for i in range(len(y_pred))]
                    logger.log(LogChannels.LOSSES, f"labels-pred ={' | '.join(asStr)}")

                    loss = criterion(y_pred, labels)
                    logger.log(LogChannels.LOSSES, f"loss = {loss.detach().cpu().item()}")

                    train_loss += loss.detach().cpu().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    progress.advance(batch_progress_bar)

                    del labels, y_pred, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                progress.advance(epoch_progress_bar)

                train_loss /= len(train_loader)
                print(f"Epoch {epoch + 1}/{n_epochs} Train loss: {train_loss:.2f}")
                train_losses.append(train_loss)

                # Test loop
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in test_loader:
                        packed_sequences, labels = batch, device
                        packed_sequences, labels = packed_sequences.to(device), labels.to(device)

                        # Iterate over the sequences untill all are over. 
                        y_pred = model.forward(packed_sequences, last_layer_mlp=True)

                        loss = criterion(y_pred, labels)
                        test_loss += loss.detach().cpu().item()

                        del labels, y_pred, loss
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    test_loss /= len(test_loader)
                    print(f"Epoch {epoch + 1}/{n_epochs} Test loss: {test_loss:.2f}")
                test_losses.append(test_loss)
            
        except Exception as e:
            print(f"Encountered exception {e}")
        finally:
            return create_figures(train_losses, test_losses, n_epochs)