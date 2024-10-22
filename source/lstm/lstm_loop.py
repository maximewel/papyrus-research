"""Contains the train and test loops for the model(s)"""
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeElapsedColumn, BarColumn

from source.model.blocks.hw_lstm import HwLstm
from source.logging.log import logger, LogChannels

from source.model.blocks.constants.tokens import Tokens

def do_training(model: HwLstm, train_loader: DataLoader, test_loader: DataLoader, device: torch.device):
    """Do the training and return the given model"""
    N_EPOCHS = 50
    LR = 0.001

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
        optimizer = Adam(model.parameters(), lr=LR)
        criterion = L1Loss(reduction='sum')

        epoch_progress_bar = progress.add_task("[blue]Epoch...", total=N_EPOCHS)
        batch_progress_bar = progress.add_task("[green]Batch...", total=len(train_loader))

        for epoch in range(N_EPOCHS):
            train_loss = 0.0

            progress.reset(batch_progress_bar)

            for batch in train_loader:
                #Get images, coord sequences as batch
                (_, _, sequences), labels = batch
                sequences, labels = sequences.to(device), labels.to(device)

                # Iterate over the sequences untill all are over.
                y_pred_finished = model.forward(sequences)
                asStr = [f"|{labels[i].cpu().detach().numpy()}-{y_pred_finished[i].cpu().detach().numpy()}|" for i in range(len(y_pred_finished))]
                logger.log(LogChannels.LOSSES, f"labels-pred ={' | '.join(asStr)}")

                loss = criterion(y_pred_finished, labels)
                logger.log(LogChannels.LOSSES, f"loss = {loss.detach().cpu().item()}")

                train_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.advance(batch_progress_bar)

                del sequences, labels, y_pred_finished, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            progress.advance(epoch_progress_bar)

            train_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{N_EPOCHS} Train loss: {train_loss:.2f}")
            train_losses.append(train_loss)

            # Test loop
            with torch.no_grad():
                test_loss = 0.0
                for batch in test_loader:
                    (_, _, sequences), labels = batch
                    sequences, labels = sequences.to(device), labels.to(device)

                    # Iterate over the sequences untill all are over. 
                    y_pred = model.forward(sequences)

                    loss = criterion(y_pred, labels)
                    test_loss += loss.detach().cpu().item()

                    del sequences, labels, y_pred, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                test_loss /= len(test_loader)
                print(f"Epoch {epoch + 1}/{N_EPOCHS} Test loss: {test_loss:.2f}")
            test_losses.append(test_loss)

        import matplotlib.pyplot as plt
        plt.title(f'Training of a LSTM model on {N_EPOCHS} epochs')
        plt.plot(train_losses, 'g', label="Train loss")
        plt.plot(test_losses, 'b', label="Test loss")
        plt.legend()
        plt.show()