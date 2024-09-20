"""Contains the train and test loops for the model(s)"""
import torch
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeElapsedColumn, BarColumn

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels
from source.model.blocks.constants.tokens import Tokens

def data_from_batch(batch, device):
    #Get images, coord sequences as batch
    (images, masks, sequences), labels = batch
    images, masks, sequences, labels = images.to(device), masks.to(device), sequences.to(device), labels.to(device)
    #Infer the stop labels - if the token to predict is EOS, then the output should be true.
    stop_labels = torch.all(labels == Tokens.EOS_TENSOR.value, dim=1).float().unsqueeze(-1)

    return images, masks, sequences, labels, stop_labels

def do_training(model: HwTransformer, train_loader: DataLoader, test_loader: DataLoader, device: torch.device,
                n_epochs: int = 10, lr: float = 0.005):
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
        coord_criterion = L1Loss(reduction='sum')
        eos_criterion = BCEWithLogitsLoss()

        epoch_progress_bar = progress.add_task("[blue]Epoch...", total=n_epochs)
        batch_progress_bar = progress.add_task("[green]Batch...", total=len(train_loader))

        for epoch in range(n_epochs):
            train_loss = 0.0

            progress.reset(batch_progress_bar)

            for batch in train_loader:
                images, masks, sequences, labels, stop_labels = data_from_batch(batch, device)

                # Iterate over the sequences untill all are over. 
                y_pred_finished, eos_output = model.forward(images, masks, sequences)
                asStr = [f"|{labels[i].cpu().detach().numpy()}-{y_pred_finished[i].cpu().detach().numpy()}|" for i in range(len(y_pred_finished))]
                logger.log(LogChannels.LOSSES, f"labels-pred ={' | '.join(asStr)}")

                coord_loss = coord_criterion(y_pred_finished, labels)
                eos_loss = eos_criterion(eos_output, stop_labels)

                loss = coord_loss + eos_loss

                logger.log(LogChannels.LOSSES, f"coord_loss = {coord_loss.detach().cpu().item()}")
                logger.log(LogChannels.LOSSES, f"stop_loss = {eos_loss.detach().cpu().item()}")
                logger.log(LogChannels.LOSSES, f"total_loss = {loss.detach().cpu().item()}")

                train_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.advance(batch_progress_bar)

                del images, masks, sequences, labels, y_pred_finished, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            progress.advance(epoch_progress_bar)

            train_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{n_epochs} Train loss: {train_loss:.2f}")
            train_losses.append(train_loss)

            # Test loop
            with torch.no_grad():
                test_loss = 0.0
                for batch in test_loader:
                    images, masks, sequences, labels, stop_labels = data_from_batch(batch, device)

                    y_pred_finished, eos_output = model.forward(images, masks, sequences)

                    coord_loss = coord_criterion(y_pred_finished, labels)
                    eos_loss = eos_criterion(eos_output, stop_labels)

                    loss = coord_loss + eos_loss                    
                    test_loss += loss.detach().cpu().item()

                    logger.log(LogChannels.LOSSES, f"coord_loss = {coord_loss.detach().cpu().item()}")
                    logger.log(LogChannels.LOSSES, f"stop_loss = {eos_loss.detach().cpu().item()}")
                    logger.log(LogChannels.LOSSES, f"total_loss = {loss.detach().cpu().item()}")

                    del images, masks, sequences, labels, y_pred_finished, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                test_loss /= len(test_loader)
                print(f"Epoch {epoch + 1}/{n_epochs} Test loss: {test_loss:.2f}")
            test_losses.append(test_loss)

        import matplotlib.pyplot as plt
        plt.plot(train_losses, 'g', label="Train loss")
        plt.plot(test_losses, 'b', label="Test loss")
        plt.legend()
        plt.show()
        
# logger.log(LogChannels.TRAINING, f"Shapes of y{sequences.shape}:, pred: {y_pred_finished.shape}")
# seq = sequences[0]
# cleaned_seq = seq[~(seq == Tokens.EOS_TENSOR.value).all(dim=-1)]
# seq_pred = y_pred_finished[0]
# cleaned_seq_pred = seq_pred[~(seq_pred == Tokens.EOS_TENSOR.value).all(dim=-1)]
# logger.log(LogChannels.TRAINING, f"First: {cleaned_seq.shape}, {cleaned_seq}")
# logger.log(LogChannels.TRAINING, f"First: {cleaned_seq_pred.shape}, {cleaned_seq_pred}")

# lengths = []
# pred_lengths = []
# for i in range(sequences.shape[0]):
#     seq = sequences[i]
#     cleaned_seq = seq[~(seq == Tokens.EOS_TENSOR.value).all(dim=-1)]
#     seq_pred = y_pred_finished[i]
#     cleaned_seq_pred = seq_pred[~(seq_pred == Tokens.EOS_TENSOR.value).all(dim=-1)]
#     lengths.append(cleaned_seq.shape)
#     pred_lengths.append(cleaned_seq_pred.shape)

# logger.log(LogChannels.TRAINING, f"lengths: {lengths}")
# logger.log(LogChannels.TRAINING, f"pred len: {pred_lengths}")