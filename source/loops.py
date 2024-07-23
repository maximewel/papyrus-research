"""Contains the train and test loops for the model(s)"""
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeElapsedColumn, BarColumn

from source.model.hw_model import HwTransformer
from source.logging.log import logger, LogChannels

# def obtain_all_predictions(transformer: HwTransformer, images: Tensor, masks: Tensor, sequences: Tensor, 
#                            device: torch.device, progress: Progress = None, seq_progress_bar_id: any = None) -> Tensor:
#     """Scan the list of sequences. Remove the finished 
#     Args
#     -----

#     Returns
#     -----
#         Tensor: predictions - the list of predictions for each image 
#     """
#     iterative_sequences = sequences[:, :1, :]

#     batch_size = images.shape[0]
#     # Track original positions
#     original_index = torch.arange(batch_size, device=device)
#     # Track masking position
#     mask_memory = torch.zeros(()).bool()
#     pred_sequences = [[] for _ in range(batch_size)]

#     EOS_TOKEN = torch.full((2,), fill_value=Tokens.COORDINATE_SEQUENCE_EOS.value, device=device)
#     iterative_images = images.clone()
#     iterative_masks = masks.clone()
#     iterative_indexes = original_index.clone()

#     i = 1 #Starts at 1 as we already have the first value loaded
#     min_seq_len, max_seq_len = TensorUtils.min_max_sequence_len_in_tensor(sequences)

#     logger.log(LogChannels.DEBUG, f"Sequences from {min_seq_len} to {max_seq_len}")

#     if progress != None and seq_progress_bar_id != None:
#         progress.reset(seq_progress_bar_id, total=max_seq_len)

#     while len(iterative_sequences) > 0:
#         if i > max_seq_len:
#             raise Exception(f"Exception when running autoregressive computation: Expected sequences of [{min_seq_len}-{max_seq_len}], while round is {i}")
#         #Generate and Store the prediction
#         next_y_pred = transformer.forward(iterative_images, iterative_masks, iterative_sequences)
#         #Store alongside the index of the sequence to be able to regroup the sequences at the end
#         for iter_index, in_batch_index in enumerate(iterative_indexes):
#             pred_sequences[in_batch_index].append(next_y_pred[iter_index])

#         #Place next truth golden token into sequences, Remove sequences that finish with EOS
#         i+=1
#         iterative_sequences = sequences[:, :i, :]
#         mask = torch.logical_or((iterative_sequences[:, -1, :] == EOS_TOKEN).all(dim=1), mask_memory)

#         if mask.any():
#             iterative_sequences = iterative_sequences[~mask]
#             iterative_images = images[~mask]
#             iterative_masks = masks[~mask]
#             iterative_indexes = original_index[~mask]

#             mask_memory = mask

#         if progress:
#             progress.advance(seq_progress_bar_id)
    
#     # Combine predictions based on original indices by re-placing the predicted tokens at the right place in an output tensor
#     output_tensors = torch.full_like(sequences, Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value).float()
#     for i in range(len(original_index)):
#         sequence_as_tensor = torch.cat(pred_sequences[i], dim=0).view(-1, 2)
#         output_tensors[i, :len(sequence_as_tensor)] = sequence_as_tensor[:]

#     return output_tensors

def do_training(model: HwTransformer, train_loader: DataLoader, test_loader: DataLoader, device: torch.device):
    """Do the training and return the given model"""
    N_EPOCHS = 20
    LR = 0.005

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
                (images, masks, sequences), labels = batch
                images, masks, sequences, labels = images.to(device), masks.to(device), sequences.to(device), labels.to(device)

                # Iterate over the sequences untill all are over. 
                y_pred_finished = model.forward(images, masks, sequences)
                asStr = [f"|{labels[i].cpu().detach().numpy()}-{y_pred_finished[i].cpu().detach().numpy()}|" for i in range(len(y_pred_finished))]
                logger.log(LogChannels.LOSSES, f"labels-pred ={' | '.join(asStr)}")

                loss = criterion(y_pred_finished, labels)
                logger.log(LogChannels.LOSSES, f"loss = {loss.detach().cpu().item()}")

                train_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.advance(batch_progress_bar)

                del images, masks, sequences, labels, y_pred_finished, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            progress.advance(epoch_progress_bar)

            train_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{N_EPOCHS} Train loss: {train_loss:.2f}")
            train_losses.append(train_loss)

            # Test loop
            with torch.no_grad():
                test_loss = 0.0
                for batch in test_loader:
                    (images, masks, sequences), labels = batch
                    images, masks, sequences, labels = images.to(device), masks.to(device), sequences.to(device), labels.to(device)

                    # Iterate over the sequences untill all are over. 
                    y_pred = model.forward(images, masks, sequences)

                    loss = criterion(y_pred, labels)
                    test_loss += loss.detach().cpu().item()

                    del images, masks, sequences, labels, y_pred, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                test_loss /= len(test_loader)
                print(f"Epoch {epoch + 1}/{N_EPOCHS} Test loss: {test_loss:.2f}")
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