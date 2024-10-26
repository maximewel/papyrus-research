import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *

from source.data_management.brush.brush_dataset import BrushDataset
from source.model.hw_model import HwTransformer
from source.model.blocks.hw_lstm import HwLstm
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device
from torch.nn.utils.rnn import pack_sequence
from source.model.blocks.constants.tokens import Tokens

import torch
import matplotlib.pyplot as plt

folder_model_to_load = "2024-10-26 23-21-31"
folder_lstm_model_to_load = "2024-10-24 22-39-02"

PATCHES_DIM = (8, 8)

MIN_DIM_SHOWOFF = 50

STOP_CONDITION_IDENTICAL_OUTPUTS = 5

DENORMALIZE_SEQUENCES = True

REPLACE_WITH_GOLDEN = False

tolerance = 0.0001
def has_identical_last_values(tensor, n: int) -> bool:
    """Return whether the last N values of the tensor are exact"""
    last_rows = tensor[-n:, :]    
    are_identical = torch.all(torch.abs(last_rows - last_rows[0, :]) < tolerance, dim=1).all()    
    return are_identical.item()

def image_from_result(resultSignal, mult_tensor):
    """
        Create an image from the result
        Security: If negative coordinates exist, adjust image
    """
    print(f"Result signal shape: {resultSignal.shape}")
    for dim in [0, 1]:
        min_dim = torch.min(resultSignal[:, dim])
        if min_dim < 0 and min_dim != Tokens.COORDINATE_SEQUENCE_EOS.value:
            print(f"Adding {-min_dim} to dim {dim}")
            resultSignal[:-1, dim] += -min_dim

    resultSignalAsInt = (resultSignal * mult_tensor).int()
    #Pad to obtain original third dimension, 'penup'
    resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
    result_image = ImageHelper.create_image(resultSignalAsInt.cpu().numpy())
    return ImageHelper.revert_image(result_image)

if __name__ == "__main__":
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, folder_model_to_load)
        filepath = os.path.join(folderPath, MODEL_FILENAME)

        print(f"Loading model from: {filepath}")
        
        model: HwTransformer = torch.load(filepath)
        model.eval()

        if folder_lstm_model_to_load is not None:
            #Load pre-trained LSTM model
            folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, LSTM_FOLDER, folder_lstm_model_to_load)
            filepath = os.path.join(folderPath, MODEL_FILENAME)
            print(f"Loading LSTM model from: {filepath}")
            lstm_model: HwLstm = torch.load(filepath)
            #Freeze model as we have a pre-trained LSTM model that doesnt need to learn in this step
            lstm_model.eval()
        else:
            lstm_model = None

        # Init data
        dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=PATCHES_DIM, save_to_file=False, strokemode=True, 
                               normalize_coordinate_sequences=True, normalize_pixel_values=True)
        dataset.transform_to_batch()
        unfolder = torch.nn.Fold(output_size=dataset.target_image_shape, kernel_size=PATCHES_DIM, stride=PATCHES_DIM)
        
        mult_tensor = torch.tensor(dataset.target_image_shape, dtype=int, device=device) if DENORMALIZE_SEQUENCES else 1
        
        plt.ion()

        nextIndex = 0
        while nextIndex < len(dataset):
            nextIndex += 1
            while len(dataset.signals[nextIndex]) < MIN_DIM_SHOWOFF:
                nextIndex += 1

            current_signal = dataset.signals[nextIndex]

            #Create image
            fig, axs = plt.subplots(1, 4, figsize=(10, 10))
            axs[0].set_title('Original image')
            axs[0].axis('off')
            axs[1].set_title('Reconstructed image from sig')
            axs[1].axis('off')
            axs[2].set_title('Patched image given to transformer, unpatched')
            axs[2].axis('off')
            axs[3].set_title('Predicted sequence from image, reconstructed')
            axs[3].axis('off')
            plt.show(block=False)

            image, padding, originalSignal = dataset.patchified_images[nextIndex], dataset.patches_padding_masks[nextIndex], dataset.signals_as_tensor[nextIndex]
            image, padding, originalSignal = image.to(device), padding.to(device), torch.tensor(originalSignal, dtype=torch.float).to(device)
            print(f"Selecting random signal n°{nextIndex} of length {len(current_signal)}")
            fig.suptitle(f'Show-off on signal n°{nextIndex}, length {len(current_signal)}')

            #Re-create images for both
            orig_image = dataset.images[nextIndex]
            orig_image_reconstructed = image_from_result(originalSignal, mult_tensor)
            patched_image_unfolded = unfolder(image.cpu().unsqueeze(0).permute(0,2,1))[0][0].numpy()

            print(orig_image)
            axs[0].imshow(orig_image, cmap='gray')
            axs[1].imshow(orig_image_reconstructed, cmap='gray')
            axs[2].imshow(patched_image_unfolded, cmap='gray')

            with torch.no_grad():
                #Limit generation to avoid infinite autoregression
                resultSignal = originalSignal[:1]
                working_signal = originalSignal[:1]

                result_image = image_from_result(resultSignal, mult_tensor)
                result_display = axs[3].imshow(result_image, cmap='gray', vmin=0, vmax=1)

                stop_signal = False

                i = 1
                while not stop_signal:
                    print(f"\rGenerating point {i}")

                    print(f"Initial signal around {i}:\n{originalSignal[max(i-5,0):i+5]}")
                    print(f"Last 5 result:\n{resultSignal[-5:]}")
                    print(f"Last 5 working:\n{working_signal[-5:]}")

                    res, eos_logit = model.forward(image.unsqueeze(0), padding.unsqueeze(0), pack_sequence(working_signal.unsqueeze(0)))
                    #Used to avoid OOM during autoregression
                    res = res.detach()
                    print(f"Generated {res}")
                    resultSignal = torch.vstack([resultSignal, res])
                    if REPLACE_WITH_GOLDEN:
                        working_signal = torch.vstack([working_signal, originalSignal[i]])
                    else:
                        working_signal = torch.vstack([working_signal, res])

                    result_image = image_from_result(resultSignal, mult_tensor)
                    result_display.set_data(result_image)

                    fig.canvas.draw()  # Redraw the canvas
                    fig.canvas.flush_events()  # Flush any GUI events

                    # Apply sigmoid to get the stop probability, convert using standard 
                    stop_probability = torch.sigmoid(eos_logit)
                    print(f"stop proba: {stop_probability}")
                    # Check if we should stop
                    stop_signal |= stop_probability >= 0.5

                    if has_identical_last_values(resultSignal, STOP_CONDITION_IDENTICAL_OUTPUTS) or (REPLACE_WITH_GOLDEN and i >= len(current_signal)):
                        print(f"Early stop - identical values loop detected in the last {STOP_CONDITION_IDENTICAL_OUTPUTS} outputs")
                        stop_signal = True

                    i += 1

            print(f"Got final signal of length {resultSignal.shape[0]}")
            entry = input("Press to next, enter anything stop:")
            plt.close(fig)

            if(entry):
                plt.ioff()
                plt.close()
                break