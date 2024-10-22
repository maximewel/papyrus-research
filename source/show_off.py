import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *

from source.model.hw_model import HwTransformer
from source.data_management.brush.brush_dataset import BrushDataset
from source.model.hw_model import HwTransformer
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device

from source.model.blocks.helper.tensor_utils import TensorUtils

import torch
import matplotlib.pyplot as plt
from time import sleep

folder_model_to_load = "2024-10-22 20-59-40"

PATCHES_DIM = (4, 4)

MIN_DIM_SHOWOFF = 50

STOP_CONDITION_IDENTICAL_OUTPUTS = 5

DENORMALIZE_SEQUENCES = True

REPLACE_WITH_GOLDEN = False

tolerance = 0.00001
def has_identical_last_values(tensor, n: int) -> bool:
    """Return whether the last N values of the tensor are exact"""
    last_rows = tensor[-n:, :]    
    are_identical = torch.all(torch.abs(last_rows - last_rows[0, :]) < tolerance, dim=1).all()    
    return are_identical.item()

def image_from_result(resultSignal, mult_tensor):
    resultSignalAsInt = (resultSignal * mult_tensor).int()
    #Pad to obtain original third dimension, 'penup'
    resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
    result_image = ImageHelper.create_image(resultSignalAsInt.cpu().numpy(), orig_signal.shape)
    return result_image

if __name__ == "__main__":
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, folder_model_to_load)
        filepath = os.path.join(folderPath, MODEL_FILENAME)

        print(f"Loading model from: {filepath}")
        
        model: HwTransformer = torch.load(filepath)

        model.eval()

        # Init data
        dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=PATCHES_DIM, save_to_file=False, strokemode=True, 
                               normalize_coordinate_sequences=True, normalize_pixel_values=True)
        dataset.transform_to_batch()
        
        mult_tensor = torch.tensor(dataset.target_image_shape, dtype=int, device=device) if DENORMALIZE_SEQUENCES else 1
        
        plt.ion()

        nextIndex = 0
        while nextIndex < len(dataset):
            nextIndex += 1
            while len(dataset.signals[nextIndex]) < MIN_DIM_SHOWOFF:
                nextIndex += 1

            #Create image
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 2 columns
            axs[0].set_title('Original image')
            axs[0].axis('off')
            axs[1].set_title('Reconstructed image from sig')
            axs[1].axis('off')
            axs[2].set_title('Predicted sequence from image, reconstructed')
            axs[2].axis('off')
            plt.show()

            image, padding, originalSignal = dataset.patchified_images[nextIndex], dataset.patches_padding_masks[nextIndex], dataset.signals_as_tensor[nextIndex]
            image, padding, originalSignal = image.to(device), padding.to(device), originalSignal.to(device)
            print(f"Selecting random signal n°{nextIndex} of length {originalSignal.shape[0]}")
            fig.suptitle(f'Show-off on signal n°{nextIndex}, length {originalSignal.shape[0]}')

            #Re-create images for both
            orig_image = dataset.images[nextIndex]

            sig_l = TensorUtils.true_seq_lengths_of_tensor(originalSignal)
            originalSignalAsInt = (originalSignal[:sig_l] * mult_tensor).int()
            originalSignalAsInt = torch.nn.functional.pad(originalSignalAsInt, (0, 1))
            orig_signal = ImageHelper.create_image(originalSignalAsInt.cpu().numpy())

            axs[0].imshow(orig_image, cmap='gray', vmin=0, vmax=255)
            axs[1].imshow(orig_signal, cmap='gray', vmin=0, vmax=255)

            with torch.no_grad():
                #Limit generation to avoid infinite autoregression
                resultSignal = originalSignal[:1]
                working_signal = originalSignal[:1]

                result_image = image_from_result(resultSignal, mult_tensor)
                result_display = axs[2].imshow(result_image, cmap='gray', vmin=0, vmax=255)

                stop_signal = False

                i = 1
                while not stop_signal:
                    print(f"\rGenerating point {i}")

                    res, eos_logit = model.forward(image.unsqueeze(0), padding.unsqueeze(0), working_signal.unsqueeze(0))
                    #Used to avoid OOM during autoregression
                    res = res.detach()
                    resultSignal = torch.vstack([resultSignal, res])
                    print(f"Last 5: {resultSignal[:-5]}")
                    if REPLACE_WITH_GOLDEN:
                        working_signal = torch.vstack([working_signal, originalSignal[i+1]])
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

                    if has_identical_last_values(resultSignal, STOP_CONDITION_IDENTICAL_OUTPUTS):
                        print(f"Early stop - identical values loop detected in the last {STOP_CONDITION_IDENTICAL_OUTPUTS} outputs")
                        stop_signal = True

                    i += 1

                    stop_signal |= i > 500

            print(f"Got final signal of length {resultSignal.shape[0]}")
            entry = input("Press to next, enter anything stop:")
            plt.close(fig)

            if(entry):
                plt.ioff()
                break
