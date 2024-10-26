import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *

from source.data_management.brush.brush_dataset import BrushDataset
from source.model.blocks.hw_lstm import HwLstm
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device

import torch
import matplotlib.pyplot as plt

folder_model_to_load = "2024-10-24 22-39-02"

PATCHES_DIM = (1, 1)

MIN_DIM_SHOWOFF = 50

DENORMALIZE_SEQUENCES = True

REPLACE_WITH_GOLDEN = True

def image_from_result(resultSignal, mult_tensor):
    resultSignalAsInt = (resultSignal * mult_tensor).int()
    #Pad to obtain original third dimension, 'penup'
    resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
    result_image = ImageHelper.create_image(resultSignalAsInt.cpu().numpy())
    return result_image

if __name__ == "__main__":
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, LSTM_FOLDER, folder_model_to_load)
        filepath = os.path.join(folderPath, MODEL_FILENAME)

        print(f"Loading model from: {filepath}")
        
        model: HwLstm = torch.load(filepath)

        model.eval()

        # Init data
        dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=PATCHES_DIM, save_to_file=False, strokemode=True, 
                               normalize_coordinate_sequences=True, normalize_pixel_values=True, lstm_mode=True)
        dataset.transform_to_batch()
        mult_tensor = torch.tensor(dataset.target_image_shape, dtype=int, device=device) if DENORMALIZE_SEQUENCES else 1


        nextIndex = 0
        while nextIndex < len(dataset):
            nextIndex += 1
            while len(dataset.batchified_sequences[nextIndex]) < MIN_DIM_SHOWOFF:
                nextIndex += 1
            
            signal = dataset.batchified_sequences[nextIndex].to(device)

            with torch.no_grad():
                #Limit generation to avoid infinite autoregression
                resultSignal = signal[:1]
                working_signal = signal[:1]

                i = 1
                while i < len(signal):
                    print(f"\rGenerating point {i}")

                    print(f"Original signal first 20: {signal[:20]}")
                    print(f"working_signal: {working_signal}")

                    res = model.forward(working_signal, last_layer_mlp=True).detach()
                    print(f"Res: {res}")
                    #Used to avoid OOM during autoregression
                    resultSignal = torch.vstack([resultSignal, res])
                    print(f"Last 5: {resultSignal[-5:]}")
                    if REPLACE_WITH_GOLDEN:
                        working_signal = torch.vstack([working_signal, signal[i]])
                    else:
                        working_signal = torch.vstack([working_signal, res])

                    i += 1

            print(f"Got final signal of length {resultSignal.shape[0]}")

            #Generate final image from result sequence
            orig_image = image_from_result(signal, mult_tensor)
            result_image = image_from_result(resultSignal, mult_tensor)

            #Create image
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
            axs[0].set_title('Original image')
            axs[0].imshow(orig_image, cmap='gray', vmin=0, vmax=255)
            axs[0].axis('off')
            axs[1].set_title('Result image')
            axs[1].imshow(result_image, cmap='gray', vmin=0, vmax=255)
            axs[1].axis('off')           

            plt.show()

            entry = input("Press to next, enter anything stop:")

            if(entry):
                break