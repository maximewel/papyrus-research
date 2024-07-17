import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *

from source.model.hw_model import HwTransformer
from source.data.brush.brush_dataset import BrushDataset, StrokeMode
from source.model.hw_model import HwTransformer
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device

from source.model.blocks.helper.tensor_utils import TensorUtils

import torch
import matplotlib.pyplot as plt


folder_model_to_load = "2024-06-27 23-53-54"

PATCHES_DIM = (8, 8)

if __name__ == "__main__":

        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, folder_model_to_load)

        filepath = os.path.join(folderPath, MODEL_FILENAME)

        print(f"Loading model from: {filepath}")

        model: HwTransformer =  torch.load(filepath)

        model.eval()

        # Init data
        dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=PATCHES_DIM, display_stats=True, save_to_file=False, strokemode=StrokeMode.SUBSTROKES)
        dataset.transform_to_batch()
        
        nextIndex = 0
        while nextIndex < len(dataset):
            nextIndex += 1
            while len(dataset.signals[nextIndex]) < 50:
                nextIndex += 1

            image, padding, originalSignal = dataset.patchified_images[nextIndex], dataset.patches_padding_masks[nextIndex], dataset.signals_as_tensor[nextIndex]
            image, padding, originalSignal = image.to(device), padding.to(device), originalSignal.to(device)
            print(f"Selecting random signal n°{nextIndex} of length {originalSignal.shape[0]}")

            with torch.no_grad():
                #Limit generation to avoid infinite autoregression
                resultSignal = originalSignal[:1]
                for i in range(originalSignal.shape[0]):
                    print(f"{1+i}/{originalSignal.shape[0]}")
                    res = model.forward(image.unsqueeze(0), padding.unsqueeze(0), resultSignal.unsqueeze(0))
                    #Used to avoid OOM during autoregression
                    res = res.detach()
                    resultSignal = torch.vstack([resultSignal, res])
                    print(f"Got {res}, final signal last 5 \n{resultSignal[-5:]}")

            print(f"Got final signal of length {resultSignal.shape[0]}")

            mult_tensor = torch.tensor(dataset.target_image_shape, dtype=int, device=resultSignal.device)
            resultSignalAsInt = (resultSignal * mult_tensor).int()
            sig_l = TensorUtils.true_seq_lengths_of_tensor(originalSignal)
            originalSignalAsInt = (originalSignal[:sig_l] * mult_tensor).int()

            print(originalSignal[:10])
            print(originalSignalAsInt[:10])

            print(resultSignal[:10])
            print(resultSignalAsInt[:10])

            #Pad to obtain original third dimension, 'penup'
            resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
            originalSignalAsInt = torch.nn.functional.pad(originalSignalAsInt, (0, 1))

            #Re-create images for both
            orig_image = dataset.images[nextIndex]
            orig_signal = ImageHelper.create_image(originalSignalAsInt.cpu().numpy())
            final_image = ImageHelper.create_image(resultSignalAsInt.cpu().numpy())

            fig, axs = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 2 columns

            fig.suptitle(f'Show-off on signal n°{nextIndex}, length {originalSignal.shape[0]}')

            axs[0].imshow(orig_image, cmap='gray', vmin=0, vmax=255)
            axs[0].set_title('Original image')
            axs[0].axis('off')

            axs[1].imshow(orig_signal, cmap='gray', vmin=0, vmax=255)
            axs[1].set_title('Reconstructed image from sig')
            axs[1].axis('off')

            axs[2].imshow(final_image, cmap='gray', vmin=0, vmax=255)
            axs[2].set_title('Predicted sequence from image, reconstructed')
            axs[2].axis('off')

            plt.show(block=False)
            entry = input("Press to next, enter anything stop:")
            plt.close(fig)

            if(entry):
                 break