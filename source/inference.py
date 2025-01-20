import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *

from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.hw_model import HwTransformer
from source.model.blocks.hw_lstm import HwLstm
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device
from torch.nn.utils.rnn import pack_sequence
from source.model.blocks.constants.tokens import Tokens
from source.model.blocks.constants.datasets_library import *

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

#folder_model_to_load = "brush_96_10epochs_pred"
folder_model_to_load = "best_m_20_epochs"

PATCHES_DIM = (16, 16)

MIN_DIM_SHOWOFF = 50

STOP_CONDITION_IDENTICAL_OUTPUTS = 20

DENORMALIZE_SEQUENCES = False
REPLACE_WITH_GOLDEN = False

REPLACE_ON_SKELETON = False
REPLACE_ON_SKELETON_ON_RES = False

SHOW_WEIGHTS = False
SHOW_WEIGHTS_SIG = 10

IMAGE_MAX_SHAPE = (96, 96)

CONTINUE = True

tolerance = 0.0001
def has_identical_last_values(tensor, n: int) -> bool:
    """Return whether the last N values of the tensor are exact"""
    if tensor.shape[0] < n:
        return False
    last_rows = tensor[-n:, :]
    are_identical = torch.all(torch.abs(last_rows - last_rows[0, :]) < tolerance, dim=1).all()    
    return are_identical.item()

## Misc used for report
def plot_patch_attention_multiple_layers(patchified_images, attention_weights_list, title="Patch Attention Map Across Layers"):
    """
    Visualize the attention map for patches across multiple layers.
    
    Args:
        patchified_images: Tensor of shape [n_patches, patch_size^2]
            Patches from which the image is reconstructed.
        attention_weights_list: List of tensors, each of shape [n_heads, n_patches, n_patches]
            Attention weights for each layer.
        title: Title for the plot.
    """
    import matplotlib.pyplot as plt
    
    # Reconstruct the image from patchified images
    patch_size = int(patchified_images.shape[1] ** 0.5)  # Assume patches are square
    n_patches_side = int(len(patchified_images) ** 0.5)  # n_patches_side x n_patches_side patches
    reconstructed_image = patchified_images.view(n_patches_side, n_patches_side, patch_size, patch_size)
    reconstructed_image = reconstructed_image.permute(0, 2, 1, 3).contiguous()  # Reorder for full image
    reconstructed_image = reconstructed_image.view(
        n_patches_side * patch_size, n_patches_side * patch_size
    )  # Combine patches into full image

    # Determine the number of layers and create subplots
    n_layers = len(attention_weights_list)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(16, 6))

    # Plot attention maps for each layer
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        print(f"Attention weights: {attention_weights.shape}")

        # Average attention weights across heads and reshape to 6x6
        print(f"Mean: {attention_weights[0].mean(dim=0).shape}")
        attention_weights = attention_weights[0].mean(dim=0).view(6, 6)
        
        # Plot attention map
        ax = axes[layer_idx]
        im = ax.imshow(attention_weights.cpu().detach().numpy(), cmap="viridis")
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xlabel("Key Patches")
        ax.set_ylabel("Query Patches")
        fig.colorbar(im, ax=ax, label="Attention Weight")

    # Plot reconstructed image
    ax = axes[-1]
    ax.imshow(reconstructed_image.cpu().detach().numpy(), cmap="gray")
    ax.set_title("Reconstructed Image")
    ax.axis("off")  # Hide axis for the image

    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    input("Press to continue")

def closest_point_on_skeletton(image_skeleton: np.ndarray, point: torch.Tensor) -> torch.Tensor:
    """Return the closest point on skeletton"""
    try:
        #The loss is simply the smallest distance between the skeletton and the predicted coordinate
        image_point = np.ones(IMAGE_MAX_SHAPE)
        image_point[*point.squeeze(0).int().tolist()] = 0

        distances = cv2.distanceTransform(image_point.astype(np.uint8), cv2.DIST_L2, 3)

        image_skeleton_padded = np.zeros(IMAGE_MAX_SHAPE)
        image_skeleton_padded[:image_skeleton.shape[0], :image_skeleton.shape[1]] = image_skeleton
        distances_masked = distances * image_skeleton_padded

        loss = np.min(distances_masked[image_skeleton_padded == 1])

        if loss == 0:
            return res
        
        coords_min = np.argwhere(distances_masked == loss)[0]
        return torch.tensor(coords_min, device=point.device).unsqueeze(0)
    except Exception as e:
        print(f"Impossible to set point {point.unsqueeze(0)} on skeleton: {e}")
        return point

def image_from_result(resultSignal, mult_tensor, target_size):
    """
        Create an image from the result
        Security: If negative coordinates exist, adjust image
    """
    for dim in [0, 1]:
        min_dim = torch.min(resultSignal[:, dim])
        if min_dim < 0 and min_dim != Tokens.COORDINATE_SEQUENCE_EOS.value:
            resultSignal[:-1, dim] += -min_dim

    resultSignalAsInt = (resultSignal * mult_tensor).int()
    #Pad to obtain original third dimension, 'penup'
    resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
    result_image = ImageHelper.create_image(resultSignalAsInt.cpu().numpy(), target_size)
    return result_image

def ink(signal):
    diff = signal[1:] - signal[:-1]
    distances = torch.sqrt(torch.sum(diff**2, dim=1))
    return distances.sum()

if __name__ == "__main__":
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, folder_model_to_load)
        filepath = os.path.join(folderPath, MODEL_FILENAME)

        print(f"Loading model from: {filepath}")
        
        model: HwTransformer = torch.load(filepath)
        model.eval()

        # Init data
        from source.logging.log import logger, LogChannels
        logger.add_log_channel(LogChannels.DATA)
        
        dataset = HandWrittingDataset(BRUSH_96_96_VALID_S)

        unfolder = torch.nn.Fold(output_size=IMAGE_MAX_SHAPE, kernel_size=PATCHES_DIM, stride=PATCHES_DIM)
        
        mult_tensor = torch.tensor(output_size=IMAGE_MAX_SHAPE, dtype=int, device=device) if DENORMALIZE_SEQUENCES else 1
        
        plt.ion()

        nextIndex = 1
        while nextIndex < len(dataset):
            #Create image
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            axs[0].set_title('Original image')
            axs[0].axis('off')
            axs[1].set_title('Original image Live')
            axs[1].axis('off')
            axs[2].set_title('Predicted sequence from image, reconstructed')
            axs[2].axis('off')

            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')            
            plt.show(block=False)

            image, patched_image, padding, current_signal, label = dataset[nextIndex]
            nextIndex += 1
            
            current_signal = torch.tensor(current_signal, device=device)
            patched_image = torch.tensor(patched_image, device=device)
            padding = torch.tensor(padding, device=device)

            print(f"Selecting random signal n°{nextIndex} of length {len(current_signal)}")
            fig.suptitle(f'Show-off on signal n°{nextIndex}, length {len(current_signal)}')

            #Re-create images for both
            orig_image = image_from_result(current_signal, mult_tensor, IMAGE_MAX_SHAPE)
            print(image.shape)
            print(torch.tensor(image).unsqueeze(0).shape)

            axs[0].imshow(orig_image, cmap='gray')

            with torch.no_grad():
                #Limit generation to avoid infinite autoregression
                resultSignal = current_signal[:1]
                working_signal = current_signal[:1]

                live_orig_image = image_from_result(current_signal[:1], mult_tensor, IMAGE_MAX_SHAPE)
                live_orig_display = axs[1].imshow(live_orig_image, cmap='gray', vmin=0, vmax=1)

                result_image = image_from_result(resultSignal, mult_tensor, IMAGE_MAX_SHAPE)
                result_display = axs[2].imshow(result_image, cmap='gray', vmin=0, vmax=1)

                stop_signal = False

                i = 1
                total_signal_ink = ink(current_signal)
                while not stop_signal:
                    print(f"\rGenerating point {i}")

                    # print(f"Initial signal around {i}:\n{current_signal[max(i-5,0):i+5]}")
                    # print(f"Last 5 result:\n{resultSignal[-5:]}")
                    # print(f"Last 5 working:\n{working_signal[-5:]}")

                    if SHOW_WEIGHTS:
                        res, weights = model.forward(patched_image.unsqueeze(0), padding.unsqueeze(0), pack_sequence(working_signal.unsqueeze(0)), return_encoder_weights=True)
                        if i % SHOW_WEIGHTS_SIG == 0:
                            plot_patch_attention_multiple_layers(patched_image, weights)
                    else:
                        res = model.forward(patched_image.unsqueeze(0), padding.unsqueeze(0), pack_sequence(working_signal.unsqueeze(0)))
                    if not DENORMALIZE_SEQUENCES:
                        res = torch.round(res)
                        
                    #Used to avoid OOM during autoregression
                    res = res.detach()
                    print(f"Generated {res}")

                    if REPLACE_ON_SKELETON and REPLACE_ON_SKELETON_ON_RES:
                        res = closest_point_on_skeletton(orig_image, res)

                    resultSignal = torch.vstack([resultSignal, res])

                    if REPLACE_ON_SKELETON and not REPLACE_ON_SKELETON_ON_RES:
                        res = closest_point_on_skeletton(orig_image, res)

                    if REPLACE_WITH_GOLDEN:
                        working_signal = torch.vstack([working_signal, current_signal[i]])
                    else:
                        working_signal = torch.vstack([working_signal, res])

                    live_orig_image = image_from_result(current_signal[:i+1], mult_tensor, IMAGE_MAX_SHAPE)
                    live_orig_display.set_data(live_orig_image)

                    #ink: Trick to early stop
                    if ink(working_signal) > total_signal_ink:
                        print(f"Stop With INK")
                        stop_signal = True
                        resultSignal = resultSignal[:-1]

                    result_image = image_from_result(resultSignal, mult_tensor, IMAGE_MAX_SHAPE)
                    result_display.set_data(result_image)

                    fig.canvas.draw()  # Redraw the canvas
                    fig.canvas.flush_events()  # Flush any GUI events

                    # In case model spits EOS
                    if torch.equal(res.squeeze(0), Tokens.EOS_TENSOR.value):
                        print(f"EOS token detected ! Res is: {res.squeeze(0)}, pred token is: {Tokens.EOS_TENSOR.value}")
                        stop_signal = True

                    # Check if we should stop
                    if has_identical_last_values(resultSignal, STOP_CONDITION_IDENTICAL_OUTPUTS) or (REPLACE_WITH_GOLDEN and i >= len(current_signal)) or (i > 2 * len(current_signal)):
                        print(f"Early stop - identical values loop detected in the last {STOP_CONDITION_IDENTICAL_OUTPUTS} outputs")
                        stop_signal = True
                    
                    i += 1

            print(f"Got final signal of length {resultSignal.shape[0]}")
            plt.close(fig)
            del current_signal, working_signal, orig_image, result_image, live_orig_image

            if CONTINUE:
                entry = False
            else:
                entry = input("Press to next, enter anything stop:")

            if(entry):
                plt.ioff()
                plt.close()
                break