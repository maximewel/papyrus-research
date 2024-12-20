import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *

from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.hw_model import HwTransformer
from source.model.blocks.hw_lstm import HwLstm
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device
from torch.nn.utils.rnn import pack_sequence
from source.model.blocks.constants.tokens import Tokens

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

folder_model_to_load = "BRUSH_AUGMENTED"
USE_LSTM = False
folder_lstm_model_to_load = ""

PATCHES_DIM = (16, 16)

MIN_DIM_SHOWOFF = 50

STOP_CONDITION_IDENTICAL_OUTPUTS = 20

DENORMALIZE_SEQUENCES = False
REPLACE_WITH_GOLDEN = False

REPLACE_ON_SKELETON = False
REPLACE_ON_SKELETON_ON_RES = False

IMAGE_MAX_SHAPE = (112, 112)

WRITER_ID = 1

tolerance = 0.0001
def has_identical_last_values(tensor, n: int) -> bool:
    """Return whether the last N values of the tensor are exact"""
    if tensor.shape[0] < n:
        return False
    last_rows = tensor[-n:, :]
    are_identical = torch.all(torch.abs(last_rows - last_rows[0, :]) < tolerance, dim=1).all()    
    return are_identical.item()

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

if __name__ == "__main__":
        folderPath = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, folder_model_to_load)
        filepath = os.path.join(folderPath, MODEL_FILENAME)

        print(f"Loading model from: {filepath}")
        
        model: HwTransformer = torch.load(filepath)
        model.eval()

        if USE_LSTM:
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
        from source.logging.log import logger, LogChannels
        logger.add_log_channel(LogChannels.DATA)
        
        dataset = HandWrittingDataset("BRUSH_100.100_test_m_aug")

        unfolder = torch.nn.Fold(output_size=IMAGE_MAX_SHAPE, kernel_size=PATCHES_DIM, stride=PATCHES_DIM)
        
        mult_tensor = torch.tensor(output_size=IMAGE_MAX_SHAPE, dtype=int, device=device) if DENORMALIZE_SEQUENCES else 1
        
        plt.ion()

        nextIndex = 1500
        while nextIndex < len(dataset):
            image, patched_image, padding, current_signal, label = dataset[nextIndex]
            while len(current_signal) < MIN_DIM_SHOWOFF:
                image, patched_image, padding, current_signal, label = dataset[nextIndex]
                nextIndex += 1
            
            current_signal = torch.tensor(current_signal, device=device)
            patched_image = torch.tensor(patched_image, device=device)
            padding = torch.tensor(padding, device=device)

            #Create image
            fig, axs = plt.subplots(1, 3, figsize=(10, 10))
            axs[0].set_title('Original image')
            axs[0].axis('off')
            axs[1].set_title('Patched image given to transformer, unpatched')
            axs[1].axis('off')
            axs[2].set_title('Predicted sequence from image, reconstructed')
            axs[2].axis('off')
            plt.show(block=False)

            print(f"Selecting random signal n°{nextIndex} of length {len(current_signal)}")
            fig.suptitle(f'Show-off on signal n°{nextIndex}, length {len(current_signal)}')

            #Re-create images for both
            orig_image = image_from_result(current_signal, mult_tensor, IMAGE_MAX_SHAPE)
            print(image.shape)
            print(torch.tensor(image).unsqueeze(0).shape)
            # patched_image_unfolded = unfolder(torch.tensor(image).unsqueeze(0).permute(0,2,1))[0][0].numpy()

            axs[0].imshow(orig_image, cmap='gray')
            # axs[1].imshow(patched_image_unfolded, cmap='gray')

            with torch.no_grad():
                #Limit generation to avoid infinite autoregression
                resultSignal = current_signal[:1]
                working_signal = current_signal[:1]

                result_image = image_from_result(resultSignal, mult_tensor, IMAGE_MAX_SHAPE)
                result_display = axs[2].imshow(result_image, cmap='gray', vmin=0, vmax=1)

                stop_signal = False

                i = 1
                while not stop_signal:
                    print(f"\rGenerating point {i}")

                    print(f"Initial signal around {i}:\n{current_signal[max(i-5,0):i+5]}")
                    print(f"Last 5 result:\n{resultSignal[-5:]}")
                    print(f"Last 5 working:\n{working_signal[-5:]}")

                    res = model.forward(patched_image.unsqueeze(0), padding.unsqueeze(0), pack_sequence(working_signal.unsqueeze(0)))
                    if not DENORMALIZE_SEQUENCES:
                        res = torch.round(res)
                        
                    #Used to avoid OOM during autoregression
                    res = res.detach()
                    print(f"Generated {res}")

                    if REPLACE_ON_SKELETON and REPLACE_ON_SKELETON_ON_RES:
                        print(f"Res not on skel: {res}")
                        res = closest_point_on_skeletton(orig_image, res)
                        print(f"Res on skel: {res}")

                    resultSignal = torch.vstack([resultSignal, res])

                    if REPLACE_ON_SKELETON and not REPLACE_ON_SKELETON_ON_RES:
                        res = closest_point_on_skeletton(orig_image, res)

                    if REPLACE_WITH_GOLDEN:
                        working_signal = torch.vstack([working_signal, current_signal[i]])
                    else:
                        working_signal = torch.vstack([working_signal, res])

                    result_image = image_from_result(resultSignal, mult_tensor, IMAGE_MAX_SHAPE)
                    result_display.set_data(result_image)

                    fig.canvas.draw()  # Redraw the canvas
                    fig.canvas.flush_events()  # Flush any GUI events

                    # Check if we should stop
                    if has_identical_last_values(resultSignal, STOP_CONDITION_IDENTICAL_OUTPUTS) or (REPLACE_WITH_GOLDEN and i >= len(current_signal)) or (i > 2 * len(current_signal)):
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