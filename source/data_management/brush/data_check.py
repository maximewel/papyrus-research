import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)


from source.model.blocks.constants.sequence_to_image import ImageHelper
import cv2

from source.data_management.brush.brush_dataset import BrushDataset, StrokeMode
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from source.model.blocks.constants.tokens import Tokens

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

BATCH_SIZE = 1
BRUSH_ROOT = "data/handwriting/BRUSH"
patch_dimension = (1,1)

def remove_padding(sequence: torch.Tensor) -> torch.Tensor:
    # Find the first occurrence of padding_value in the row (if any)
    return sequence[sequence != Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value].view(-1, 2)  # Keep only non-padding values


def test_dataset():
    # Init data
    dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=patch_dimension, save_to_file=False, strokemode=StrokeMode.SUBSTROKES, 
                           normalize_coordinate_sequences=False, normalize_pixel_values=True)
    
    dataset.transform_to_batch()

    unshuffled_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    # Create the initial plot
    fig, axes = plt.subplots(3)

    axes[0].title.set_text('Live data for prediction')
    axes[1].title.set_text('Ref image')
    axes[2].title.set_text('Padding image')
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[2].set_axis_off()

    plt.ion()
    plt.show()  # Keep the final plot open

    first = True

    displayed_image = None

    unfolder = torch.nn.Fold(output_size=dataset.target_image_shape, kernel_size=patch_dimension, stride=patch_dimension)

    for batch in unshuffled_loader:

        (images, masks, sequences), labels = batch
        label = labels[0].int().tolist()

        sequence = sequences[0]

        image = ImageHelper.create_image(torch.nn.functional.pad(sequence.int(), (0, 1)).numpy(), canvas_size=dataset.target_image_shape)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        cv2.circle(image, (label[0], label[1]), radius=0, color=(220,20,60), thickness=2)

        ref_image = images.permute(0,2,1)
        mask = masks.unsqueeze(0).float()

        ref_image = unfolder(ref_image)[0][0].numpy()
        mask = unfolder(mask)[0][0].numpy()

        ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2RGB)
        mask_image_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if first:
            displayed_image = axes[0].imshow(image, aspect="auto")
            displayed_ref = axes[1].imshow(ref_image_rgb, aspect="auto")
            displayed_mask = axes[2].imshow(mask_image_rgb, aspect="auto")
            first = False
        else:
            displayed_image.set_data(image)
            displayed_ref.set_data(ref_image_rgb)
            displayed_mask.set_data(mask_image_rgb)

        fig.canvas.draw()  # Redraw the canvas
        fig.canvas.flush_events()  # Flush any GUI events
        time.sleep(0.01)  # Pause for visualization

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep the final plot open

def test_lstm_dataset():
        # Init data
    dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=(1,1), save_to_file=False, strokemode=StrokeMode.SUBSTROKES, normalize_coordinate_sequences=False, 
                        lstm_forecast_length=25)
    dataset.transform_to_batch()

    unshuffled_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    # Create the initial plot
    fig, ax = plt.subplots()

    plt.ion()
    plt.show()  # Keep the final plot open

    first = True

    displayed_image = None

    for batch in unshuffled_loader:

        (images, masks, sequences), labels = batch
        label = labels[0].int().tolist()

        #sequence = remove_padding(sequences[0])
        sequence = sequences[0]

        image = ImageHelper.create_image(torch.nn.functional.pad(sequence.int(), (0, 1)).numpy(), canvas_size=dataset.target_image_shape)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        cv2.circle(image, (label[0], label[1]), radius=0, color=(220,20,60), thickness=2)

        if first:
            displayed_image = ax.imshow(image, aspect="auto")
            first = False 
        else:
            displayed_image.set_data(image)

        fig.canvas.draw()  # Redraw the canvas
        fig.canvas.flush_events()  # Flush any GUI events
        time.sleep(0.01)  # Pause for visualization

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep the final plot open

if __name__ == "__main__":
    test_dataset()
    #test_lstm_dataset()