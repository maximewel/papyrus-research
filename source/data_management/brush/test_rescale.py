"""
This module is used to display stats about the BRUSH dataset
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from source.data_management.brush.brush_dataset import BrushDataset
from source.model.blocks.constants.files import *
from source.logging.log import logger, LogChannels



logger.add_log_channel(LogChannels.DATA)
dataset = BrushDataset(brush_root=BRUSH_ROOT, patches_dim=(1,1),
                        strokemode=True, window_size=0,
                        normalize_pixel_values=True, normalize_coordinate_sequences=True)

#Get all X 
images_widths, images_heigth = [image.shape[0] for image in dataset.images], [image.shape[1] for image in dataset.images]

print(f"Maximum shape of images: {np.max(images_widths)} * {np.max(images_heigth)}")
print(f"Average size of images: {np.mean(images_widths)} * {np.mean(images_heigth)}")

sequences_len = [len(signal) for signal in dataset.signals]

print(f"Stat of sequences: Range [{np.min(sequences_len)} - {np.max(sequences_len)}]")
print(f"Average length, deviation of sequences: {np.mean(images_widths)}, {np.std(images_heigth)}")

print(f"Dataset has {len(dataset.signals)} sequences and {sum([len(signal) for signal in dataset.signals])} points")

#Display histogram of sequences
plt.figure()
plt.hist(sequences_len, bins=range(0, np.max(sequences_len), 50))
plt.title(f"Histogram of the sequences length of the BRUSH dataset with {stroke}")
plt.xlabel("Sequences lenghts in points")
plt.ylabel("Amount of sequences")

plt.show()