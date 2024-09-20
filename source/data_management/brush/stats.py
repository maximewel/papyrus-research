"""
This module is used to display stats about the BRUSH dataset
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from source.data_management.brush.brush_dataset import BrushDataset, StrokeMode

BRUSH_ROOT = "data/handwriting/BRUSH"

def get_stats(dataset):
    #Get all images length
    images_len = [image.shape for image in dataset.images]

    #Get all X 
    images_widths = images_len[:][0]
    images_heigth = images_len[:][1]

    print(f"Maximum shape of images: {np.max(images_widths)} * {np.max(images_heigth)}")
    print(f"Average size of images: {np.mean(images_widths)} * {np.mean(images_heigth)}")

    sequences_len = [len(signal) for signal in dataset.signals]

    print(f"Stat of sequences: Range [{np.min(sequences_len)} - {np.max(sequences_len)}]")
    print(f"Average length, deviation of sequences: {np.mean(images_widths)}, {np.std(images_heigth)}")

    #Display histogram of sequences
    plt.figure()
    plt.hist(sequences_len, bins=range(0, np.max(sequences_len), 50))

dataset_substrokes = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, patches_dim=(1,1),
                           strokemode=StrokeMode.SUBSTROKES)
get_stats(dataset_substrokes)

dataset_strokes = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, patches_dim=(1,1),
                           strokemode=StrokeMode.STROKES)
get_stats(dataset_strokes)

plt.show()