"""
This module is used to display stats about the BRUSH dataset
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from source.logging.log import logger, LogChannels
from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.blocks.constants.datasets_library import *

BRUSH_ROOT = "data/handwriting/BRUSH"

def plot_hist_on(dataset: BrushDataset, stroke, bin_size: int, ax):
    # Get all images length
    images = HandWrittingDataset.build_images(dataset.signals)
    images_widths = [image.shape[0] for image in images]
    images_heigth = [image.shape[1] for image in images]

    print(f"Maximum shape of images: {np.max(images_widths)} * {np.max(images_heigth)}")
    print(f"Average size of images: {np.mean(images_widths)} * {np.mean(images_heigth)}")

    sequences_len = [len(signal) for signal in dataset.signals]

    print(f"Stat of sequences: Range [{np.min(sequences_len)} - {np.max(sequences_len)}]")
    print(f"Average length, deviation of sequences: {np.mean(sequences_len)}, {np.std(sequences_len)}")

    print(f"Dataset has {len(dataset.signals)} sequences and {sum([len(signal) for signal in dataset.signals])} points")

    # Display histogram of sequences on the same figure
    ax.hist(sequences_len, bins=range(0, np.max(sequences_len), bin_size), alpha=0.7, label=f"{stroke}")
    ax.set_xlabel("Sequences lengths in points")
    ax.set_ylabel("Amount of sequences")
    ax.set_title("Histogram of the sequences length of the BRUSH dataset")
    ax.legend()

def get_stats(dataset: BrushDataset, stroke, bin_size: int):
    #Get all images length
    images = HandWrittingDataset.build_images(dataset.signals)
    images_widths, images_heigth = [image.shape[0] for image in images], [image.shape[1] for image in images]

    print(f"Maximum shape of images: {np.max(images_widths)} * {np.max(images_heigth)}")
    print(f"Average size of images: {np.mean(images_widths)} * {np.mean(images_heigth)}")

    sequences_len = [len(signal) for signal in dataset.signals]

    print(f"Stat of sequences: Range [{np.min(sequences_len)} - {np.max(sequences_len)}]")
    print(f"Average length, deviation of sequences: {np.mean(images_widths)}, {np.std(images_heigth)}")

    print(f"Dataset has {len(dataset.signals)} sequences and {sum([len(signal) for signal in dataset.signals])} points")

    #Display histogram of sequences
    plt.figure()
    bins_range = range(0, np.max(sequences_len) + bin_size, bin_size)
    plt.hist(sequences_len, bins=bins_range, color='orange')
    plt.title(f"Histogram of the sequences length of the BRUSH dataset, level of {stroke}")
    plt.xlabel("Sequences lenghts in points")
    plt.ylabel("Amount of sequences")

def before_after():
    #Get all images length
    single_dataset = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, separate_strokes=False, image_max_shape=(1000, 1000), single_signal=True, restrict_id=2)
    first_image = HandWrittingDataset.build_images(single_dataset.signals)

    #Display histogram of sequences
    plt.imshow(first_image[0])
    plt.title(f"Original image")
    plt.axis("off")

    #Get all images length
    single_dataset = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, separate_strokes=True, image_max_shape=(1000, 1000), single_signal=True, restrict_id=2)
    images = HandWrittingDataset.build_images(single_dataset.signals)

    fig, axes = plt.subplots(1, len(images))
    for i, image in enumerate(images):
        #Display histogram of sequences
        axes[i].imshow(image, cmap='gray')
        axes[i].axis("off")

    plt.show()

for logchannel in LogChannels:
    logger.add_log_channel(logchannel)

brush_dataset_orig = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, separate_strokes=False, image_max_shape=(1000, 1000))
brush_dataset_strokes = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, separate_strokes=True, image_max_shape=(1000, 1000))
# brush_dataset_limited_strokes = BrushDataset(brush_root=BRUSH_ROOT, save_to_file=False, separate_strokes=True, 
#                                              image_max_shape=(96, 96), restrict_id=1, single_signal=True)

# get_stats(brush_dataset_orig, "Signals", bin_size=50)
# get_stats(brush_dataset_strokes, "Strokes", bin_size=5)
# get_stats(brush_dataset_limited_strokes, "Restricted strokes", bin_size=5)

fig, ax = plt.subplots()
plot_hist_on(brush_dataset_orig, "Signals", bin_size=5, ax=ax)
plot_hist_on(brush_dataset_strokes, "Strokes", bin_size=5, ax=ax)
# # plot_hist_on(brush_dataset_limited_strokes, "Restricted strokes", bin_size=5, ax=ax)

plt.show()

# # before_after()

ds = HandWrittingDataset(BRUSH_96_96_TRAIN_M_MIXED)
ds2= HandWrittingDataset(BRUSH_96_96_TEST_M_MIXED)

ds3 = HandWrittingDataset(BRUSH_96_96_TRAIN_S_MIXED)
ds4 = HandWrittingDataset(BRUSH_96_96_TEST_S_MIXED)

print(f"Small dataset len {len(ds3)} - {len(ds4)}")
print(f"Medium dataset len {len(ds)} - {len(ds2)}")
