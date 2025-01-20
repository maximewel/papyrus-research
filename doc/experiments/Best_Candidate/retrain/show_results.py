
import sys
import os
import pandas as pd 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.datasets_library import *
from dtw import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

base_filepath = os.path.join(os.path.abspath(__file__), '..', 'inferences')

def create_image(signal):
    signal_as_int = signal.int()
    #Pad to obtain original third dimension, 'penup'
    signal_as_int = torch.nn.functional.pad(signal_as_int, (0, 1))
    result_image = ImageHelper.create_image(signal_as_int.numpy(), (96,96))
    return result_image

def process_signals_from_predictions(signals_to_compare: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    # Compare signals, run loss on everyone of them
    diffes_orig = []
    diffes_pred = []
    diffes_union = []
    dtw_values = []
    dist = []

    for original_signal, predicted_signal in signals_to_compare:
        #Compute mask between both
        original_image = create_image(original_signal)
        predicted_image = create_image(predicted_signal)
        
        #Compute skeletton diff between original and predicted
        diff_orig_pred = cv2.subtract(original_image, predicted_image)
        diff_pred_orig = cv2.subtract(predicted_image, original_image)

        diff_union_img = cv2.add(diff_orig_pred, diff_pred_orig)
        
        diff_orig = 1 - numpy.round(diff_orig_pred.sum() / original_image.sum(), 2)
        diff_pred = 1 - numpy.round(diff_pred_orig.sum() / predicted_image.sum(), 2)
        diff_union = 1 - numpy.round(diff_union_img.sum() / cv2.add(original_image,predicted_image).sum(), 2)

        diffes_orig.append(diff_orig)
        diffes_pred.append(diff_pred)
        diffes_union.append(diff_union)
        #print(f"Differences in skeletton: {diff_orig}, {diff_pred}, {diff_union}")

        #DTW is always possible
        dtw_diff = numpy.round(dtw(original_signal, predicted_signal).distance)
        #print(f"DTW distance: {dtw_diff}")
        dtw_values.append(dtw_diff)

        #If signals don't have the same size, align them so we can do a point-to-point euclidian distance over the signals
        aligned_min_len = min(predicted_signal.size(0), original_signal.size(0))
        clipped_predicted_signal = predicted_signal[:aligned_min_len, :]
        clipped_original_signal = original_signal[:aligned_min_len, :]

        sum_squared_diff = ((clipped_original_signal - clipped_predicted_signal) ** 2).sum(dim=1)
        euclidean_distances = torch.sqrt(sum_squared_diff)
        diff_dist = numpy.round(euclidean_distances.cpu().sum().item())
            
        #print(f"Diff in point-to-point euclidian distance: {diff_dist}")
        dist.append(diff_dist)
        
    return diffes_orig, diffes_pred, diffes_union, dtw_values, dist

seq_ratios = 0.25, 0.5, 0.75, 1
def plot_signal_at(title, signal_bundle):
    original_signal, predicted_signal = signal_bundle
    original_sig_len = len(original_signal)

    # Create a figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Loop through sequence ratios and plot the corresponding images
    for i, seq in enumerate(seq_ratios):
        original_ind = int(round(seq * original_sig_len))
        original_image_at_ind = create_image(original_signal[:original_ind])
        predicted_image_at_ind = create_image(predicted_signal[:original_ind])

        # Display the original signal image
        axes[i].imshow(original_image_at_ind, cmap='gray')
        axes[i].set_title(f"Original {int(seq * 100)}%")
        axes[i].axis('off')  # Hide axes for cleaner presentation

        # Display the predicted signal image
        axes[i + 5].imshow(predicted_image_at_ind, cmap='gray')
        axes[i + 5].set_title(f"Predicted {int(seq * 100)}%")
        axes[i + 5].axis('off')

    # Plot the full predicted signal image at specific locations (indexes 4, 2)
    full_pred = create_image(predicted_signal)
    axes[9].imshow(full_pred, cmap='gray')
    axes[9].set_title("Full Prediction")
    axes[9].axis('off')

    axes[4].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title

    print(f"Results: {process_signals_from_predictions([signal_bundle])}")

def create_figures_from_data(models_to_test):
    all_signals = []
    for model_to_test in models_to_test:
        stopped_str = 'natural'
        model_id = f"{model_to_test}_{stopped_str}"
        model_folderpath = os.path.join(base_filepath, model_id)

        #Load signals from disk
        signals_to_compare: list[tuple] = torch.load(os.path.join(model_folderpath, 'expected_predicted_signals.pt'))
        all_signals.append((model_to_test, signals_to_compare))

    START_IND = 0
    for ind in range(START_IND, 2000):
        for model_id, signals_to_compare in all_signals:
            title = f"Signal prediction with incremental steps every 25%  progressions; model {model_id}"
            plot_signal_at(title, signals_to_compare[ind])
        plt.show()
        plt.close()

    # small_signals = all_signals[0][1][:1000]
    # large_signals = all_signals[0][1][1000:]

    # SMALL = False

    # small_signals_res = process_signals_from_predictions(small_signals)
    # large_signals_res = process_signals_from_predictions(large_signals)

    # sorted_index = np.argsort(small_signals_res if SMALL else large_signals_res)
    
    # print(f"Best // worst: {sorted_index}")

    # for i in range(2, 5):
    #     print(f"Best indexes {i}")
    #     for ind in sorted_index[i][:3]:
    #         print(f"Ind {ind}")
    #         if not SMALL:
    #             ind += 1000
    #         for model_id, signals_to_compare in all_signals:
    #             title = f"Signal prediction with steps at 1/4 progressions; model {model_id}"
    #             plot_signal_at(title, signals_to_compare[ind])
    #         plt.show()
    #         plt.close()

    #     print(f"Worst indexes")
    #     for ind in sorted_index[i][-3:][::-1]:
    #         print(f"Ind {ind}")
    #         if not SMALL:
    #             ind += 1000
    #         for model_id, signals_to_compare in all_signals:
    #             title = f"Signal prediction with steps at 1/4 progressions; model {model_id}"
    #             plot_signal_at(title, signals_to_compare[ind])
    #         plt.show()
    #         plt.close()

if __name__ == "__main__":
    model_to_test = ["best_30_epochs"]
    create_figures_from_data(model_to_test)