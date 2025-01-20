import sys
import os
import pandas as pd 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.files import *
from concurrent.futures import ThreadPoolExecutor
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.hw_model import HwTransformer
from source.model.blocks.hw_lstm import HwLstm
from source.model.blocks.constants.files import *
from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.device_helper import device
from torch.nn.utils.rnn import pack_sequence
from source.model.blocks.constants.tokens import Tokens
from source.model.blocks.constants.datasets_library import *
from dtw import *
import matplotlib.pyplot as plt
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt

import cv2

STOP_CONDITION_IDENTICAL_OUTPUTS = 20
IMAGE_MAX_SHAPE = (96, 96)

SHOW_PLOT = False
SHOW_FIGS = False

base_filepath = os.path.join(os.path.abspath(__file__), '..', 'inferences')

tolerance = 0.0001
def has_identical_last_values(tensor, n: int) -> bool:
    """Return whether the last N values of the tensor are exact"""
    if tensor.shape[0] < n:
        return False
    last_rows = tensor[-n:, :]
    are_identical = torch.all(torch.abs(last_rows - last_rows[0, :]) < tolerance, dim=1).all()    
    return are_identical.item()

def image_from_result(resultSignal, target_size):
    """
        Create an image from the result
        Security: If negative coordinates exist, adjust image
    """
    for dim in [0, 1]:
        min_dim = torch.min(resultSignal[:, dim])
        if min_dim < 0 and min_dim != Tokens.COORDINATE_SEQUENCE_EOS.value:
            resultSignal[:-1, dim] += -min_dim

    resultSignalAsInt = resultSignal.int()
    #Pad to obtain original third dimension, 'penup'
    resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
    result_image = ImageHelper.create_image(resultSignalAsInt.cpu().numpy(), target_size)
    return result_image

def create_prediction_signals(model_name, stop_at_siglen: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Retrieve model
    model_path = os.path.join(os.path.abspath(__file__), '..', model_name, MODEL_FILENAME)
    print(f"Loading model from: {model_path}")
    
    model: HwTransformer = torch.load(model_path)
    model.eval()

    # Run whole inference to obtain signal
    dataset = HandWrittingDataset(BRUSH_96_96_VALID_S)

    signals_to_compare = []

    data_ind = 1
    for data in dataset:
        if data_ind % 50 == 0:
            print(f"Thread {threading.get_ident()} at signal {data_ind}/{len(dataset)+1}")

        _, patched_image, padding, current_signal, _ = data
        patched_image, padding, current_signal = patched_image.to(device), padding.to(device), current_signal.to(device)
        #Transform the signal to pytorch tensor to have matching types and be able to give it to model
        working_signal = current_signal[:1]

        i = 1
        stop_signal = False
        while not stop_signal:
            #Generate next point
            res = model.forward(patched_image.unsqueeze(0), padding.unsqueeze(0), pack_sequence(working_signal.unsqueeze(0)))
            res = res.detach().round()

            #Add it to working signal
            working_signal = torch.vstack([working_signal, res])

            # Check if we should stop
            if stop_at_siglen:
                stop_signal = (len(working_signal) == len(current_signal))
            else:
                stop_signal = torch.equal(res.squeeze(0), Tokens.EOS_TENSOR.value) or has_identical_last_values(working_signal, STOP_CONDITION_IDENTICAL_OUTPUTS) or (len(working_signal) >= 2 * len(current_signal))

            i += 1

        signals_to_compare.append((current_signal.cpu(), working_signal.cpu()))
        data_ind += 1

    return signals_to_compare

def process_signals_from_predictions(signals_to_compare: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    # Compare signals, run loss on everyone of them
    diffes_orig = []
    diffes_pred = []
    diffes_union = []
    dtw_values = []
    dist = []

    for original_signal, predicted_signal in signals_to_compare:
        #Compute mask between both
        original_image = image_from_result(original_signal, IMAGE_MAX_SHAPE)
        predicted_image = image_from_result(predicted_signal, IMAGE_MAX_SHAPE)
        
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

        if SHOW_PLOT:
            # Create a figure with subplots
            fig, axes = plt.subplots(1, 5)
            fig.suptitle("Image Comparisons")

            # Plot each image in a subplot
            axes[0].imshow(original_image, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(predicted_image, cmap="gray")
            axes[1].set_title("Predicted Image")
            axes[1].axis("off")

            axes[2].imshow(diff_orig_pred, cmap="gray")
            axes[2].set_title("Difference from Original")
            axes[2].axis("off")

            axes[3].imshow(diff_pred_orig, cmap="gray")
            axes[3].set_title("Difference from Prediction")
            axes[3].axis("off")

            axes[4].imshow(diff_union_img, cmap="gray")
            axes[4].set_title("Union of Differences")
            axes[4].axis("off")

            # Display the plot
            plt.tight_layout()
            plt.show()
        
    return diffes_orig, diffes_pred, diffes_union, dtw_values, dist

# colors = ['#1f77b4', '#d62728', '#9467bd', '#1f77b4', '#d62728', '#9467bd']
# labels = ["mixed_4", "best_s_30", "best_m_8", "mixed_4", "best_s_30", "best_m_8"]

colors = ['#1f77b4', '#d62728', '#1f77b4', '#d62728']
labels = ["best_30", "best_30_lstm", "best_30", "best_30_lstm"]
def plot_diffs_list(diffa, diffb, diffc):
    fig, axes = plt.subplots(3, 1, sharex=False)

    boxes = []

    box = axes[0].boxplot(diffa, tick_labels=labels, patch_artist=True)
    boxes.append(box)
    axes[0].set_title("Boxplot of difference from original image")
    axes[0].set_ylabel("Values (0-1)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    box = axes[1].boxplot(diffb, tick_labels=labels, patch_artist=True)
    boxes.append(box)
    axes[1].set_title("Boxplot of difference from predicted image")
    axes[1].set_ylabel("Values (0-1)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    box = axes[2].boxplot(diffc, tick_labels=labels, patch_artist=True)
    boxes.append(box)
    axes[2].set_title("Boxplot of difference from union image")
    axes[2].set_ylabel("Values (0-1)")
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Assign colors to the boxes
    for box in boxes:
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

    # Add vertical line for the subplot
    for ax in axes:
        ax.set_ylim(0, 1.1)
        middle_x = 2.5
        ax.axvline(middle_x, color='black', linestyle='--')
        ax.text(0.43, 0.92, "Early Stopping", transform=ax.transAxes, fontsize=10, color='black')
        ax.text(0.51, 0.92, "Model Stopping", transform=ax.transAxes, fontsize=10, color='black')

    # Adjust layout and show the plot
    fig.set_size_inches(16, 9)
    plt.tight_layout()
    if SHOW_FIGS:
        plt.show()

    return fig

def plot_distances_list(dtw, distances):
    fig, axes = plt.subplots(2, 1, sharex=False)

    boxes = []

    box = axes[0].boxplot(dtw, tick_labels=labels, patch_artist=True)
    boxes.append(box)
    axes[0].set_title("Boxplot of DTW distances")
    axes[0].set_ylabel("Values")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    box = axes[1].boxplot(distances, tick_labels=labels, patch_artist=True)
    boxes.append(box)
    axes[1].set_title("Boxplot of point-to-point euclidian distances")
    axes[1].set_ylabel("Values")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Assign colors to the boxes
    for box in boxes:
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

    # Add vertical line for the subplot
    for ax in axes:
        middle_x = 2.5
        ax.axvline(middle_x, color='black', linestyle='--')
        ax.text(0.43, 0.95, "Early Stopping", transform=ax.transAxes, fontsize=10, color='black')
        ax.text(0.51, 0.95, "Model Stopping", transform=ax.transAxes, fontsize=10, color='black')

    # Adjust layout and show the plot
    fig.set_size_inches(16, 9)
    plt.tight_layout()
    if SHOW_FIGS:
        plt.show()

    return fig

def plot_distances_single_model(model_name, stop_at_siglen, diff_orig, diff_pred, diff_union, dtw, distances):
    fig, axes = plt.subplots(2, 1, sharex=False)

    plt.title(f"Metrics auto-regression inferences with augmentation {model_name} and {'early stop' if stop_at_siglen else 'natural stop'}")

    axes[0].boxplot([diff_orig, diff_pred, diff_union], tick_labels=["differences origin", "differences predictions", "differences union"], patch_artist=True)
    axes[0].set_title("Boxplot of pixel correlations between images")
    axes[0].set_ylabel("Values")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].boxplot([dtw, distances], tick_labels=["dtw", "euclidian distances"], patch_artist=True)
    axes[1].set_title("Boxplot of distances")
    axes[1].set_ylabel("Values")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and show the plot
    fig.set_size_inches(16, 9)
    plt.tight_layout()
    if SHOW_FIGS:
        plt.show()

    return fig

colors_len_distrib = ["#4daf4a", '#1f77b4', '#d62728', '#9467bd']
colors_len_distrib = ["#4daf4a", '#1f77b4']
def plot_length_distribution(legends, distributions, length):
    fig = plt.figure()
    plt.title(f"Length distributions of the expected and predicted signals over the validation datasets and the model's predictions")

    smalls_signals_distributions = distributions[0::3][3:]
    large_signals_distributions = distributions[1::3][3:]
    all_signals_distributions = distributions[2::3][3:]

    leg = [""]

    smalls_signals_legends = legends[0::length]
    smalls_signals_legends = ["Small Signals Ground Truth"] + leg
    large_signals_legends = legends[1::length]
    large_signals_legends = ["Large Signals Ground Truth"] + leg
    all_signals_legends = legends[2::length]
    all_signals_legends = ["All Signals Ground Truth"] + leg

    ordered_distributions = smalls_signals_distributions + large_signals_distributions + all_signals_distributions
    ordered_legends = smalls_signals_legends + large_signals_legends + all_signals_legends

    bp = plt.boxplot(ordered_distributions, tick_labels=ordered_legends, patch_artist=True)
    plt.ylabel("Values")

    names_color = [('Ground Truth', "#4daf4a"), ('best_s_4', '#1f77b4'),  ('best_s_30', '#d62728'),  ('best_m_8', '#9467bd')]
    names_color = [('best_30', "#4daf4a"), ('best_30_lstm', '#1f77b4')]
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in names_color]
    plt.legend(handles=legend_patches, title="Augmentations", loc="upper left")

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    for patch, color in zip(bp['boxes'], colors_len_distrib * 3):
        patch.set_facecolor(color)

    # Adjust layout and show the plot
    fig.set_size_inches(16, 9)
    plt.tight_layout()
    if SHOW_FIGS:
        plt.show()

    return fig

def process_combination(model_to_test, stop_at_siglen):
    # Generate prediction signals
    signals_to_compare = create_prediction_signals(model_to_test, stop_at_siglen)

    # Create the directory
    model_folderpath = os.path.join(
        base_filepath, f"{model_to_test}_{'early' if stop_at_siglen else 'natural'}"
    )
    os.makedirs(model_folderpath, exist_ok=True)

    # Save signals to disk
    torch.save(signals_to_compare, os.path.join(model_folderpath, 'expected_predicted_signals.pt'))

def generate_signals_multithread():
    #models_to_test = ["best_30_epochs", "best_30_epochs_lstm"]
    models_to_test = ["best_m_8_epochs"]

    # Prepare all combinations
    combinations = [
        (model, stop_at_siglen) for stop_at_siglen in [True, False] for model in models_to_test
    ]

    # Multi-threading
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_combination, model, stop_at_siglen)
            for model, stop_at_siglen in combinations
        ]

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in thread: {e}")

SMALL_SIGNALS = "small_signals"
LARGE_SIGNALS = "large_signals"
ALL_SIGNALS = "all_signals"
ALL_SIGNALS_KEYS = [SMALL_SIGNALS, LARGE_SIGNALS, ALL_SIGNALS]

def create_figures_from_data(models_to_test):

    data = { k:[] for k in ALL_SIGNALS_KEYS }

    diffs_orig = { k:[] for k in ALL_SIGNALS_KEYS }
    diffs_pred = { k:[] for k in ALL_SIGNALS_KEYS }
    diffs_union = { k:[] for k in ALL_SIGNALS_KEYS }
    dtw_lists = { k:[] for k in ALL_SIGNALS_KEYS }
    distances_lists = { k:[] for k in ALL_SIGNALS_KEYS }

    first_pass = True
    length_legends_distribution = [[], []]

    for stop_at_siglen in [True, False]:
        for model_to_test in models_to_test:
            stopped_str = 'early' if stop_at_siglen else 'natural'
            model_folderpath = os.path.join(base_filepath, f"{model_to_test}_{stopped_str}")

            #Load signals from disk
            signals_to_compare = torch.load(os.path.join(model_folderpath, 'expected_predicted_signals.pt'))

            #Separate small and big signals in order to compare eventual differences.
            names_signal_bundles = [
                (SMALL_SIGNALS, signals_to_compare[:1000]),
                (LARGE_SIGNALS, signals_to_compare[1000:]),
                (ALL_SIGNALS, signals_to_compare)
            ]

            if first_pass:
                first_pass = False
                for key, exp_pred_signals in names_signal_bundles:
                    length_legends_distribution[0].append(f"{key} Ground Truth")
                    length_legends_distribution[1].append([len(exp_signal[0]) for exp_signal in exp_pred_signals])
                        
            for name, current_signals_to_compare in names_signal_bundles:                
                length_legends_distribution[0].append(f"pred_{name}_{model_to_test}_{stopped_str}")
                length_legends_distribution[1].append([len(signals[1]) for signals in current_signals_to_compare])

                #Compute metrics, save values to disk
                diff_orig, diff_pred, diff_union, dtw_values, distances = process_signals_from_predictions(current_signals_to_compare)
                torch.save([diff_orig, diff_pred, diff_union, dtw_values, distances], os.path.join(model_folderpath, f'metrics_over_predictions_{name}.pt'))

                #Store differences localy
                diffs_orig[name].append(diff_orig)
                diffs_pred[name].append(diff_pred)
                diffs_union[name].append(diff_union)
                dtw_lists[name].append(dtw_values)
                distances_lists[name].append(distances)

                #Plot this model figure
                fig = plot_distances_single_model(model_to_test, stop_at_siglen, diff_orig, diff_pred, diff_union, dtw_values, distances)
                fig.savefig(os.path.join(model_folderpath, f'summary_metrics_for_model_{name}.png'), bbox_inches='tight', dpi=300)
                plt.close()

                #Store averages in data for final dataframe, save locally
                data[name].append([model_to_test, stop_at_siglen, np.mean(diff_orig), np.mean(diff_pred), np.mean(diff_union), np.mean(dtw_values), np.mean(distances)])

    plot_distribution = plot_length_distribution(length_legends_distribution[0], length_legends_distribution[1], len(model_to_test))
    plot_distribution.savefig(os.path.join(base_filepath, f'length_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()

    for key in ALL_SIGNALS_KEYS:
        fig = plot_diffs_list(diffs_orig[key], diffs_pred[key], diffs_union[key])
        fig.savefig(os.path.join(base_filepath, f'diff_list_all_models_{key}.png'), dpi=500)
        
        fig = plot_distances_list(dtw_lists[key], distances_lists[key])
        fig.savefig(os.path.join(base_filepath, f'distances_list_all_models_{key}.png'), dpi=500)
        plt.close()

        df = pd.DataFrame(data[key], columns=['Augmentation', 'Match_siglen', 'diffA', 'diffB', 'diffC', 'DTW_value', 'dist_value'])
        df.to_csv(os.path.join(base_filepath, f'results_{key}.csv'))
    

def extract_plot_data(fig):
    """Extract the data from a matplotlib figure object."""
    axes = fig.get_axes()  # Get the axes from the figure
    data = {"train": None, "test": None}
    
    if len(axes) > 0:
        lines = axes[0].get_lines()  # Get the lines plotted on the first axis
        if len(lines) >= 2:  # Assuming two lines: train and test
            data["train"] = lines[0].get_ydata()
            data["test"] = lines[1].get_ydata()
    return data

import pickle
def merge_train_test_losses(folder_list):
    train_losses = {}
    test_losses = {}

    # Load train and test losses from each folder
    for folder in folder_list:
        file_path = os.path.join(os.path.abspath(__file__), '..', folder, "train_test_losses.pickle")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                fig = pickle.load(f)  # Load the matplotlib figure
                data = extract_plot_data(fig)  # Extract train and test data
                train_losses[folder] = data["train"]
                test_losses[folder] = data["test"]        
        else:
            print(f"File not found: {file_path}")
    
    print(train_losses)
    print(test_losses)

    # Plot all losses on a single figure
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors  # Use tab10 colormap for distinct colors

    for idx, folder in enumerate(folder_list):
        color = colors[idx % len(colors)]  # Cycle through colors
        if folder in train_losses and train_losses[folder] is not None:
            plt.plot(train_losses[folder], label=f"Train - {os.path.basename(folder)}", color=color, linestyle='-')
        if folder in test_losses and test_losses[folder] is not None:
            plt.plot(test_losses[folder], label=f"Test - {os.path.basename(folder)}", color=color, linestyle='--')

    plt.title("Train and Test Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #generate_signals_multithread()

    # models_to_test = ["mixed", "best_30_epochs", "best_m_8_epochs"]
    # merge_train_test_losses(models_to_test)
    # create_figures_from_data(models_to_test)

    models_to_test = ["best_30_epochs", "best_30_epochs_lstm"]
    merge_train_test_losses(models_to_test)
    create_figures_from_data(models_to_test)