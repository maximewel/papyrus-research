import pandas as pd
import ast
import matplotlib.pyplot as plt

import os

# Function to calculate average losses
def calculate_average_losses(group):
    # Transpose the list of lists to compute averages across rows for each epoch
    avg_train = []
    avg_test = []

    for epoch_losses in zip(*group["train_losses"]):
        # Filter out None values and compute the average
        valid_values = [x for x in epoch_losses if x is not None]
        avg_train.append(sum(valid_values) / len(valid_values) if valid_values else None)

    for epoch_losses in zip(*group["test_losses"]):
        # Filter out None values and compute the average
        valid_values = [x for x in epoch_losses if x is not None]
        avg_test.append(sum(valid_values) / len(valid_values) if valid_values else None)

    return avg_train, avg_test

def create_plots(first_group, second_group, first_label, second_label):

    # Calculate averages
    avg_train_lstm, avg_test_lstm = calculate_average_losses(first_group)
    avg_train_no_lstm, avg_test_no_lstm = calculate_average_losses(second_group)
    # Create a figure with two axes
    fig, axes = plt.subplots(2, 1, sharex=True)

    # First axis: Train Losses
    axes[0].set_title(f"Average Train Losses ({first_label} vs {second_label})")
    axes[0].set_ylabel("Training Loss")

    # Plot individual train losses for LSTM used
    first = True
    for i, row in first_group.iterrows():
        axes[0].plot(row["train_losses"], "b--", alpha=0.5, label=f"Train Loss ({first_label})" if first else "")
        first = False

    # Plot individual train losses for LSTM not used
    first = True
    for i, row in second_group.iterrows():
        axes[0].plot(row["train_losses"], "y--", alpha=0.5, label=f"Train Loss ({second_label})" if first else "")
        first = False

    # Plot average train losses
    axes[0].plot(avg_train_lstm, "b", linewidth=2, label=f"Avg Train Loss ({first_label})")
    axes[0].plot(avg_train_no_lstm, "y", linewidth=2, label=f"Avg Train Loss ({second_label})")

    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Second axis: Test Losses
    axes[1].set_title(f"Average Test Losses ({first_label} vs {second_label})")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Test Loss")

    # Plot individual test losses for LSTM used
    first = True
    for i, row in first_group.iterrows():
        axes[1].plot(row["test_losses"], "b--", alpha=0.5, label=f"Test Loss ({first_label})" if first else "")
        first = False

    # Plot individual test losses for LSTM not used
    first = True
    for i, row in second_group.iterrows():
        axes[1].plot(row["test_losses"], "y--", alpha=0.5, label=f"Test Loss ({second_label})" if first else "")
        first = False

    # Plot average test losses
    axes[1].plot(avg_test_lstm, "b", linewidth=2, label=f"Avg Test Loss ({first_label})")
    axes[1].plot(avg_test_no_lstm, "y", linewidth=2, label=f"Avg Test Loss ({second_label})")

    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    # Set x-axis ticks for both subplots
    for ax in axes:
        ax.set_xticks(range(0, 4))
        ax.set_xticklabels(range(1, 5))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()

def prepare_df(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    #Drop single outlier that is a pain for averages and graphes
    df = df[df["train_loss"] <= 100]

    # Parse the train_losses and test_losses columns
    df["train_losses"] = df["train_losses"].apply(ast.literal_eval)
    df["test_losses"] = df["test_losses"].apply(ast.literal_eval)

    # Function to pad lists to a given length
    def pad_to_length(lst, length, pad_value=None):
        return lst + [pad_value] * (length - len(lst)) if len(lst) < length else lst

    # Pad train_losses and test_losses to ensure they have 4 elements
    max_length = 4
    df["train_losses"] = df["train_losses"].apply(lambda x: pad_to_length(x, max_length))
    df["test_losses"] = df["test_losses"].apply(lambda x: pad_to_length(x, max_length))

    return df

if __name__ == "__main__":
    path = os.path.join(os.path.abspath(__file__), '..', "results_df.csv")

    df = prepare_df(path)

    # Filter rows based on LSTM usage
    lstm_used = df[df["config/use_lstm"] == True]
    lstm_not_used = df[df["config/use_lstm"] == False]
    create_plots(lstm_used, lstm_not_used, "LSTM", "No LSTM")

    pred_token_used = df[df["config/pred_token"] == True]
    pred_token_not_used = df[df["config/pred_token"] == False]
    create_plots(pred_token_used, pred_token_not_used, "Pred Token", "No Pred Token")

    pos_learnable = df[df["config/positional_learnable"] == True]
    pos_not_learnable = df[df["config/positional_learnable"] == False]
    create_plots(pos_learnable, pos_not_learnable, "Position Learnable", "Position Not learnable")