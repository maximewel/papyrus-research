import pandas as pd
import os
import ast
from enum import Enum
import matplotlib.pyplot as plt

path = os.path.join(os.path.abspath(__file__), '..', 'raytune_results', "results_df.csv")

def create_plots(df):
    fig, axes = plt.subplots(2, 1, sharex=True)

    # First axis: Train Losses
    axes[0].set_title(f"Train losses")
    axes[0].set_ylabel("Loss")

    # Plot individual train losses for LSTM used
    first = True
    for i, row in df.iterrows():
        axes[0].plot(row["train_losses"], "r", alpha=0.5, label=f"Train Loss" if first else "")
        first = False

    # Plot average train losses
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Second axis: Test Losses
    axes[1].set_title(f"Test Losses")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")

    # Plot individual test losses for LSTM used
    first = True
    for i, row in df.iterrows():
        axes[1].plot(row["test_losses"], "b", alpha=0.5, label=f"Test Loss" if first else "")
        first = False

    # Plot average test losses

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

class Metrics(Enum):
    LR = "config/lr"
    WEIGHTS = "config/weights"
    HEADS = "config/heads"
    LAYERS = "config/layers"

def prevalence_of_feature_in(df):
    #Count features in each row
    count = {
        Metrics.LR.value: {},
        Metrics.WEIGHTS.value: {},
        Metrics.HEADS.value: {},
        Metrics.LAYERS.value: {},
    }

    for i, row in df.iterrows():
        for key in Metrics:
            key = key.value
            val = row[key]
            try:
                count[key][str(val)] += 1
            except KeyError:
                count[key][str(val)] = 1

    print(count)

if __name__ == "__main__":
    df = pd.read_csv(path)

    # Parse the train_losses and test_losses columns
    df["train_losses"] = df["train_losses"].apply(ast.literal_eval)
    df["test_losses"] = df["test_losses"].apply(ast.literal_eval)

    sorted_df = df.sort_values(by=['test_loss'])

    print(sorted_df)

    create_plots(df)

    prevalence_of_feature_in(sorted_df[:20])

    print(sorted_df[:10]['experiment_tag'])
    sorted_df.to_csv('./sorted_res')