import pandas as pd
import os
import ast
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

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

def plot_prevalence(prevalence, l):
    # Flatten all unique values and assign colors
    unique_values = {}
    for feature, values in prevalence.items():
        for val in values:
            if val not in unique_values:
                unique_values[val] = len(unique_values)  # Assign unique index

    # Define colors for each unique value
    colors = plt.cm.tab20.colors  # Up to 20 unique colors
    value_colors = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}

    # Data preparation
    features = list(prevalence.keys())
    x = np.arange(len(features))  # X positions of bars
    width = 0.8  # Width of the bars

    # Initialize bar stacks
    bottom = np.zeros(len(features))

    # Plot stacked bars
    fig, ax = plt.subplots(figsize=(10, 6))
    for val, idx in unique_values.items():
        # Extract counts for this value across features
        counts = [prevalence[f].get(val, 0) for f in features]
        ax.bar(x, counts, width, label=f"{val}", color=value_colors[val], bottom=bottom)
        bottom += counts  # Update bottom for next stack

    # Customization
    ax.set_ylabel("Count of Prevalence")
    ax.set_title(f"Feature Prevalence in Top {l} Candidates")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.legend(title="Values", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_prevalence_shared(prevalences, lengths):
    # Flatten all unique values and assign colors
    unique_values = {}
    for prevalence in prevalences:
        for feature, values in prevalence.items():
            for val in values:
                if val not in unique_values:
                    unique_values[val] = len(unique_values)  # Assign unique index

    # Define colors for each unique value
    colors = plt.cm.tab20.colors  # Up to 20 unique colors
    value_colors = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}

    # Create subplots
    num_plots = len(prevalences)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)

    if num_plots == 1:  # Handle case where there's only one subplot
        axes = [axes]

    for ax, prevalence, l in zip(axes, prevalences, lengths):
        # Data preparation
        features = list(prevalence.keys())
        x = np.arange(len(features))  # X positions of bars
        width = 0.8  # Width of the bars

        # Initialize bar stacks
        bottom = np.zeros(len(features))

        for val, idx in unique_values.items():
            # Extract counts for this value across features
            counts = [prevalence[f].get(val, 0) for f in features]
            ax.bar(x, counts, width, label=f"{val}", color=value_colors[val], bottom=bottom)
            bottom += counts  # Update bottom for next stack

        # Customization for each subplot
        ax.set_title(f"Top {l} Candidates")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.set_ylabel("Count of Prevalence" if ax == axes[0] else "")  # Only on first plot

    # Create a single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Values", loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.show()

def prevalence_of_feature_in(df):
    #Count features in each row
    prevalence = {
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
                prevalence[key][str(val)] += 1
            except KeyError:
                prevalence[key][str(val)] = 1

    print(prevalence)
    plot_prevalence(prevalence, len(df))
    return prevalence

def ranking_analysis(df):
    # Step 1: Rank the trials by test_loss
    df['rank'] = df['test_loss'].rank(method='min')

    # Step 2: Group by each configuration parameter
    grouped_by_lr = df.groupby('config/lr')['rank'].mean()
    grouped_by_weights = df.groupby('config/weights')['rank'].mean()
    grouped_by_heads = df.groupby('config/heads')['rank'].mean()
    grouped_by_layers = df.groupby('config/layers')['rank'].mean()

    # Display the results
    print("Average Rank by Learning Rate:\n", grouped_by_lr)
    print("\nAverage Rank by Weights:\n", grouped_by_weights)
    print("\nAverage Rank by Heads:\n", grouped_by_heads)
    print("\nAverage Rank by Layers:\n", grouped_by_layers)

if __name__ == "__main__":
    df = pd.read_csv(path)

    # Parse the train_losses and test_losses columns
    df["train_losses"] = df["train_losses"].apply(ast.literal_eval)
    df["test_losses"] = df["test_losses"].apply(ast.literal_eval)

    sorted_df = df.sort_values(by=['test_loss'])

    #create_plots(df)

    #ranking_analysis(df)
    prevalences = []
    lengths = []
    for i in [64, 30, 10]:
        prevalences.append(prevalence_of_feature_in(sorted_df[:i]))
        lengths.append(i)
    
    plot_prevalence_shared(prevalences, lengths)

    print(sorted_df[:10]['experiment_tag'])
    sorted_df.to_csv('./sorted_res')