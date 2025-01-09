import pickle
import os 

import matplotlib.pyplot as plt

# Load the figure from the pickle file
with open(os.path.join(os.path.abspath(__file__), '..', 'train_test_losses.pickle'), 'rb') as f:
    fig = pickle.load(f)

#Read data from the axes
axes = fig.get_axes()
for ax in axes:
    for line in ax.get_lines():
        x_data, y_data = line.get_data()
        print("X Data:", x_data)
        print("Y Data:", y_data)

#Corrections of legend
for ax in axes:
    legend = ax.get_legend()
    if legend is not None:  # Check if a legend exists
        legend.remove()
plt.legend(["Train loss", "Test loss"], loc="lower right")

#Misc corrections
plt.grid(True)
plt.title("Second training of a combination of structural hyper-parameters")
plt.ylim([4, 7])
ax.set_xticks(range(0, 4))
ax.set_xticklabels(range(1, 5))
plt.show()

#Second plot
# Load the figure from the pickle file
with open(os.path.join(os.path.abspath(__file__), '..', 'detailed_train_test_losses.pickle'), 'rb') as f:
    fig = pickle.load(f)

#Read data from the axes
axes = fig.get_axes()
for ax in axes:
    for line in ax.get_lines():
        x_data, y_data = line.get_data()
        print("X Data:", x_data)
        print("Y Data:", y_data)

#Corrections of legend
for ax in axes:
    legend = ax.get_legend()
    if legend is not None:  # Check if a legend exists
        legend.remove()

print(len(axes[0].get_lines()))
print(len(axes[1].get_lines()))

plt.legend([axes[0].get_lines()[0], axes[0].get_lines()[1]] + axes[1].get_lines(), ["Skeletton loss", "Epochs" ,"Coordinate loss"], loc="upper right")
#Misc corrections
plt.grid(True)
for ax in fig.get_axes():
    ax.set_title("")  # Remove the title for each axis
plt.title("Second training of a combination of structural hyper-parameters - Detailed losses")

plt.show()