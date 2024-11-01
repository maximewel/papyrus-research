import torch
import torch.nn as nn 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from enum import Enum, auto

from source.model.blocks.constants.sequence_to_image import ImageHelper

class SkeletonLossMode(Enum):
    """
    Set the mode to the skeletton loss
    SUM_PIX: Compute sum of the pixels of the diff image
    DIST_PIX: Compute the distance of each pix to the skeletton
    """
    SUM_PIX = auto()
    DIST_PIX = auto()

class SkeletonLoss(nn.Module):
    display: bool
    dataset_image_shape: tuple
    normalized_sequences: bool
    loss_mode: SkeletonLossMode
    
    ## Coords are expected to be [0,1] and are normalized within the dataset target length (the maximum shape of an image)
    ## As such, clamping coordinates by the model between [0,5] prevent us from building nonsensical images
    MIN_COORD_CLIP = 0
    MAX_COORD_CLIP = 2

    # In SUMPIX, The distance value is divided by this factor in order to create a boosting zone 
    # As the distance factor is applied before the squaring, this makes a boosting effect where loss is diminished
    # In any case, this also lowers the max loss value that could be great over a full image.
    SUMPIX_DISTANCE_RATIO = 5

    def __init__(self, normalized_sequences:bool, dataset_image_shape: tuple, display: bool = False, mode: SkeletonLossMode = SkeletonLossMode.SUM_PIX):
        super().__init__()
        self.display = display
        self.dataset_image_shape = dataset_image_shape
        self.normalized_sequences = normalized_sequences
        self.loss_mode = mode

        if not self.normalized_sequences:
            self.MAX_COORD_CLIP *= max(dataset_image_shape)

    def forward(self, last_coordinates: torch.Tensor, predicted_coordinates: torch.Tensor, reference_images: list[np.ndarray]):
        """
            Return a loss based on the difference between the line drawn and the base skeletton

            Args
            -----
                - last_coordinates: torch.Tensor - The last coordinates of the signal
                - predicted_coordinates: torch.Tensor - The next prediction of coordinates
                - target: list[np.ndarray] - The reference images containing the relevant skelettons
        """
        #Ensure we have coords on CPU
        last_coordinates = last_coordinates.cpu()

        #As we will make images out of predicted coordinates, protect image shape by clamping 
        clamped_coordinates = torch.clamp(predicted_coordinates.cpu(), min=self.MIN_COORD_CLIP, max=self.MAX_COORD_CLIP)

        losses = []

        for i in range(len(last_coordinates)):
            #Create the image from the line between the last coordinate and the expected coordinate
            line_created = torch.vstack([last_coordinates[i], clamped_coordinates[i]])
            image_line = self.image_from_result(line_created)
            w, h = image_line.shape

            #Restrict skeletton to created image window. If image is bigger than skeletton, pad skeletton with 0
            skeletton_window = reference_images[i][:w, :h]
            sk_w, sk_h = skeletton_window.shape
            diff_w, diff_h = max(w - sk_w, 0), max(h - sk_h, 0)
            if diff_w != 0 or diff_h != 0:
                #Apply pad after
                skeletton_window = np.pad(skeletton_window, pad_width=((0, diff_w), (0, diff_h)), constant_values=0)

            if self.loss_mode is SkeletonLossMode.SUM_PIX:
                #Calculate loss by sumply counting the wrong pixels created by the line that are not on skeletton
                diff = cv2.subtract(image_line, skeletton_window)
                loss = np.sum(diff)

                if self.display:
                    self.display_images(image_line, skeletton_window, diff)
            else:
                #As distancetransform computes the distance from each point to 0, we can flip the skeleton in order to have
                #A distance map that computes the distance of each point to the skeleton
                inverted_skeleton = 1 - skeletton_window
                distance_to_skeleton = cv2.distanceTransform(inverted_skeleton.astype(np.uint8), cv2.DIST_L2, 3)

                #Now, we can simply mask the distance map with the line in order to have the distance of 
                # each created pixel to the skeletton
                line_to_skeleton_distances = distance_to_skeleton * image_line

                #Apply some computations to scale the distance and have a smoother loss
                #scaled_distance_to_skeleton = line_to_skeleton_distances / self.SUMPIX_DISTANCE_RATIO
                scaled_distance_to_skeleton = cv2.normalize(line_to_skeleton_distances, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                line_to_skeleton_distances_squared = np.power(scaled_distance_to_skeleton, 2)

                if self.display:
                    self.display_images_sumpix(image_line, skeletton_window, 
                                        inverted_skeleton, distance_to_skeleton, 
                                        line_to_skeleton_distances, line_to_skeleton_distances_squared)
                
                loss = np.sum(line_to_skeleton_distances_squared)

            losses.append(loss)

        return torch.mean(torch.Tensor(losses))
    
    def display_images_sumpix(self, line_drawn, skeleton, invert_skeletton, distances, distances_masked, result_distances):
        """Quick function used to display the images for computing the loss when sumpix"""
        # Set up the figure and subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 5))
        fig.suptitle("Skeletton loss construction")

        # Display each image with titles
        axs[0, 0].imshow(line_drawn, cmap='gray')
        axs[0, 0].set_title("Line Drawn")
        axs[0, 0].axis('off')  # Hide axes

        axs[0, 1].imshow(skeleton, cmap='gray')
        axs[0, 1].set_title("Skeleton")
        axs[0, 1].axis('off')  # Hide axes

        axs[1, 0].imshow(invert_skeletton, cmap='gray')
        axs[1, 0].set_title("Inverse skeleton")
        axs[1, 0].axis('off')  # Hide axes

        axs[1, 1].imshow(distances, cmap='gray')
        axs[1, 1].set_title("Distance map")
        axs[1, 1].axis('off')  # Hide axes

        axs[2, 0].imshow(distances_masked, cmap='gray')
        axs[2, 0].set_title("distance map masked")
        axs[2, 0].axis('off')  # Hide axes

        axs[2, 1].imshow(result_distances, cmap='gray')
        axs[2, 1].set_title("Result distances (Factor + squared)")
        axs[2, 1].axis('off')  # Hide axes

        plt.show()
    
    def display_images(self, line_drawn, skeleton, diff):
        """Quick function used to display the images for computing the loss"""
        # Set up the figure and subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Skeletton loss construction")

        # Display each image with titles
        axs[0].imshow(line_drawn, cmap='gray')
        axs[0].set_title("Line Drawn")
        axs[0].axis('off')  # Hide axes

        axs[1].imshow(skeleton, cmap='gray')
        axs[1].set_title("Skeleton")
        axs[1].axis('off')  # Hide axes

        axs[2].imshow(diff, cmap='gray')
        axs[2].set_title("Subtraction")
        axs[2].axis('off')  # Hide axes

        # Show the figure
        plt.show()

    def image_from_result(self, resultSignal: torch.Tensor):
        """
            Create an image from the result
            Security: If negative coordinates exist, clip it
        """
        mult_tensor = torch.tensor(self.dataset_image_shape, dtype=int) if self.normalized_sequences else 1

        resultSignalAsInt = (resultSignal * mult_tensor).int()
        #Pad to obtain original third dimension, 'penup'
        resultSignalAsInt = torch.nn.functional.pad(resultSignalAsInt, (0, 1))
        return ImageHelper.create_image(resultSignalAsInt.cpu().numpy())