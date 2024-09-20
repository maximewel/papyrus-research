import torch
from torch import Tensor
import numpy as np

from source.logging.log import logger, LogChannels

class Patchificator():
    """
    Helper - Helps with Patchification, either in-model or in data.
    """
    unfolder: torch.nn.Unfold
    fixed_image_dim: tuple

    IMG_MAX_VALUE = 255

    def __init__(self, patch_dimension: tuple, fixed_image_dim: tuple) -> None:
        """
        Build a patchificator
        Args
        -----
            patch_dimension: tuple - The patch dimensions, expected Tuple(width, height)
            fixed_image_dim: tuple - The maximum fixed-length output dimension of the images, expected Tuple(width, height)
        """
        self.unfolder = torch.nn.Unfold(patch_dimension, stride=patch_dimension)
        self.fixed_image_dim = fixed_image_dim
    
    def normalize_patchify_images(self, images: list[np.ndarray], normalize_value: bool) -> tuple[Tensor, Tensor]:
        """
        Normalize the images to obtain fixed-length images.
        Return the patchified image as well as the corresponding padding masks

        Carefull: This function does padding expecting the fixed_image_dim to be greater or equal to the dimension
        of the inner images. In case of lower dimensions, images will be ungracefully cropped.

        Args
        -----
            images: list[np.ndarray] - Input images. List of np.ndarray
                List: Because a list allows for non-homogeneous images
                ndarray: Expecting images as ndarray
            normalize_value: bool - Whether to normalize the pixel's values by the max, 255 (expecting grayscale)

        Returns
        -----
            Tuple[Tensor, Tensor]: Padded images, padding masks
                - Padded images: Turn the un-homogeneous list of images into an homogeneous tensor, dims = [b, n_patches, patch_dim]
                - Padding masks: An homogeneous tensor, dims = [b, n_patches], that maps the padding patches for each padded image. A 
                    padding patch is a patch where every pixel was padded and is therefore to discard during attention.
        """
        batch_size = len(images)
        W, H = self.fixed_image_dim

        # Initialize padded images and padding masks
        padded_images = torch.zeros((batch_size, 1, W, H), dtype=float)
        padding_masks = torch.ones((batch_size, 1, W, H)) # correspond to channel, assume 1 (would be easy to adapt to multiple channels)
        
        # Copy original images into the padded tensor and create the padding mask
        for i in range(len(images)):
            image = images[i]
            if len(image.shape) == 3:
                c, w, h = image.shape
            else:
                w, h = image.shape

            image_tensor = torch.from_numpy(image).float()
            if normalize_value:
                image_tensor /= self.IMG_MAX_VALUE

            padded_images[i, 0, :w, :h] = image_tensor
            padding_masks[i, 0, :w, :h] = 0

        logger.log(LogChannels.PADDING, f"Padded dim: {padded_images.shape}, masks dim: {padding_masks.shape}")
        logger.log(LogChannels.PADDING, f"Example of padded image: {padded_images[0]}")
        logger.log(LogChannels.PADDING, f"Example of padded Mask: {padding_masks[0]}")

        # Use unfolder to obtain patches. Float(): Unfolder does not work on int tensors.
        # Will give a result in the shape of [batch_size, n_patches, patches_dim]
        patchified_images: torch.Tensor = self.unfolder(padded_images).permute(0, 2, 1).float()

        #The pachified mask is pixel-wise and has the same dimensions as the padding images:  [batch_size, n_patches, patches_dim]
        #To obtain a padding mask, the last dim is reduced - if each pixel is padded, return true as this patch is a full padding.
        #If at least one pixel is false (not padding), then the patch is not masked (false).
        #Result in a tensor of shape [batch_size, n_patches]
        patchified_mask: torch.Tensor = self.unfolder(padding_masks).permute(0, 2, 1)
        patchified_mask = torch.all(patchified_mask, dim=2)

        logger.log(LogChannels.PADDING, f"Example of patchified padded image: {patchified_images[0]}")
        logger.log(LogChannels.PADDING, f"Example of patchified padded Mask: {patchified_mask[0]}")

        logger.log(LogChannels.PADDING, f"Patchified dim: {patchified_images.shape}, Patchified mask dim: {patchified_mask.shape}")

        return patchified_images, patchified_mask