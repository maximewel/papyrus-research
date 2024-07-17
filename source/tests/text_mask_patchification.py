
import torch
import numpy as np
from torch import Tensor

fixed_size_image_dimension = (6, 6)
patch_size = (2, 2)

def normalize_patchify_images(images: list[np.ndarray]):
    """Normalize the images to obtain fixed-length images.
    Return the patchified image as well as the corresponding padding masks"""
    batch_size = len(images)
    W,H = fixed_size_image_dimension

    # Initialize padded images and padding masks
    padded_images: Tensor = torch.zeros((batch_size, 1, W, H), dtype=int)
    padding_masks = torch.ones((batch_size, 1, W, H))
    
    # Copy original images into the padded tensor and create the padding mask
    for i in range(len(images)):
        image = images[i]
        c,w,h = image.shape
        print(image.shape)
        print(f"Image {i}: {image}")
        padded_images[i, 0, :w, :h] = image[:,:]
        padding_masks[i, 0, :w, :h] = 0

    for i in range(len(images)):
        print(f"Padded image {i}: {padded_images[i]}")
        print(f"Padded mask {i}: {padding_masks[i]}")

    print(f"Padded dim: {padded_images.shape}, masks dim: {padding_masks.shape}")

    unfolder = torch.nn.Unfold(patch_size, stride=patch_size)

    patchified_images: torch.Tensor = unfolder(padded_images.float()).int().permute(0, 2, 1)
    patchified_mask: torch.Tensor = unfolder(padding_masks).permute(0, 2, 1)

    for i in range(len(images)):
        print(f"patchified image {i}: {patchified_images[i]}")
        print(f"patchified mask {i}: {patchified_mask[i]}")

    a = (patchified_mask == 1).all(dim=0)
    print(f"IS padding: {a}")
    
    patchified_mask = torch.all(patchified_mask, dim=2)

    for i in range(len(images)):
        print(f"patchified image {i}: {patchified_images[i]}")
        print(f"patchified validation {i}: {patchified_mask[i]}")

    print(f"Patchified dim: {patchified_images.shape}, Patchified mask dim: {patchified_mask.shape}")


image_a = torch.IntTensor([[
    [1, 2, 3, 4], 
    [10, 20, 30, 40],
    [100, 200, 300, 400],
    [1000, 2000, 3000, 4000]
]])

image_b = torch.IntTensor([[
    [1, 2, 3, 4], 
    [10, 20, 30, 40],
    [100, 200, 300, 400]
]])

images = [image_a, image_b]
images = [image_b]


normalize_patchify_images(images)