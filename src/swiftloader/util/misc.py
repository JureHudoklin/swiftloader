import os
import sys
import numpy as np
import torch

from torch import Tensor
from typing import List, Any


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def get_bbox_from_mask(
    alpha_mask: Tensor,
    background_threshold: int = 230,
    white_is_foreground: bool = True,
) -> Tensor:
    """
    Given a binary mask, returns the bounding box of the object in the mask.

    Parameters
    ----------
    alpha_mask : Tensor | np.ndarray
        A mask of the object. shape: (H, W)
    background_threshold : int, optional
        The background_threshold value for the mask. Pixels with values greater than or equal to this value are considered foreground.
    white_is_foreground : bool, optional
        If True, pixels with values greater than or equal to the background_threshold are considered foreground. Otherwise, pixels with values less than or equal to the background_threshold are considered foreground.

    Returns
    -------
    Tensor
        A tensor of shape (4,) representing the bounding box of the object in the mask.
        The tensor contains the coordinates (x0, y0, x1, y1) of the top-left and bottom-right corners of the bounding box, respectively.
    """
    if white_is_foreground:
        mask = np.array(alpha_mask) >= background_threshold
    else:
        mask = np.array(alpha_mask) <= background_threshold

    mask_tensor = torch.tensor(mask)
    h, w = mask.shape
    # Get the first and last non-zero index of the mask
    h_non, w_non = torch.nonzero(mask_tensor, as_tuple=True)
    if len(h_non) == 0:
        return torch.tensor([0, 0, 0, 0])
    # sort the indices
    h_non, indices = torch.sort(h_non)
    w_non = w_non[indices]
    y0 = torch.clamp(h_non[0], min=0)
    y1 = torch.clamp(h_non[-1], max=h - 1)

    w_non, indices = torch.sort(w_non)
    x0 = torch.clamp(w_non[0], min=0)
    x1 = torch.clamp(w_non[-1], max=w - 1)

    bbox = torch.tensor([x0, y0, x1, y1])

    return bbox