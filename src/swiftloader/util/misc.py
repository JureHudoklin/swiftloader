import os
import sys
from pathlib import Path
import json
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage
import numpy as np
import torch

from torch import Tensor
from typing import List, Any, Dict, Tuple, Optional


class MaskRLE:
    """
    A class to handle Run-Length Encoding (RLE) for masks.
    
    Converts numpy masks of shape (H, W, n_instances) into RLE format for efficient storage.
    """
    
    def __init__(self, masks: Optional[np.ndarray] = None, rle_data: Optional[Dict] = None):
        """
        Initialize MaskRLE either from numpy masks or from RLE data.
        
        Parameters
        ----------
        masks : np.ndarray, optional
            Numpy array of shape (H, W, n_instances) containing binary masks
        rle_data : Dict, optional
            Dictionary containing RLE encoded data
        """
        if masks is not None and rle_data is not None:
            raise ValueError("Cannot specify both masks and rle_data")
        
        if masks is not None:
            self.height, self.width, self.n_instances = masks.shape
            self.rle_data = self._encode_masks(masks)
        elif rle_data is not None:
            self.rle_data = rle_data
            self.height = rle_data['height']
            self.width = rle_data['width']
            self.n_instances = rle_data['n_instances']
        else:
            raise ValueError("Must specify either masks or rle_data")
    
    def _encode_masks(self, masks: np.ndarray) -> Dict:
        """
        Encode masks to RLE format.
        
        Parameters
        ----------
        masks : np.ndarray
            Binary masks of shape (H, W, n_instances)
            
        Returns
        -------
        Dict
            Dictionary containing RLE encoded data
        """
        rle_list = []
        
        for i in range(masks.shape[2]):
            mask = masks[:, :, i].astype(bool)
            # Flatten mask row-wise (C-order)
            flat_mask = mask.flatten()
            
            # Find run lengths
            if len(flat_mask) == 0:
                rle_list.append({'counts': [], 'size': [int(self.height), int(self.width)]})
                continue
                
            # Get positions where the value changes
            diff = np.diff(np.concatenate(([False], flat_mask, [False])).astype(int))
            start_positions = np.where(diff == 1)[0]
            end_positions = np.where(diff == -1)[0]
            
            # Calculate run lengths
            counts = []
            current_pos = 0
            
            for start, end in zip(start_positions, end_positions):
                # Add length of zeros before this run
                if start > current_pos:
                    counts.append(int(start - current_pos))
                # Add length of ones for this run
                counts.append(int(end - start))
                current_pos = end
                
            # Add remaining zeros if any
            if current_pos < len(flat_mask):
                counts.append(int(len(flat_mask) - current_pos))
                
            # If mask starts with True, we need to add a 0 at the beginning
            if len(start_positions) > 0 and start_positions[0] == 0:
                counts = [0] + counts

            rle_list.append({'counts': counts, 'size': [int(self.height), int(self.width)]})

        return {
            'height': int(self.height),
            'width': int(self.width),
            'n_instances': int(self.n_instances),
            'rle_masks': rle_list
        }
    
    def decode_masks(self) -> np.ndarray:
        """
        Decode RLE data back to numpy masks.
        
        Returns
        -------
        np.ndarray
            Binary masks of shape (H, W, n_instances)
        """
        masks = np.zeros((self.height, self.width, self.n_instances), dtype=bool)
        
        for i, rle_mask in enumerate(self.rle_data['rle_masks']):
            counts = rle_mask['counts']
            if len(counts) == 0:
                continue
                
            # Decode RLE
            flat_mask = np.zeros(self.height * self.width, dtype=bool)
            current_pos = 0
            is_foreground = False
            
            for count in counts:
                if is_foreground:
                    flat_mask[current_pos:current_pos + count] = True
                current_pos += count
                is_foreground = not is_foreground
                
            # Reshape back to 2D
            masks[:, :, i] = flat_mask.reshape(self.height, self.width)
        
        return masks
    
    def decode_mask(self, instance_index: int) -> np.ndarray:
        """
        Decode a single instance mask from RLE data.
        
        Parameters
        ----------
        instance_index : int
            Index of the instance to decode
            
        Returns
        -------
        np.ndarray
            Binary mask of shape (H, W) for the specified instance
        """
        if instance_index < 0 or instance_index >= self.n_instances:
            raise IndexError("Instance index out of bounds")
        
        rle_mask = self.rle_data['rle_masks'][instance_index]
        counts = rle_mask['counts']
        
        if len(counts) == 0:
            return np.zeros((self.height, self.width), dtype=bool)
        
        flat_mask = np.zeros(self.height * self.width, dtype=bool)
        current_pos = 0
        is_foreground = False
        
        for count in counts:
            if is_foreground:
                flat_mask[current_pos:current_pos + count] = True
            current_pos += count
            is_foreground = not is_foreground
            
        return flat_mask.reshape(self.height, self.width)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return self.rle_data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MaskRLE':
        """Create MaskRLE instance from dictionary."""
        return cls(rle_data=data)


def save_resolver(
    data: Any,
    path: Path,
    entry_name: str,
) -> None:
    if isinstance(data, PILImage):
        data.save(path / f"{entry_name}.jpg")
    elif isinstance(data, MaskRLE):
        with open(path / f"{entry_name}.rle.json", "w") as f:
            json.dump(data.to_dict(), f)
    elif isinstance(data, np.ndarray):
        np.savez_compressed(path / f"{entry_name}.npz", data)
    elif isinstance(data, torch.Tensor):
        torch.save(data, path / f"{entry_name}.pt")
    elif isinstance(data, dict | list):
        with open(path / f"{entry_name}.json", "w") as f:
            json.dump(data, f)
    else:
        print(data)
        raise ValueError(f"Unsupported data type: {type(data)}")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

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
