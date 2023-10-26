import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid

from PIL import Image
from PIL.Image import Image as PILImage

import torch
from torch import Tensor
from torchvision.transforms.v2.functional import to_image
from torchvision.ops import box_convert, clip_boxes_to_image
from torchvision.utils import draw_bounding_boxes
from torchvision.tv_tensors import BoundingBoxFormat

from typing import List, Callable, Union

from .type_structs import Target
from .target import target_box_format_to_enum, target_enum_to_box_format
from .misc import NestedTensor

def plot_switft_dataset(img: Union[torch.Tensor, PILImage], target: Target | None = None) -> Figure:
    """Plot an image with bounding boxes and labels.

    Parameters
    ----------
    img : Union[torch.Tensor, PILImage]
        The image to plot.
    target : Target, optional
        A dictionary containing the target annotations for the image, by default None.

    Returns
    -------
    Figure
        The plotted figure.
    """
    img = to_image(img)
    
    if target is not None:
        labels = target["labels"]
        labels = [f"cls: {label}" for i, label in enumerate(labels)]
        boxes = target["boxes"]
        if target["box_format"] != BoundingBoxFormat.XYXY:
            boxes = box_convert(boxes, in_fmt=target_box_format_to_enum(target["box_format"]).value, out_fmt="XYXY")
        boxes = clip_boxes_to_image(boxes, img.shape[-2:]) # type: ignore[call-overload]
        
        line_width = int(max(img.shape[-2:]) / 500)
        font_size = int(max(img.shape[-2:]) / 50)
        
        img = draw_bounding_boxes(img, boxes, labels, width=line_width, font_size=font_size, font="DejaVuSans", colors="green")
    
    fig = plt.figure()
    plt.imshow(img.permute(1, 2, 0).numpy())
    
    return fig

def plot_switft_dataset_batch(samples: NestedTensor, targets: List[Target], width_plots = 2, images_format: Callable | None = None) -> Figure:
    """
    Plots a batch of images with their corresponding bounding boxes (if provided).

    Parameters
    ----------
    samples : NestedTensor
        A batch of images with shape (batch_size, channels, height, width).
    targets : List[Target]
        A list of targets, where each target is a dictionary containing the following keys:
            - "boxes": A tensor of shape (N, 4) containing the coordinates of N bounding boxes in (x1, y1, x2, y2) format.
            - "labels": A tensor of shape (N,) containing the class labels for each bounding box.
            - "box_format": A string indicating the format of the bounding box coordinates. Can be one of ["XYXY", "XYWH"].
    width_plots : int, optional
        The number of images to plot in each row, by default 2.
    images_format : Callable | None, optional
        A function that takes a tensor of shape (batch_size, channels, height, width) and returns a tensor of the same shape,
        Used to convert the images to the desired de-normalize the images and convert them to a datatype torch.uint8,
        by default None.

    Returns
    -------
    Figure
        A matplotlib Figure object containing the plotted images.
    """
    tensors = samples.tensors
    if images_format is not None:
        tensors = images_format(tensors)
    assert tensors.dtype == torch.uint8, "The images must be of type torch.uint8"
    masks = samples.mask
    assert masks is not None, "The NestedTensor must have a mask"
    
    bs = samples.shape[0]
    
    plot_grid_h = 1+bs//width_plots if width_plots//bs == 0 else width_plots//bs
    plot_grid_w = width_plots
    
    fig = plt.figure(dpi = 500)
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                     nrows_ncols=(plot_grid_h, plot_grid_w), # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     share_all=True,
                     )
    
    for b in range(bs):
        idx_h, idx_w = b//width_plots, b%width_plots
        img = tensors[b]
        mask = masks[b]
        valid_h = (~mask).sum(0)[0]
        valid_w = (~mask).sum(1)[0]
        
        line_width = int(max((valid_h, valid_w)) / 500) # type: ignore[operator]
        font_size = int(max((valid_h, valid_w)) / 50) # type: ignore[operator]
        
        if targets is not None:
            labels = targets[b]["labels"]
            labels_str = [f"{l}" for l in labels]
            boxes = targets[b]["boxes"]
            if targets[b]["box_format"] != "XYXY":
                boxes = box_convert(boxes, in_fmt=target_box_format_to_enum(targets[b]["box_format"]).value, out_fmt="XYXY")
            if (boxes < 1).all():
                boxes = boxes * torch.tensor([valid_w, valid_h, valid_w, valid_h]).to(boxes) # (N, 4)
            img = draw_bounding_boxes(img, boxes, labels=labels_str, colors="green", width=line_width, font_size=font_size, font="DejaVuSans")
            
        grid[b].imshow(img.permute(1, 2, 0).cpu().numpy())
        grid[b].axis("off")
            
    return fig
