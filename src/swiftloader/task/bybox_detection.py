from typing import Dict, List, Tuple, Callable, Any, Literal, Generator
from pathlib import Path
import warnings
import json
import copy
import tempfile
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage

import numpy as np
import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as TF
import albumentations as A

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util.type_structs import DatasetInfo

class ByBoxDatasetFolder():
    def __init__(self,
                dataset: FolderDataset,
                format_data: Callable[[dict], Any] | None = None,
                *,
                noise_bbox: list[float] = [0.0, 0.0, 0.0, 0.0],
                crop_to_bbox: bool = True,
    ):
        self._dataset = dataset
        self.noise_bbox = noise_bbox
        self.crop_to_bbox = crop_to_bbox    
        self._format_data = format_data
        
        self.ann_per_image = self.setup()
        self.ann_cumulative = np.cumsum(self.ann_per_image)

    def setup(self):
        # get number of annotations for each image
        ann_per_image = np.array([])
        for i in range(self._dataset.__len__()):
            annotations = self.get_image_annotations(i)
            if annotations is None:
                ann_per_image = np.append(ann_per_image, 0)
            else:
                N_annotations = len(annotations)
                ann_per_image = np.append(ann_per_image, N_annotations)
        
        return ann_per_image

    def get_image_annotations(self, idx: int):
        paths, data_info = self._dataset._get_data(idx)
            
        annotation_path = [path for path in paths if path["field"] == "annotations"][0]
        if not annotation_path["data"].parent.exists():
            return None
        annotations = annotation_path["loader"](annotation_path["data"])
        return annotations
    
    def __len__(self):
        # Calculate total number of annotations
        total_annotations = np.sum(self.ann_per_image)
        return int(total_annotations)
           
    def __getitem__(self, idx: int):
        # Calculate image index
        if idx == 0:
            img_idx = 0
            ann_idx = 0
        else:
            img_idx = (self.ann_cumulative-1) // idx
            img_idx = np.argmax(img_idx > 0)
            
            # Check if image contains any annotations
            ann_per_image_ = self.ann_per_image[img_idx:]
            if len(ann_per_image_) == 0:
                img_idx_offset = 0
            else:
                img_idx_offset = np.argmax(ann_per_image_ > 0)
            if img_idx_offset > 0:
                img_idx += img_idx_offset
            
            # Calculate annotation index
            ann_idx = idx - self.ann_cumulative[img_idx - 1] if img_idx > 0 else idx
        

        if img_idx >= len(self):
            raise IndexError(f"Image index {img_idx} is out of bounds for dataset of size {len(self)}.")

        data = self._dataset[int(img_idx)]
        image = data.get("image")
        image_h, image_w = image.shape[0], image.shape[1]
        annotation = data.get("annotations", [])
        image_annotation = data.get("image_annotation")
        
        if ann_idx >= len(annotation):
            raise IndexError(f"Annotation index {ann_idx} is out of bounds for image {img_idx}.")
        
        annotation = annotation[int(ann_idx)]

        # Crop image to bounding box
        if annotation["bbox"] is not None and self.crop_to_bbox:
            x, y, w, h = annotation["bbox"]
            annotation["bbox_original"] = [x, y, w, h]
            
            if self.noise_bbox is not None:
                x = int(x + self.noise_bbox[0]*(np.random.rand()-0.5)* w)
                y = int(y + self.noise_bbox[1]*(np.random.rand()-0.5) * h)
                w = int(w + self.noise_bbox[2]*np.random.rand() * w)
                h = int(h + self.noise_bbox[3]*np.random.rand() * h)
                
            
            # Ensure bounding box is within image bounds
            x = max(0, min(x, image_w - 1))
            y = max(0, min(y, image_h - 1))
            w = max(1, min(w, image_w - x))
            h = max(1, min(h, image_h - y))
            
            annotation["bbox"] = [x, y, w, h]
            image_crop = A.crop(img=image, x_min=x, y_min=y, x_max=x + w, y_max=y + h)
        else:
            image_crop = image
            
            
        if image_annotation is None:
            image_annotation = {}
            
        image_crop_h, image_crop_w = image_crop.shape[0], image_crop.shape[1]
        image_annotation["width"] = image_crop_w
        image_annotation["height"] = image_crop_h
        image_annotation["original_width"] = image_w
        image_annotation["original_height"] = image_h
        
        
        new_data = {
            "image": image_crop,
            "annotation": annotation,
            "image_annotation": image_annotation,
        }
        
        # Add any other fields from the original data
        for key, value in data.items():
            if key not in new_data:
                new_data[key] = value
                
        return self._format_data(new_data) if self._format_data is not None else new_data
        
       