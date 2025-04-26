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

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util.type_structs import DatasetInfo

class ByBoxDatasetFolder(FolderDataset):
    def __init__(self,
                root_dir: str | Path,
                datasets_info: List[DatasetInfo],
                format_data: Callable[[dict], Any] | None = None,
                noise_bbox: list[float] = [0.0, 0.0, 0.0, 0.0],
    ):
        FolderDataset.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            dataset_schema=[{"field": "annotations", "dtype": ".json", "loader": loaders.JsonLoader()},
                            {"field": "image_annotation", "dtype": ".json", "loader": loaders.JsonLoader()},
                            {"field": "image", "dtype": "PIL", "loader": loaders.ImageLoader(out_type="pil")}],
            format_data=None,
        )
        self.noise_bbox = noise_bbox
        self._format_data = format_data
        
        self.ann_per_image = self.setup()
        self.ann_cumulative = np.cumsum(self.ann_per_image)

    def setup(self):
        # get number of annotations for each image
        ann_per_image = np.array([])
        for i in range(super().__len__()):
            annotations = self.get_image_annotations(i)
            if annotations is None:
                ann_per_image = np.append(ann_per_image, 0)
            else:
                N_annotations = len(annotations)
                ann_per_image = np.append(ann_per_image, N_annotations)
        
        return ann_per_image

    def get_image_annotations(self, idx: int):
        paths, data_info = self._get_data(idx)
            
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
        
        data = super().__getitem__(int(img_idx))
        image = data.get("image")
        annotation = data.get("annotations", [])
        image_data = data.get("image_annotation")
        
        if ann_idx >= len(annotation):
            raise IndexError(f"Annotation index {ann_idx} is out of bounds for image {img_idx}.")
        
        annotation = annotation[int(ann_idx)]

        # Crop image to bounding box
        if annotation["bbox"] is not None and image is not None:
            x, y, w, h = annotation["bbox"]
            
            if self.noise_bbox is not None:
                x = int(x + self.noise_bbox[0]*np.random.rand()* w)
                y = int(y + self.noise_bbox[1]*np.random.rand() * h)
                w = int(w + self.noise_bbox[2]*np.random.rand() * w)
                h = int(h + self.noise_bbox[3]*np.random.rand() * h)
            
            # Ensure bounding box is within image bounds
            x = max(0, min(x, image.size[0] - 1))
            y = max(0, min(y, image.size[1] - 1))
            w = max(1, min(w, image.size[0] - x))
            h = max(1, min(h, image.size[1] - y))
            
            image_crop = image.crop((x, y, x + w, y + h))
        
        
        new_data = {
            "image": image_crop,
            "annotation": annotation,
            "image_annotation": {
                "width": image_crop.size[0],
                "height": image_crop.size[1],
                "original_width": image.size[0],
                "original_height": image.size[1],
            }
        }
        
        return self._format_data(new_data) if self._format_data is not None else new_data
        
       