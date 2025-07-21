from typing import List, Dict, Any, cast
import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm
from PIL.Image import Image as PILImage

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader import FolderDataset, ParquetDataset
from swiftloader.folder_dataset import DataToFolder
from swiftloader import loaders
from swiftloader.util import DatasetToCoco
from swiftloader.util.display import draw_bounding_boxes, draw_keypoints

if __name__ == "__main__":
    root_dir = "/media/jure/ssd/datasets/folder_datasets"
    datasets_info = [{"name": "test", "scenes": ["test3"]}]

    new_dataset_name = "test"
    new_scene_name = "test4"
    
    # Transforms
    dataset = FolderDataset(
        root_dir=root_dir,
        datasets_info=datasets_info, # type: ignore
        dataset_schema = [
                {"field": "mask_vis", "dtype": "numpy", "loader": loaders.NumpyLoader()},
                {"field": "image", "dtype": "PIL", "loader": loaders.ImageLoader()},
                {"field": "image_annotation", "dtype": "json", "loader": loaders.JsonLoader()},
                {"field": "annotations", "dtype": "json", "loader": loaders.JsonLoader()},
            ],
        drop_last=False,
        shuffle=True,
    )

    data_to_folder = DataToFolder(
        root_dir=root_dir,
        dataset_name=new_dataset_name,
        scene_name=new_scene_name
    )
    
    
    image_id = 0
    annotation_id = 0
    
    for data in dataset:
        image = data["image"]
        image = cast(PILImage, image)
        annotations = data["annotations"]
        
        for idx, ann in enumerate(annotations):
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue
            category = ann.get('category_id', 'Object')

            # COCO format: [x, y, width, height]
            x, y, w, h = bbox

            # Get image containing box
            image_box = image.crop((x, y, x + w, y + h))
            
            image_annotation = {
                "width": image_box.width,
                "height": image_box.height,
                "image_id" : image_id,
            }
            ann["annotation_id"] = annotation_id
            ann["image_id"] = image_id
            ann["bbox"] = [0, 0, w, h]
            
            # Generate new data entry
            data_entry = {
                "image": image_box,
                "image_annotation": image_annotation,
                "annotations": [ann],
            }
            
            data_to_folder.add_entry(data_dict=data_entry)
        
      
    exit()

