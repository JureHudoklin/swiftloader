import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm
import numpy as np
import time

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util.display import draw_bounding_boxes

if __name__ == "__main__":
    
    # Dataset setup
    root_dir = "/home/jure/datasets/parquet_datasets"
    dataset_name = "objects365"
    scenes = ["val"]
    dataset_schema = [
                {"field": "image", "dtype": "binary", "loader": loaders.ImageLoader},
                {"field": "image_annotation", "dtype": "string", "loader": loaders.JsonLoader},
                {"field": "annotations", "dtype": "string", "loader": loaders.JsonLoader},
            ]
    
    
    # Transforms
    base_transforms = T.Resize((512, 512))
    input_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ]
    )
    output_transforms = T.Compose([
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        T.ToDtype(torch.uint8, scale=True),
    ])
    
    # Dataset
    dataset = ParquetDataset(
        root_dir=root_dir,
        datasets_info=[{"name": dataset_name, "scenes": scenes}],
        format_data=lambda x: {
            "image": np.array(x["image"]),
            "annotations": x["annotations"],
            "dataset_idx": x["dataset_idx"],
        },
    )
    
    dataloder = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda x: x,
        shuffle=False,
        prefetch_factor=4,
    )

    # Load and display
    iterator = iter(dataloder)
    start_time = time.time()
    for _ in tqdm(range(len(dataloder))):
        data = next(iterator)[0]
        image = data["image"]
        annotations = data["annotations"]
                
        if data["dataset_idx"] > 12000:
            break
        
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
        


    
