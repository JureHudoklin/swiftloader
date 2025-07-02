import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util import DatasetToCoco
from swiftloader.util.display import draw_bounding_boxes, draw_keypoints

if __name__ == "__main__":
    
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

    dataset = FolderDataset(
        root_dir="/home/jure/datasets/folder_datasets",
        datasets_info=[{"name": "TIM_1_Zaliti", "scenes": ["TIM_1_Zaliti_scene_4"]}],
        dataset_schema = [
                {"field": "mask_vis", "dtype": "numpy", "loader": loaders.NumpyLoader()},
                {"field": "image", "dtype": "PIL", "loader": loaders.ImageLoader()},
                {"field": "image_annotation", "dtype": "json", "loader": loaders.JsonLoader()},
                {"field": "annotations", "dtype": "json", "loader": loaders.JsonLoader()},
            ],
        drop_last=False,
        shuffle=True,
    )
    
    
    for data in dataset:
        image = data["image"]
        
        image = draw_bounding_boxes(image, data["annotations"])
        image = draw_keypoints(image, data["annotations"])
        
        plt.imshow(image)
        plt.show()
        
        # Show mask if available
        if "mask_vis" in data:
            mask = data["mask_vis"]
            print(mask)
            plt.imshow(image)
            plt.imshow(mask, alpha=0.5)
            plt.show()
        
    exit()

