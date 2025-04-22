import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader.task import ByBoxDatasetFolder
from swiftloader import loaders
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

    dataset = ByBoxDatasetFolder(
        root_dir="/home/jure/datasets/folder_datasets",
        datasets_info=[{"name": "SM", "scenes": ["pose_detection_v1"]}],
    )
    print(f"Total number of images: {len(dataset)}")
    
    i = 0
    for data in dataset:
        print(f"Image {i}")
        i += 1
        image = data["image"]
        
        # image = draw_bounding_boxes(image, data["annotations"])
        # image = draw_keypoints(image, data["annotations"])
        
        plt.imshow(image)
        plt.show()
        
        # Show mask if available
        if "mask_full" in data:
            mask = data["mask_full"]
            print(mask)
            plt.imshow(image)
            plt.imshow(mask, alpha=0.5)
            plt.show()
        
    exit()

