import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader import FolderDataset, ParquetDataset
from swiftloader.task.object_detection import ObjectDetectionDatasetParquet
from swiftloader import loaders
from swiftloader.util import DatasetToCoco
from swiftloader.util.display import draw_bounding_boxes

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
    
    # dataset = ParquetDataset(
    #     root_dir="/media/jure/ssd/datasets/parquet_datasets",
    #     datasets_info=[{"name": "industrial_objects", "scenes": ["train_v3"]}],
    #     dataset_schema = [
    #             {"field": "image", "dtype": "binary", "loader": loaders.image_loader},
    #             {"field": "image_annotation", "dtype": "string", "loader": loaders.json_loader},
    #             {"field": "annotations", "dtype": "string", "loader": loaders.json_loader},
    #         ],
    #     batch_size=1,
    #     drop_last=False,
    #     shuffle=True,
    # )
    
    dataset = FolderDataset(
        root_dir="/media/jure/ssd/datasets/folder_datasets",
        datasets_info=[{"name": "SM_object_detection", "scenes": ["SM_train_real", "test2"]}],
        dataset_schema = [
                {"field": "image", "dtype": ".jpg", "loader": loaders.image_loader},
                {"field": "image_annotation", "dtype": ".json", "loader": loaders.json_loader},
                {"field": "annotations", "dtype": ".json", "loader": loaders.json_loader},
            ],
        drop_last=False,
        shuffle=True,
    )
    
    
    for data in dataset:
        print(data)
        image = data["image"]
        
        image = draw_bounding_boxes(image, data["annotations"])
        
        plt.imshow(image)
        plt.show()
        
    exit()
    
    ds_to_coco = DatasetToCoco(
        dataset=dataset,
        save_dir="/media/jure/ssd/datasets/coco_datasets",
        dataset_name="industrial_objects",
        split_ratio=[0.8, 0.2, 0],
    )
    
    ds_to_coco.to_coco()

    
