# Set matplotlib backend 
# This has to be done before importing pyplot
import matplotlib.pyplot as plt
import torch
from functools import partial
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import json
from swiftloader.object_detection import ObjectDetectionDatasetParquet, ParquetDataset
from swiftloader.util.display import plot_switft_dataset

if __name__ == "__main__":
    
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
    
    def format_data(d):
        for data in d:
            data["annotations"] = json.loads(data["annotations"])
            for ann in data["annotations"]["annotations"]:
                if ann["stability_score"] < 0.9:
                    # Remove unstable annotations
                    data["annotations"]["annotations"].remove(ann)
            
            data["annotations"] = json.dumps(data["annotations"])
        return d
        
            
    
    dataset = ObjectDetectionDatasetParquet(
        root_dir = "/media/jure/ssd/datasets/parquet_datasets",
        datasets_info=[{"name": "industrial_objects", "scenes": ["train"]}, {"name": "objects365", "scenes": ["val"]}],
        batch_size=2,
        input_transform=input_transforms,
        base_transform=base_transforms,
        classless=True,
        #format_data=format_data,
    )
    # dataset = ParquetDataset(
    #     root_dir = "/media/jure/ssd/datasets/parquet_datasets",
    #     datasets_info=[{"name": "industrial_objects", "scenes": ["train"]}],
    #     batch_size=2,
    # )
    print(len(dataset))
   
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
    )
    
    for data in tqdm(dataloader):

        images, targets = data
        img = output_transforms(images[0])
        
        # fig = plot_switft_dataset(img, targets[0])
        # plt.show()
        # plt.close(fig)
    
    # dataloader = get_swift_loader(
    #     dataset=dataset,
    #     split="val",
    #     batch_size=2,
    #     num_workers=4,
    #     pin_memory=True,
    #     collate_fn=partial(create_nested_tensor_batch, size_constant=(512, 1024)),
    # )
    
    # itr = iter(dataloader)
    
    # for i in range(len(dataloader)):
    #     batch = next(itr)
    #     fig = plot_switft_dataset_batch(batch.samples, batch.targets, images_format=output_transforms)
    #     plt.show()
