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
        root_dir="/media/jure/ssd/datasets/folder_datasets",
        datasets_info=[{"name": "test", "scenes": ["test9"]}],
        noise_bbox=[0.0, 0.0, 0.0, 0.0],
        crop_to_bbox=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )
    
    
    i = 0
    for data in dataset:
        i += 1

        image = data["image"]
        mask_full = data["mask_full"]
        annotation = data["annotation"]
        bbox = annotation["bbox"] # xywh

        mask = mask_full[:, :, annotation["mask_id"]]

        
        # Crop mask to the bounding box
        x1, y1, width, height = annotation["bbox"]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x1 + width)
        y2 = int(y1 + height)
        #mask = mask[y1:y2, x1:x2]
        
        plt.imshow(image)
        plt.imshow(mask, alpha=0.1)
        # Plot bounding box
        plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, edgecolor='red', facecolor='none'))
        plt.show()
            
        
        # image = draw_bounding_boxes(image, data["annotations"])
        # image = draw_keypoints(image, data["annotations"])
        
    #     plt.imshow(image)
    #     plt.show()
        
    #     # Show mask if available
    #     if "mask_full" in data:
    #         mask = data["mask_full"]
    #         print(mask)
    #         plt.imshow(image)
    #         plt.imshow(mask, alpha=0.5)
    #         plt.show()
        
    # print("Finished")
    # exit(0)

