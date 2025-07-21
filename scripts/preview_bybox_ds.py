import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm
from pathlib import Path
from PIL import Image

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
        datasets_info=[{"name": "TIM_1_Zaliti", "scenes": ["TIM_1_Zaliti_scene_4"]}],
        # noise_bbox=[0.05, 0.05, 0.1, 0.1],
        dataset_schema=[
            {"field": "annotations", "dtype": ".json", "loader": loaders.JsonLoader()},
            {"field": "image_annotation", "dtype": ".json", "loader": loaders.JsonLoader()},
            {"field": "image", "dtype": "PIL", "loader": loaders.ImageLoader(out_type="numpy")},
        ],
    )
    
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=lambda x: x,
    # )
    
    path = Path("/home/jure/datasets/MvTEC/TIM_1_Zaliti")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    path_good = path / "train" / "good"
    path_bad = path / "train" / "bad"
    if not path_good.exists():
        path_good.mkdir(parents=True, exist_ok=True)
    
    if not path_bad.exists():
        path_bad.mkdir(parents=True, exist_ok=True)
    
    i = 0
    for data in dataset:
        i += 1
        image = data["image"] # numpy image
        bbox = data["annotation"]["bbox"]
        if data["annotation"]["damaged"] > 1:
            print(data["annotation"]["damaged"])
            print(data["image_annotation"])

        # Save image to path (if damaged save under bad else good)
        # # damaged = data["annotation"]["damaged"]
        # # if damaged:
        # #     save_path = path_bad / f"{i:04d}.png"
        # # else:
        # #     save_path = path_good / f"{i:04d}.png"
            
        # # image_pil = Image.fromarray(image)
        # # image_pil.save(save_path)
        
        x, y, w, h = bbox

        # Crop mask if available
        if "mask_vis" in data:
            mask = data["mask_vis"]
            mask = mask[y:y+h, x:x+w]
            data["mask_vis"] = mask
        

        # plt.imshow(image)
        # plt.show()
        
        # Show mask if available
        plt.imshow(image)
        # plt.imshow(data["mask_vis"], alpha=0.5)
        plt.title(f"Image {i} - cavity: {data['annotation']['cavity']} - damaged: {data['annotation']['damaged']}")
        # plt.show()
        
    print("Finished")
    exit(0)

