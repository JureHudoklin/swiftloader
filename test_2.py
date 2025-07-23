import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util import DatasetToCoco, DatasetToYolo, MaskRLE
from swiftloader.util.display import draw_bounding_boxes

def collate_fn(batch):
    # Return list of dictionaries
    return batch

if __name__ == "__main__":

    
    dataset = FolderDataset(
        root_dir="/home/jure/datasets/folder_datasets",
        datasets_info=[{"name": "TIM_1", "scenes": ["scene_2_annotated"]}], # "test", "test1", "test2", "SM_train_real", "SM_val_real"
        dataset_schema = [
                {"field": "image", "dtype": ".jpg", "loader": loaders.ImageLoader()},
                {"field": "image_annotation", "dtype": ".json", "loader": loaders.JsonLoader()},
                {"field": "annotations", "dtype": ".json", "loader": loaders.JsonLoader()},
                {"field": "mask_full", "dtype": "numpy", "loader": loaders.NumpyLoader()},
                {"field": "mask_vis", "dtype": "numpy", "loader": loaders.NumpyLoader()},
            ],
    )

    
    
    for data in dataset:
        
        dataset_idx = data["dataset_idx"]
        mask_vis = data["mask_vis"]
        mask_full = data["mask_full"]

        mask_vis_rle = MaskRLE(masks=mask_vis)
        mask_full_rle = MaskRLE(masks=mask_full)
        
        print(mask_vis_rle.to_dict())
            
        dataset.modify_entry(
            idx = dataset_idx,
            data_dict={
                "mask_vis_rle": mask_vis_rle,
                "mask_full_rle": mask_full_rle,}
        )