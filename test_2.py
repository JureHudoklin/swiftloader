import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util import DatasetToCoco, DatasetToYolo
from swiftloader.util.display import draw_bounding_boxes

def collate_fn(batch):
    # Return list of dictionaries
    return batch

if __name__ == "__main__":

    
    dataset = FolderDataset(
        root_dir="/home/jure/datasets/folder_datasets",
        datasets_info=[{"name": "TIM_1_Zaliti", "scenes": ["TIM_1_Zaliti_scene_4"]}], # "test", "test1", "test2", "SM_train_real", "SM_val_real"
        dataset_schema = [
                {"field": "image", "dtype": ".jpg", "loader": loaders.ImageLoader()},
                {"field": "image_annotation", "dtype": ".json", "loader": loaders.JsonLoader()},
                {"field": "annotations", "dtype": ".json", "loader": loaders.JsonLoader()},
                {"field": "mask_vis", "dtype": "numpy", "loader": loaders.NumpyLoader()},
            ],
    )

    
    
    for data in dataset:
        
        annotations = data["annotations"]
        dataset_idx = data["dataset_idx"]
        
        
        new_annotations = []
        
        for i, annotation in enumerate(annotations):
            annotation["mask_idx"] = i
            new_annotations.append(annotation)
            
        dataset.modify_entry(
            idx = dataset_idx,
            data_dict={
                "annotations": new_annotations}
        )