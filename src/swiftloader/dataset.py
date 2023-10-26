import os
import time
import logging
import json
import tempfile
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, Literal, Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from pycocotools.coco import COCO

from .util.type_structs import (
    Target,
    CocoCat,
)
from .util.target import target_recalculate_area, target_set_dtype

class SwiftDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        datasets: List[str],
        base_transforms: Optional[Callable] = None,
        input_transforms: Optional[Callable] = None,
        keep_crowded: bool = True,
    ) -> None:
        """Dataloader for COCO dataset

        Parameters
        ----------
        img_folder : str
            Path to the folder where the images are stored
        ann_file : str
            Path to the annotation file
        """
        self.data_root = Path(data_root)
        self.datasets = sorted(datasets)
        
        self._check_datasets_exist(self.data_root, self.datasets)
        self.images_data = self._load_dataset(self.data_root, self.datasets)
        self.cat_map, self.cats = self._build_category_map(self.data_root, self.datasets)
        logging.info(f"Loaded {len(self.images_data)} images from {datasets} datasets")
        
        self.base_transforms = base_transforms
        self.input_transforms = input_transforms
        self.keep_crowded = keep_crowded

        self.fail_save = self.__getitem__(0)
        
    def _check_datasets_exist(self, data_root: Path, datasets: List[str]):
        for dataset in datasets:
            assert (data_root / dataset).exists(), f"Dataset {dataset} does not exist"
            
    def _load_dataset(self, data_root: Path, datasets: List[str]) -> np.ndarray:
        images_ = []
        for dataset in datasets:
            # Get all subfolders in the dataset folder (exclude files)
            scenes = sorted((data_root / dataset).glob("*"))
            scenes = [scene for scene in scenes if scene.is_dir()]
            for scene in scenes:
                image_annotations_path = scene / "image_annotations"
                files = os.scandir(image_annotations_path)
                images_.extend([f"{dataset}/{scene.name}/{file.name}" for file in files if file.is_file()])
                
        if len(images_) < 1000000:
            images_.sort()
        else:
            logging.warning("Sorting images will take a long time. Skipping sorting. Images will be loaded in random order.")
            
        images = np.array(images_).astype(np.string_)
        return images
            
    def _build_category_map(self, data_root: Path, datasets: List[str]) -> Tuple[Dict[str, Dict[int, int]], Dict[int, CocoCat]]:
        cat_map = {}
        cats = {}
        dataset_cats = {}
        for dataset in datasets:
            with open(data_root / dataset / "categories.json") as f:
                categories = json.load(f)
                dataset_cats[dataset] = categories
                
        new_cat_id = 1
        for dataset, categories in dataset_cats.items():
            if dataset not in cat_map:
                cat_map[dataset] = {}
            for cat in categories:
                cat_map[dataset][cat["id"]] = new_cat_id
                cat["id"] = new_cat_id
                cats[new_cat_id] = cat
                new_cat_id += 1
                
        return cat_map, cats

    def _get_data_paths(self, idx: int) -> Tuple[Path, Path, Path, Dict]:
         # Get data paths
        image_data = self.images_data[idx].decode("utf-8")
        dataset, scene, image_name = image_data.split("/")
        image_name = image_name.split(".")[0]
        image_data = {"dataset": dataset, "scene": scene, "image_name": image_name}
        
        img_ann_path = self.data_root / dataset / scene / "image_annotations"  / f"{image_name}.json"
        ann_path = self.data_root / dataset / scene / "annotations"  / f"{image_name}.json"
        image_path = self.data_root / dataset / scene /  "images" / f"{image_name}.png"
        
        return image_path, ann_path, img_ann_path, image_data

    def _get_target(self, image: PILImage, ann, img_ann) -> Target:
        w, h = image.size
        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor(img_ann["id"])

        if not self.keep_crowded:
            ann = [obj for obj in ann if "iscrowd" not in obj or obj["iscrowd"] == 0]
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in ann]
        )

        boxes = [obj["bbox"] for obj in ann]  # xywh
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        area = boxes[:, 2] * boxes[:, 3]
        boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
        boxes = tv_tensors.BoundingBoxes(boxes, canvas_size=(h, w), format=tv_tensors.BoundingBoxFormat.XYXY) # type: ignore[call-overload]

        labels = torch.tensor([obj["category_id"] for obj in ann])

        target = Target(
            image_id=image_id,
            boxes=boxes,
            labels=labels,
            iscrowd=iscrowd,
            area=area,
            orig_size=size,
            size=size,
            box_format=torch.tensor(0),
        )

        return target

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx: int) -> Tuple[Any, Target]:
        image_path, ann_path, img_ann_path, image_data= self._get_data_paths(idx)
        
        try:   
            # Load image and annotations
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image = ImageOps.exif_transpose(image)
            with open(img_ann_path) as f:
                img_ann = json.load(f)
            with open(ann_path) as f:
                ann = json.load(f)
                
            target = self._get_target(image, ann, img_ann)
            target["image_id"] = torch.tensor([idx])
            
            # Remap category ids
            new_labels = torch.zeros_like(target["labels"])
            for i, label in enumerate(target["labels"]):
                new_labels[i] = self.cat_map[image_data["dataset"]][label.item()] # type: ignore[index]
            target["labels"] = new_labels
                
            if self.base_transforms is not None:
                image, target = self.base_transforms(image, target)
                target = target_recalculate_area(target)
                
            if self.input_transforms is not None:
                image, target = self.input_transforms(image, target)
                target = target_recalculate_area(target)
                
            target_set_dtype(target)
            return image, target
        except:
            logging.warning(f"Failed to load image or annotations: \n \
                         (img_path={image_path},\n ann_path={ann_path},\n img_ann_path={img_ann_path},\n idx={idx})")
            return self.fail_save

    def get_categories(self) -> List:
        return list(self.cats.values())
    
    def get_dataset_api(self, valid_categories: List[Dict] | None = None) -> COCO:
        images, annotations, categories = [], [], []
        
        ann_id = 0
        for idx in range(len(self)):
            image_path, ann_path, img_ann_path, _ = self._get_data_paths(idx)

            with open(img_ann_path) as f:
                img_ann = json.load(f)
                img_ann["id"] = idx
                images.append(img_ann)
            
            with open(ann_path) as f:
                ann = json.load(f)
                for obj in ann:
                    obj["image_id"] = idx
                    obj["id"] = ann_id
                    ann_id += 1
                    annotations.append(obj)
        
        categories = self.get_categories()
        if valid_categories is not None:
            categories = valid_categories
        
        info = {
            "description": f"Datasets: {self.datasets}",
            "data_root": str(self.data_root),
        }
        
        dataset = {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(dataset, f)
            f.flush()
            coco = COCO(f.name)
                        
        return coco

# if __name__ == "__main__":
    
#     base_transforms = T.Resize((512, 512))
    
#     dataloader = DatasetLoader(
#         "/home/jure/datasets/OBJECTS_DATASET",
#         ["AHIL", "ICBIN"],
#         "val",
#         input_transforms=None,
#         base_transforms=base_transforms,
#         keep_crowded=True,
#     )
#     dataloader.get_dataset_api(None)
    
    # img, target = dataloader[0]
    
    # plt.imshow(img)
    # plt.savefig("test.png")
    #print(target)