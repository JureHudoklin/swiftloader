from typing import Literal, Tuple
import os
from pathlib import Path
import json
import random

from .type_structs import CocoAnn, CocoCat, CocoImage, CocoInfo



class DatasetToCoco:
    def __init__(self,
                 dataset,
                 save_dir: str | Path,
                 dataset_name: str,
                 split_ratio: Tuple[float, float, float]  = (0.7, 0.2, 0.1)
    ):
        
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        
        self.dataset_path = self.save_dir / self.dataset_name
        
        self._create_dir_structure()

    def _create_dir_structure(self):
        if self.dataset_path.exists():
            raise FileExistsError(f"Dataset {self.dataset_name} already exists in {self.save_dir}")
        
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "annotations").mkdir(parents=True, exist_ok=True)       
        (self.dataset_path / "images").mkdir(parents=True, exist_ok=True)
        
        (self.dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "images" / "test").mkdir(parents=True, exist_ok=True)
        
    def to_coco(self):
        """
        Each entry in the dataset contains a batch of dictionaries, where each dictionary contains the following keys:
        - image: PIL.Image
        - annotations: List[Dict]
            - category_id, bbox, image_id
        - Optional[image_annotation: Dict]
            - width, height, image_id
        
        The function formats the dataset into COCO format and saves it in the specified directory.
        
        dataset/
            ├── images/
            │   ├── train/
            │   │   ├── image1.jpg
            │   │   ├── image2.jpg
            │   │   └── ...
            │   ├── val/
            │   │   ├── image1.jpg
            │   │   ├── image2.jpg
            │   │   └── ...
            └── annotations/
                ├── instances_train.json
                ├── instances_val.json
                └── ...
        """ 
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [],
            "info": CocoInfo(
                name=self.dataset_name,
                year=2023,
                version="1.0",
                description=f"Converted {self.dataset_name} dataset",
                contributor="SwiftLoader",
                url="",
                date_created=""
            )
        }

        categories = set()
        image_id = 0
        annotation_id = 0

        for batch in self.dataset:
            for item in batch:
                image = item['image']
                annotations = item['annotations']
                image_annotation = item.get('image_annotation', {})

                # Process image
                file_name = f"image_{image_id}.jpg"
                image_path = self.dataset_path / "images" / file_name
                image.save(image_path)

                width = image_annotation.get('width', image.width)
                height = image_annotation.get('height', image.height)

                coco_image = CocoImage(
                    id=image_id,
                    width=width,
                    height=height,
                    file_name=file_name,
                    license=0,
                    flickr_url="",
                    coco_url="",
                    date_captured=""
                )
                coco_format["images"].append(coco_image)

                # Process annotations
                for ann in annotations:
                    category_id = ann['category_id']
                    categories.add(category_id)

                    x, y, w, h = ann['bbox']
                    area = w * h

                    coco_ann = CocoAnn(
                        image_id=image_id,
                        category_id=category_id,
                        bbox=[x, y, w, h],
                        area=area,
                        segmentation=None,
                        iscrowd=0,
                        id=annotation_id
                    )
                    coco_format["annotations"].append(coco_ann)
                    annotation_id += 1

                image_id += 1

        # Process categories
        for cat_id in categories:
            coco_cat = CocoCat(
                id=cat_id,
                name=f"category_{cat_id}",
                supercategory="",
                isthing=1,
                color=None
            )
            coco_format["categories"].append(coco_cat)

        # Split the dataset
        train_ratio, val_ratio, test_ratio = self.split_ratio
        num_images = len(coco_format["images"])
        train_size = int(num_images * train_ratio)
        val_size = int(num_images * val_ratio)

        indices = list(range(num_images))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }

        # Save split datasets
        for split, split_indices in splits.items():
            split_coco = {
                "images": [coco_format["images"][i] for i in split_indices],
                "annotations": [ann for ann in coco_format["annotations"] if ann["image_id"] in split_indices],
                "categories": coco_format["categories"],
                "info": coco_format["info"]
            }

            # Move images to split-specific directories and update file paths
            for img in split_coco["images"]:
                old_path = self.dataset_path / "images" / img["file_name"]
                new_path = self.dataset_path / "images" / split / img["file_name"]
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
                img["file_name"] = str(new_path.relative_to(self.dataset_path / "images"))

            # Save JSON
            json_path = self.dataset_path / "annotations" / f"instances_{split}.json"
            with open(json_path, 'w') as f:
                json.dump(split_coco, f, indent=2)

        print(f"COCO dataset saved to {self.dataset_path}")
    