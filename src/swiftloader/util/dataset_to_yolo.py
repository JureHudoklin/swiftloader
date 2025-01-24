from typing import Literal, Tuple, List
import os
from pathlib import Path
import json
import random

from .type_structs import CocoAnn, CocoCat, CocoImage, CocoInfo

class DatasetToYolo:
    def __init__(self,
                 dataset,
                 save_dir: str | Path,
                 dataset_name: str,
                 split_ratio: Tuple[float, float, float]  = (0.7, 0.2, 0.1),
                 categories: List[str] | None = None
    ):
        
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.categories = categories
        
        self.dataset_path = self.save_dir / self.dataset_name
        
        self._create_dir_structure()

    def _create_dir_structure(self):
        if self.dataset_path.exists():
            raise FileExistsError(f"Dataset {self.dataset_name} already exists in {self.save_dir}")
        
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        if self.split_ratio[0] > 0:
            (self.dataset_path / "train").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "train" / "images").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
        if self.split_ratio[1] > 0:
            (self.dataset_path / "val").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "val" / "images").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "val" / "labels").mkdir(parents=True, exist_ok=True)
        if self.split_ratio[2] > 0:
            (self.dataset_path / "test").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "test" / "images").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "test" / "labels").mkdir(parents=True, exist_ok=True)
        
    def to_yolo_object_detection(self):
        """
        Each entry in the dataset contains a batch of dictionaries, where each dictionary contains the following keys:
        - image: PIL.Image
        - annotations: List[Dict]
            - category_id, bbox, image_id
        - Optional[image_annotation: Dict]
            - width, height, image_id
        
        The function formats the dataset into ultralytics YOLO format and saves it in the specified directory.
        
        dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── data.yaml
        
        """ 
        # Create data.yaml
        data_yaml = {
            "path": str(self.dataset_path),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 0, # Number of classes
            "names": []
        }
        
        if self.categories is not None:
            data_yaml["nc"] = len(self.categories)
            data_yaml["names"] = self.categories
        
        categories = set()
        image_id = 0
        annotation_id = 0

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_size = int(self.split_ratio[0] * dataset_size)
        val_size = int(self.split_ratio[1] * dataset_size)
        test_size = int(self.split_ratio[2] * dataset_size)
        
        for batch in self.dataset:
            for item in batch:
                if image_id <= train_size:
                    split = "train"
                elif image_id <= train_size + val_size:
                    split = "val"
                else:
                    split = "test"
                
                image = item['image']
                annotations = item['annotations']
                image_annotation = item.get('image_annotation', {})
                
                file_name = f"image_{image_id}.jpg"
                
                # Process image
                image.save(self.dataset_path / split / "images" / file_name)
                
                # Process annotations
                with open(self.dataset_path / split / "labels" / f"image_{image_id}.txt", "w") as f:
                    for ann in annotations:
                        category_id = ann["category_id"]
                        category_id = 0 
                        categories.add(category_id)
                        
                        bbox = ann["bbox"]
                        x, y, w, h = bbox
                        x_center = x + w / 2
                        y_center = y + h / 2
                        
                        # Normalize the coordinates
                        x_center /= image.width
                        y_center /= image.height
                        w /= image.width
                        h /= image.height
                        
                        f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")
        
                        annotation_id += 1
                        
                image_id += 1
                
        # Process categories
        if self.categories is None:
            self.categories = [f"category_{cat_id}" for cat_id in categories]
            
        data_yaml["nc"] = len(self.categories)
        data_yaml["names"] = self.categories
        
        with open(self.dataset_path / "data.yaml", "w") as f:
            json.dump(data_yaml, f)
            
        print(f"Dataset saved in {self.dataset_path}")