from typing import Dict, List, Tuple, Callable, Any, Literal, Generator
from pathlib import Path
import warnings
import json
import io
import tempfile
from collections import defaultdict
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as TF
from pycocotools.coco import COCO
from target_utils import Target
from target_utils.util import target_filter
from target_utils.formating import target_set_dtype, target_reset_tvtensor

from swiftloader import FolderDataset, ParquetDataset
from swiftloader import loaders
from swiftloader.util.type_structs import DatasetInfo, CocoCat
from swiftloader.util.misc import HiddenPrints
from .utils import AnyDict

class ObjectDetectionBase:
    def __init__(self,
                 root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 classless: bool = False,
                 transform: Callable | None = None,
                 ):
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.datasets_info = datasets_info
    
        self.transform = transform
        self.classless = classless
        self.cat_map, self.cats = self._build_category_map(self.root_dir, self.datasets_info)
       
    def _build_category_map(
        self, data_root: Path, datasets_info: List[DatasetInfo]
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[int, CocoCat]]:
        cat_map = defaultdict(dict)
        cats = {}
        
        if self.classless:
            cat_map = {}
            cats[1] = {
                "id": 1,
                "name": "object",
                "supercategory": "object",
            }
            for dataset_info in datasets_info:
                cat_map[dataset_info["name"]] = AnyDict(1)
                
        else:
            dataset_cats = {}
            for dataset_info in datasets_info:
                cats_path = data_root / dataset_info["name"] / "categories.json"
                if not cats_path.exists() and not self.classless:
                    raise ValueError(f"Categories file not found for dataset {dataset_info['name']}")
                with open(data_root / dataset_info["name"] / "categories.json") as f:
                    categories = json.load(f)
                    dataset_cats[dataset_info["name"]] = categories
            new_cat_id = 0

            for dataset, categories in dataset_cats.items():
                for cat in categories:
                    new_cat_id += 1
                    cat_map[dataset][cat["id"]] = new_cat_id
                    cat["id"] = new_cat_id
                    cats[new_cat_id] = cat

        return cat_map, cats
    
    def _get_target(self, data: Dict) -> Target:
        image_ann = data["image_annotation"]
        annotations = data["annotations"]
        image_id = image_ann.get("image_id", None)
        
        if image_id is None:
            warnings.warn("Image id not found in image annotations. Setting to 0. Some functions may not work.")
            image_id = 0
        
        w, h = image_ann["width"], image_ann["height"]
        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor(image_ann["image_id"])

        boxes = [obj["bbox"] for obj in annotations]  # xywh
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
        boxes = tv_tensors.BoundingBoxes(boxes, canvas_size=(h, w), format=tv_tensors.BoundingBoxFormat.XYXY)  # type: ignore[call-overload]

        labels = torch.tensor([obj["category_id"] for obj in annotations])

        target = Target(
            image_id=image_id,
            boxes=boxes,
            labels=labels,
            orig_size=size,
            size=size,
            box_format=torch.tensor(0),
            attributes=None,
        )

        return target
    
    def get_categories(self) -> List:
        return list(self.cats.values())

class ObjectDetectionDatasetFolder(FolderDataset, ObjectDetectionBase):
    def __init__(self,
                root_dir: str | Path,
                datasets_info: List[DatasetInfo],
                format_data: Callable[[dict], Any] | None = None,
                dataset_schema: List[Dict[Literal["field", "dtype", "loader"], Any]] =
                    [{"field": "annotations", "dtype": ".json", "loader": loaders.json_loader},
                    {"field": "image_annotation", "dtype": ".json", "loader": loaders.json_loader},
                    {"field": "images", "dtype": ".jpg", "loader": loaders.image_loader}],
                classless: bool = False,
    ):
        FolderDataset.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            dataset_schema=dataset_schema,
            format_data=format_data,
        )
        ObjectDetectionBase.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            classless=classless,
        )
    
    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        target = self._get_target(data)
        image = data["images"]
        
        # Remap category ids
        if self.classless:
            new_labels = torch.ones_like(target["labels"])
        else:
            new_labels = torch.zeros_like(target["labels"])
            for i, label in enumerate(target["labels"]):
                new_labels[i] = self.cat_map[image_data["dataset"]][label.item()]  # type: ignore[index]
        target["labels"] = new_labels
        
        if self.transform is not None:
            image, target = self.transform(image, target)
            target = target_reset_tvtensor(target)
        
        target_set_dtype(target)
        return image, target

    def get_dataset_api(self, valid_categories: List[Dict] | None = None) -> Tuple[COCO, Dict]:
        images, annotations, categories = [], [], []

        ann_id = 0
        for idx in range(len(self)):
            paths, data_info = self._get_data(idx)
            img_ann_path = None
            ann_path = None
            for p in paths:
                if p["field"] == "image_annotation":
                    img_ann_path = p
                elif p["field"] == "annotations":
                    ann_path = p
            if img_ann_path is None or ann_path is None:
                raise ValueError("Image annotations or annotations not found.")

            with open(img_ann_path["data"]) as f:
                img_ann_ = json.load(f)
            
            img_ann = {
                "file_name": data_info["image_name"],
                "height": img_ann_["height"],
                "width": img_ann_["width"],
                "id": img_ann_["image_id"],
            }
            images.append(img_ann)
            
            with open(ann_path["data"]) as f:
                ann = json.load(f)
            
            for obj in ann["annotations"]:
                obj["image_id"] =ann["image_id"],
                obj["id"] = ann_id
                obj["category_id"] = self.cat_map[data_info["dataset"]][obj["category_id"]]
                ann_id += 1
                annotations.append(obj)

        categories = self.get_categories()
        if valid_categories is not None:
            categories = valid_categories

        info = {
            "description": f"Datasets: {self.datasets_info}",
            "data_root": str(self.root_dir),
        }

        dataset = {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            with HiddenPrints():
                json.dump(dataset, f)
                f.flush()
                coco = COCO(f.name)

        return coco, dataset
    
    
class ObjectDetectionDatasetParquet(ParquetDataset, ObjectDetectionBase):
    def __init__(self,
                root_dir: str | Path,
                datasets_info: List[DatasetInfo],
                batch_size: int,
                dataset_schema: List[Dict[Literal["field", "dtype", "loader"], Any]] =
                    [{"field": "annotations", "dtype": "string", "loader": loaders.json_loader},
                    {"field": "image_annotation", "dtype": "string", "loader": loaders.json_loader},
                    {"field": "image", "dtype": "binary", "loader": loaders.image_loader}],
                format_data: Callable[[dict], Any] | None = None,
                batch_format_data: Callable[[List[dict]], List[dict]] | None = None,
                transform: Callable | None = None,
                shuffle: bool = True,
                drop_last: bool = False,
                classless: bool = False,
    ):
        ParquetDataset.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            dataset_schema=dataset_schema,
            batch_size=batch_size,
            format_data=format_data,
            batch_format_data=batch_format_data,
            drop_last=drop_last,
            shuffle=shuffle,
        )
        ObjectDetectionBase.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            classless=classless,
            transform=transform,
        )
        
    def __iter__(self) -> Generator[Tuple[List, List], None, None]:
        for data in super().__iter__():
            images, targets = self.data_to_target(data)
            yield images, targets
    
    def data_to_target(self, data: List[Dict]):
        targets = []
        images = []
        for d in data:
            img = d["image"]
            target = super()._get_target(d)
            target_set_dtype(target)
            
            # Remap category ids
            if self.classless:
                new_labels = torch.ones_like(target["labels"])
            else:
                new_labels = torch.zeros_like(target["labels"])
                for i, label in enumerate(target["labels"]):
                    new_labels[i] = self.cat_map[self.datasets_info[0]["name"]][label.item()]  # type: ignore[index]
                target["labels"] = new_labels
            
            if self.transform is not None:
                img, target = self.transform(img, target)
                target = target_reset_tvtensor(target)
           
            targets.append(target)
            images.append(img)
                        
        del data
        return images, targets
    
    def get_dataset_api(self, valid_categories: List[Dict] | None = None) -> Tuple[COCO, Dict]:
        images, annotations, categories = [], [], []

        ann_id = 0
        for data in super().__iter__():
            for d in data:
                annotations = d["annotations"]
                image_annotation = d["image_annotation"]
            
                img_ann = {
                    "file_name": image_annotation["file_name"],
                    "height": image_annotation["height"],
                    "width": image_annotation["width"],
                    "id": image_annotation["image_id"],
                }
                images.append(img_ann)
                
                for obj in annotations["annotations"]:
                    obj["image_id"] = annotations["image_id"],
                    obj["id"] = ann_id
                    obj["category_id"] = self.cat_map[self.datasets_info[0]["name"]][obj["category_id"]]
                    ann_id += 1
                    annotations.append(obj)

        categories = self.get_categories()
        if valid_categories is not None:
            categories = valid_categories

        info = {
            "description": f"Datasets: {self.datasets_info}",
            "data_root": str(self.root_dir),
        }

        dataset = {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            with HiddenPrints():
                json.dump(dataset, f)
                f.flush()
                coco = COCO(f.name)

        return coco, dataset
