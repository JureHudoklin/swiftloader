from typing import Dict, List, Tuple, Callable, Any, Literal, Generator
from pathlib import Path
import numpy as np
import json
import warnings
import tempfile
import functools
import random
from collections import defaultdict
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage

import torch
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


class TemplateLoader:
    def __init__(self,
                 root_dir: str | Path,
                 return_n_templates: int = 1,
                 use_cache: bool = True,
                 load_transform: Callable | None = None,
                 out_transform: Callable | None = None,
    ) -> None:
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.use_cache = use_cache
        self.load_transform = load_transform
        self.out_transform = out_transform
        self.return_n_templates = return_n_templates
        
    @functools.cache
    def call_with_caching(self, object_name: str) -> Any:
        return self.call_no_caching(object_name)
        
    def call_no_caching(self, object_name: str) -> Any:
        object_dir_path = self.root_dir / object_name
        if not object_dir_path.exists():
            raise ValueError(f"Object {object_name} not found in {self.root_dir}")
        
        image_paths = [str(file) for file in object_dir_path.iterdir() if file.is_file()]
        templates = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image = ImageOps.exif_transpose(image)
            if self.load_transform is not None:
                image = self.load_transform(image)
                
            templates.append(image)
                        
        samples = random.sample(templates, self.return_n_templates)
        return samples
    
    def __call__(self, object_name: str) -> Any:
        if self.use_cache:
            samples = self.call_with_caching(object_name)
        else:
            samples = self.call_no_caching(object_name)
            
        if self.out_transform is not None:
            samples = [self.out_transform(sample) for sample in samples]    
            
        return samples
        

class TempleteDetectionBase:
    def __init__(self,
                 root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 object_loader: Callable,
                 transform: Callable | None = None,
                 ):
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.datasets_info = datasets_info
    
        self.transform = transform
        self.cat_map, self.cats = self._build_category_map(self.root_dir, self.datasets_info)
        self.object_loader = object_loader
        
    def _build_category_map(
        self, data_root: Path, datasets_info: List[DatasetInfo]
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[int, CocoCat]]:
        cat_map = defaultdict(dict)
        cats = {}
        
        dataset_cats = {}
        for dataset_info in datasets_info:
            cats_path = data_root / dataset_info["name"] / "categories.json"
            if not cats_path.exists():
                raise ValueError(f"Categories file not found for dataset {dataset_info['name']}")
            with open(data_root / dataset_info["name"] / "categories.json") as f:
                categories = json.load(f)
                dataset_cats[dataset_info["name"]] = categories
        max_cat_id = 0

        cat_names = {}
        for dataset, categories in dataset_cats.items():
            for cat in categories:
                new_cat_id = cat_names.get(cat["name"], None)
                if new_cat_id is None:
                    max_cat_id += 1
                    new_cat_id = max_cat_id
                    cat_names[cat["name"]] = new_cat_id
                    
                cat_map[dataset][cat["id"]] = new_cat_id
                cat["id"] = new_cat_id
                cats[new_cat_id] = cat

        return cat_map, cats
    
    def _get_target(self, data: Dict) -> Target:
        image_ann = data["image_annotations"]
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
        
    def _load_objects(self, objects_root_dir: Path) -> Dict[str, List[str]]:
        objects = {}
        for object_dir in objects_root_dir.iterdir():
            object_name = object_dir.name
            
            files = [str(file) for file in object_dir.iterdir() if file.is_file()]
            files = np.array(sorted(files)).astype(np.string_)
            objects[object_name] = files

        return objects
        
        
class TemplateDetectionDatasetParqeut(ParquetDataset, TempleteDetectionBase):
    def __init__(self,
                 root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 batch_size: int,
                 object_loader: Callable,
                 dataset_schema: List[Dict[Literal["field", "dtype", "loader"], Any]] =
                     [{"field": "annotations", "dtype": "string", "loader": loaders.json_loader},
                     {"field": "image_annotations", "dtype": "string", "loader": loaders.json_loader},
                     {"field": "image", "dtype": "binary", "loader": loaders.image_loader}],
                 format_data: Callable[[dict], Any] | None = None,
                 batch_format_data: Callable[[List[dict]], List[dict]] | None = None,
                 transform: Callable | None = None,
                 shuffle: bool = True,
                 drop_last: bool = False,
    ) -> None:
        
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
        TempleteDetectionBase.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            transform=transform,
            object_loader=object_loader,
        )
        
    def __iter__(self) -> Generator[Tuple[List, List, List], None, None]:
        for data in super().__iter__():
            images, templates, targets = self.data_to_target(data)
            yield images, templates, targets 
            
    def data_to_target(self, data: List[Dict]):
        targets = []
        templates = []
        images = []
        for d in data:
            img = d["image"]
            info = d["image_annotations"]["info"]
            target = self._get_target(d)
            target_set_dtype(target)
            
            # Remap category ids
            dataset_name = info["name"]
            new_labels = torch.zeros_like(target["labels"])
            for i, label in enumerate(target["labels"]):
                new_labels[i] = self.cat_map[dataset_name][label.item()]  # type: ignore[index]
            target["labels"] = torch.as_tensor(new_labels)
            
            # Pick object and change labels to similarity labels
            selected_label = random.choice(list(target["labels"]))
            cat = self.cats[int(selected_label.item())]
            template = self.object_loader(cat["name"])
            similarity_labels = torch.zeros_like(target["labels"])
            similarity_labels[target["labels"] == selected_label] = 1
            target["labels"] = similarity_labels
            
            if self.transform is not None:
                img, target = self.transform(img, target)
                target = target_reset_tvtensor(target)
           
            targets.append(target)
            images.append(img)
            templates.append(template)
                        
        del data
        return images, templates, targets