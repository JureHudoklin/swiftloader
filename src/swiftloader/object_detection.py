from typing import Dict, List, Tuple, Callable, Any, Literal
from pathlib import Path
import json
import io
import tempfile
from collections import defaultdict
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage

import torch
from torchvision import tv_tensors
from torchvision.ops import box_convert
from pycocotools.coco import COCO
from target_utils import Target
from target_utils.util import target_filter
from target_utils.formating import target_set_dtype

from .folder_dataset import FolderDataset
from .parquet_dataset import ParquetDataset 
from .util.type_structs import DatasetInfo, CocoCat
from .util.misc import HiddenPrints, loader


class ObjectDetectionBase:
    def __init__(self,
                 root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 classless: bool = False,
                 ):
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.datasets_info = datasets_info
    
        self.classless = classless
        self.cat_map, self.cats = self._build_category_map(self.root_dir, self.datasets_info)
       
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
        new_cat_id = 0

        for dataset, categories in dataset_cats.items():
            for cat in categories:
                if self.classless:
                    cat_map[dataset][cat["id"]] = 1
                    cat["id"] = 1
                    cat["name"] = "object"
                    cat["supercategory"] = "object"
                    cats[1] = cat
                else:
                    new_cat_id += 1
                    cat_map[dataset][cat["id"]] = new_cat_id
                    cat["id"] = new_cat_id
                    cats[new_cat_id] = cat

        return cat_map, cats
    
    def _get_target(self, data: Dict) -> Target:
        image = data["image"]
        image_ann = data["annotations"]
        image_id = image_ann["image_id"]
        annotations = image_ann["annotations"]
        
        w, h = image.size
        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor(image_id)

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
                base_transform: Callable | None = None,
                input_transform: Callable | None = None,
                data_folders: List[Dict[Literal["name", "ext"], str]] = [{"name": "images", "ext": "jpg"}],
                annotations_folders: List[str] = ["annotations"],
                data_loader: Callable[[str, Path], Any] = loader,
                classless: bool = False,
    ):
        FolderDataset.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            format_data=format_data,
            data_folders=data_folders,
            annotations_folders=annotations_folders,
            data_loader=data_loader,
        )
        ObjectDetectionBase.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            classless=classless,
        )
        self.base_transform = base_transform
        self.input_transform = input_transform
    
    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        target = self._get_target(data)
        image = data["images"]
        
        # Remap category ids
        new_labels = torch.zeros_like(target["labels"])
        for i, label in enumerate(target["labels"]):
            new_labels[i] = self.cat_map[image_data["dataset"]][label.item()]  # type: ignore[index]
        target["labels"] = new_labels
        
        if self.base_transform is not None:
            image, target = self.base_transform(image, target)
        if self.input_transform is not None:
            image, target = self.input_transform(image, target)
            
        target_set_dtype(target)
        return image, target

    def get_dataset_api(self, valid_categories: List[Dict] | None = None,
                        annotations_folder_name = "annotations") -> Tuple[COCO, Dict]:
        images, annotations, categories = [], [], []

        ann_id = 0
        for idx in range(len(self)):
            paths, data_info = self._get_data_paths(idx)
            ann_path = None
            for p in paths:
                if p.name == annotations_folder_name:
                    ann_path = p
                    break
            else:
                raise FileNotFoundError(f"Annotations folder not found in {paths}")

            with open(ann_path) as f:
                ann = json.load(f)
            
            img_ann = {
                "file_name": data_info["image_name"],
                "height": ann["height"],
                "width": ann["width"],
                "id": ann["image_id"],
            }
            images.append(img_ann)
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
                format_data: Callable[[List[dict]], Any] | None = None,
                drop_last: bool = False,
                shuffle: bool = True,
                base_transform: Callable | None = None,
                input_transform: Callable | None = None,
                classless: bool = False,
    ):
        ParquetDataset.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            batch_size=batch_size,
            format_data=format_data,
            drop_last=drop_last,
            shuffle=shuffle,
        )
        ObjectDetectionBase.__init__(
            self,
            root_dir=root_dir,
            datasets_info=datasets_info,
            classless=classless,
        )
        self.base_transform = base_transform
        self.input_transform = input_transform
    
    def _format_data(self, data: List[Dict]):
        targets = []
        images = []
        for d in data:
            d["image"] = Image.open(io.BytesIO(d["image"]))
            d["annotations"] = json.loads(d["annotations"])
            target = super()._get_target(d)

            image = d["image"]
            if self.base_transform is not None:
                image, target = self.base_transform(image, target)
            if self.input_transform is not None:
                image, target = self.input_transform(image, target)
                
            target_set_dtype(target)
            targets.append(target)
            images.append(image)
            
        return images, targets
